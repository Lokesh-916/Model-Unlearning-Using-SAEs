"""D2 closed-form null-space edit (AlphaEdit-style). Suppress the DSG selected-feature decoder
directions in the MLP down-projection (W_out) of layers <= L*, projecting the update onto the null
space of the retain-key covariance so retain activations move as little as possible. Baseline:
plain decoder orthogonalisation (no null-space projection). Saves an HF checkpoint for eval + A6."""
import json

import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.data.mcq import format_prompt, load_mcq
from dsgx.data.splits import get_split
from dsgx.methods import dsg
from dsgx.train.core import decoder_layers, load_hf


@torch.no_grad()
def retain_keys(model, tok, prompts, layer, max_rows=64):
    """MLP input activations (keys) at `layer` over benign prompts: [n, d_mlp_in]."""
    keys = []

    def hook(mod, inp, out):
        keys.append(inp[0].detach()[0].float().cpu())

    h = decoder_layers(model)[layer].mlp.down_proj.register_forward_hook(hook)
    for p in prompts[:max_rows]:
        model(tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device))
    h.remove()
    return torch.cat(keys)


def task(ctx):
    a = ctx.args
    case = a.get("case", "bio")
    Lstar = int(a.get("l_star", 3))
    mode = a.get("mode", "nullspace")  # nullspace | ortho
    from dsgx.models import loader

    b_sae = loader.get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b_sae, f"{case}-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, int(a.get("n_features", 20)), 95)
    W_dec = b_sae.sae.W_dec[torch.tensor(feats)].float().cpu().numpy()  # [k, d_model]
    # Free the TransformerLens model + SAE before loading the HF model (both won't fit in 16 GB).
    loader.clear()
    torch.cuda.empty_cache()
    model, tok = load_hf(dtype=torch.bfloat16)
    benign_subj = "high_school_geography"
    prompts = [format_prompt(load_mcq(benign_subj)[i]) for i in get_split(benign_subj, "dev")[:48]]
    ctx.progress.update(items_total=Lstar + 1, items_done=0, force=True)
    layers = decoder_layers(model)
    edited = []
    for L in range(Lstar + 1):
        U = torch.tensor(W_dec, dtype=torch.float32)
        U = U / U.norm(dim=1, keepdim=True)  # [k, d] unit decoder directions
        down = layers[L].mlp.down_proj.weight.data.float().cpu()  # [d, d_mlp], edit in fp32 on CPU
        if mode == "nullspace":
            K = retain_keys(model, tok, prompts, L).cpu()  # [n, d_mlp]
            C = (K.T @ K) / max(1, K.shape[0])
            evals, evecs = torch.linalg.eigh(C + 1e-3 * torch.eye(C.shape[0]))
            keep = evals < evals.max() * float(a.get("nullspace_eps", 0.05))
            P = evecs[:, keep] @ evecs[:, keep].T  # projector onto the retain null space [d_mlp, d_mlp]
        else:
            P = torch.eye(down.shape[1])
        proj_out = U.T @ U  # [d, d] projector onto span of the selected decoder directions
        down_new = (torch.eye(down.shape[0]) - proj_out) @ down
        delta = (down_new - down) @ P
        w = layers[L].mlp.down_proj.weight
        w.data = (down + delta).to(w.dtype).to(w.device)
        edited.append(L)
        ctx.progress.advance(1)
    out = ctx.cache_dir("models", ctx.exp_id, mode)
    model.save_pretrained(out); tok.save_pretrained(out)
    ctx.progress.advance(1)
    ctx.write_metrics({"mode": mode, "l_star": Lstar, "edited_layers": edited, "n_features": len(feats),
                       "features": [int(f) for f in feats], "checkpoint": str(out)})
    ctx.finish({"view": f"d2-edit:{mode}", "forget": None})
    return {"checkpoint": str(out)}
