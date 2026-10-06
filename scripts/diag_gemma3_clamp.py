"""A7 diagnosis (session 17): why DSG's clamp fires on Gemma 3 but changes (almost) no answers.

CPU only, benign sentences only (no WMDP text). Loads the model + SAE exactly as the harness does
(dsgx.models.loader.get_bundle), forces the clamp_all edit (selected features -> -500) and reports:
  resid             median ||resid|| at the SAE hook (non-BOS tokens)
  edit              median ||hooked out - resid|| at the clamped positions
  edit_expected     ||(a' - a) @ W_dec[f]||: equal to `edit` => the error term does not cancel the edit
  next_layer_delta  relative change of blocks.<L+1>.hook_resid_post => the hooked output feeds the next layer
  kl_last, top1_changed_last  effect on the next-token distribution at the last position
MODE all = clamp every non-BOS token; one = clamp only the middle token (like a gate that fires on one token).

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python scripts/diag_gemma3_clamp.py g3-1b float32 one
"""
import json
import sys

import torch

from dsgx.checks.sanity import LEGACY_FEATURES
from dsgx.models.loader import get_bundle

# selected features as recorded in the gpuws A7 DSG runs (method_info.features)
A7_1B = [16361, 14790, 7947, 4738, 5205, 2250, 3789, 3303, 6025, 7600, 3753, 16069, 5047, 14774, 5798, 12405, 9794,
         16267, 3764, 608]
CASES = {"g2": ("gemma-2-2b-it", "gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142", LEGACY_FEATURES),
         "g3-1b": ("gemma-3-1b-it", "gemma-scope-2-1b-it-res", "layer_13_width_16k_l0_medium", A7_1B),
         "g3-4b": ("gemma-3-4b-it", "gemma-scope-2-4b-it-res", "layer_17_width_16k_l0_medium", None)}
TEXT = ["The river flows through the old town, past the market square and the cathedral, before reaching the sea.",
        "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen.",
        "The committee met on Tuesday to discuss the budget for the new library and the park renovation."]
M = 500.0


def main(case, dtype="float32", mode="one", feats=None):
    name, rel, sid, default = CASES[case]
    feats = feats or default
    if feats is None:
        raise SystemExit("g3-4b: pass the A7-4b method_info.features as a JSON list (4th argument)")
    b = get_bundle(name, rel, sid, dtype, "cpu")
    m, sae = b.model, b.sae
    f = torch.tensor(feats)
    wd = sae.W_dec.float()
    nl = f"blocks.{b.layer + 1}.hook_resid_post"
    res = {k: [] for k in ("resid", "edit", "edit_expected", "next_layer_delta", "kl_last", "top1_changed_last")}
    for t in TEXT:
        toks = m.to_tokens(t)
        m.reset_hooks()
        logits0, c0 = m.run_with_cache(toks, names_filter=lambda n: n in (b.hook_name, nl))
        res["resid"].append(c0[b.hook_name][0, 1:].float().norm(dim=-1).median().item())
        cap = {}

        def hook(resid, hook=None):
            acts = sae.encode(resid)
            acts[:, 0, :] = 0.0
            err = resid - sae.decode(acts)
            a = acts.clone()
            mid = resid.shape[1] // 2
            pos = slice(1, None) if mode == "all" else slice(mid, mid + 1)
            a[0, pos, f] = -M
            out = sae.decode(a) + err
            cap["edit"] = (out - resid)[0, pos].float().norm(dim=-1).median().item()
            cap["edit_expected"] = ((a[0, pos, f] - acts[0, pos, f]).float() @ wd[f]).norm(dim=-1).median().item()
            return out

        m.add_hook(b.hook_name, hook)
        logits1, c1 = m.run_with_cache(toks, names_filter=lambda n: n == nl)
        m.reset_hooks()
        d = (c1[nl] - c0[nl])[0, 1:].float().norm(dim=-1) / c0[nl][0, 1:].float().norm(dim=-1)
        p0, p1 = logits0[0, -1].float().log_softmax(-1), logits1[0, -1].float().log_softmax(-1)
        res["next_layer_delta"].append(d.median().item())
        res["kl_last"].append((p0.exp() * (p0 - p1)).sum().item())
        res["top1_changed_last"].append(float(p0.argmax() != p1.argmax()))
        res["edit"].append(cap["edit"])
        res["edit_expected"].append(cap["edit_expected"])
    out = {k: round(sum(v) / len(v), 4) for k, v in res.items()}
    out.update(case=case, dtype=dtype, mode=mode, hook=b.hook_name, sae=type(sae).__name__,
               w_dec_norm=round(wd[f].norm(dim=1).mean().item(), 3), edit_over_resid=round(out["edit"] / out["resid"], 4))
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else "float32", a[2] if len(a) > 2 else "one", json.loads(a[3]) if len(a) > 3 else None)
