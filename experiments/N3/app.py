"""N3 interactive demo (local only, TOFU domain): shows the per-token rho trace, the firing SAE
features, the gate decision, and a dilution slider (benign filler prepended). Uses the A2 TOFU model
and its TOFU DSG features. Serve manually with `python -m experiments.N3.app` (127.0.0.1 only)."""
import numpy as np
import torch

from dsgx.attacks.transforms import filler

_STATE = {}


def load(weights="ckpt:A2/tofu_full", exp_id="N3"):
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    b = get_bundle(weights=resolve_weights(weights, exp_id))
    cache = ac.ActivationCache(ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0))
    feats = dsg.select_features(cache, 20, 95)
    _STATE.update(b=b, feats=feats, tau=dsg.calibrate_tau(cache, feats, 95))
    return _STATE


@torch.no_grad()
def analyze(question: str, pad_tokens: int = 0):
    b, feats, tau = _STATE["b"], _STATE["feats"], _STATE["tau"]
    body = (filler("wikitext", int(pad_tokens), "demo") + "\n\n" if pad_tokens else "") + question
    prompt = f"<bos><start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"
    t = b.model.to_tokens(prompt, prepend_bos=False).to(b.device)
    _, c = b.model.run_with_cache(t, stop_at_layer=b.layer + 1, names_filter=b.hook_name)
    a = b.sae.encode(c[b.hook_name])[0]
    a[0] = 0
    fire = (a[:, feats] > 0).any(1).float().cpu().numpy()
    running = np.cumsum(fire) / np.arange(1, len(fire) + 1)
    counts = (a[:, feats] > 0).sum(0).cpu().numpy()
    top = [{"feature": int(feats[i]), "tokens_fired": int(counts[i])}
           for i in np.argsort(-counts)[:8] if counts[i]]
    rho = float(fire.mean())
    return {"rho": rho, "tau": tau, "gate": "ON (clamp)" if rho > tau else "off",
            "n_tokens": int(len(fire)), "running_rho": running.tolist(), "firing_features": top}


def build_ui():
    import gradio as gr
    import pandas as pd

    with gr.Blocks(title="DSG gate explorer (TOFU, local)") as ui:
        q = gr.Textbox(label="Question about a (fictitious) TOFU author")
        pad = gr.Slider(0, 1600, value=0, step=50, label="Dilution: benign filler tokens")
        out = gr.JSON(label="Gate analysis")
        plot = gr.LinePlot(x="token", y="rho", label="Running rho")
        gr.Button("Analyze").click(
            lambda qq, pp: (lambda r: (r, pd.DataFrame({"token": range(len(r["running_rho"])),
                                                        "rho": r["running_rho"]})))(analyze(qq, pp)),
            [q, pad], [out, plot])
    return ui


def task(ctx):
    from datasets import load_dataset

    load(ctx.args.get("weights", "ckpt:A2/tofu_full"), ctx.exp_id)
    qs = load_dataset("locuslab/TOFU", "forget10", split="train").select(range(int(ctx.args.get("n", 10))))
    res = [{"pad": p, **{k: analyze(x["question"], p)[k] for k in ("rho", "gate")}}
           for x in qs for p in ctx.args.get("pads", [0, 400, 1600])]
    build_ui()
    ctx.write_metrics({"examples": res, "tau": _STATE["tau"], "ui_built": True, "served": False})
    ctx.finish({"view": "n3-demo", "forget": None})


if __name__ == "__main__":
    load()
    build_ui().launch(server_name="127.0.0.1", share=False)
