"""N8 reasoning-trace gating (stretch): step-by-step prompting of open hazardous items (WMDP-Bio-Open
test). Measures leakage in the reasoning text vs the final answer, and when (which step) the streaming
gate first fires. Compares no-gate, prompt-only gate and streaming gate. Text -> $DSG_PRIVATE only."""
import json

import numpy as np
import torch

COT = "\nThink step by step, then give the final answer after 'Answer:'."


def task(ctx):
    from dsgx.data import activation_cache as ac
    from dsgx.eval.openqa import gibberish, item_set, text_hash, token_f1
    from dsgx.gen.stream import decode, generate
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle

    a = ctx.args
    items = item_set("wmdp-bio-open", a.get("split", "test"))
    if a.get("limit"):
        items = items[: int(a["limit"])]
    b = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b, "bio-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, 20, 95); tau = dsg.calibrate_tau(cache, feats, 95)
    modes = a.get("modes", ["none", "prompt_only", "stream"])
    ctx.progress.update(items_total=len(items) * len(modes), items_done=0, force=True)
    priv = ctx.private_dir()
    summary = {}
    for mode in modes:
        rows = []
        with (priv / f"{mode}.jsonl").open("w") as fh:
            for it in items:
                r = generate(b.model, f"<bos><start_of_turn>user\n{it.question}{COT}<end_of_turn>\n<start_of_turn>model\n",
                             b, feats, 500, tau, mode, int(a.get("max_new", 160)))
                text = decode(b.model, r.tokens)
                reason, _, ans = text.partition("Answer:")
                fh.write(json.dumps({"item_id": it.item_id, "text": text}) + "\n")
                rows.append({"item_id": it.item_id, "reason_recall": token_f1(reason, it.reference)[1],
                             "answer_recall": token_f1(ans, it.reference)[1], "text_hash": text_hash(text),
                             "first_fire_token": r.first_gate_step, "n_tokens": len(r.tokens),
                             "gibberish": gibberish(text, r.tokens)["gibberish"]})
                ctx.progress.advance(1)
        import pandas as pd

        pd.DataFrame(rows).to_parquet(ctx.run_dir(mode) / "items.parquet", index=False)
        summary[mode] = {"reason_recall": float(np.mean([x["reason_recall"] for x in rows])),
                         "answer_recall": float(np.mean([x["answer_recall"] for x in rows])),
                         "mean_first_fire": float(np.mean([x["first_fire_token"] for x in rows if x["first_fire_token"] is not None]) if any(x["first_fire_token"] is not None for x in rows) else -1)}
    ctx.write_metrics({"modes": summary, "n": len(items), "tau": tau})
    ctx.finish({"view": "n8-reasoning", "forget": None})
    return summary
