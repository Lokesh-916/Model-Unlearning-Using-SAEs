"""DSG-subset ids: items the BASE model answers correctly under all 24 answer permutations
(the 'correct' target metric of DSG), for every dataset (user decision 4). Task entry `task`.

Output: $DSG_CACHE/dsg_subset/<dataset>.json {"ids": [...], "n_items", "batch_size", ...}.
Batched evaluation of the unguarded model (no gate, so batching does not change semantics; bf16
batch numerics can flip borderline items, which is why agreement with the legacy bs=1 ids is
recorded). Legacy ids, where they exist, remain the ones used by the DSG-subset view.
"""
from itertools import permutations

import numpy as np

from dsgx import paths
from dsgx.util import atomic_write_json

PERMS = list(permutations(range(4)))


def task(ctx):
    """args: datasets (list or '@all'), batch_size, limit (items per dataset; smoke), case."""
    from dsgx.data.mcq import format_prompt, legacy_name, load_mcq
    from dsgx.data.splits import ALL_DATASETS
    from dsgx.eval.mcq_eval import score_prompts
    from dsgx.models.loader import get_bundle

    a = ctx.args
    ds_list = ALL_DATASETS if a.get("datasets", "@all") == "@all" else a["datasets"]
    bs = int(a.get("batch_size", 32))
    limit = a.get("limit")
    out_dir = paths.cache_dir() / ("dsg_subset" if not ctx.smoke else "dsg_subset_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in ds_list if not (out_dir / f"{d}.json").exists()]
    n_total = sum(min(len(load_mcq(d)), int(limit) if limit else 10 ** 9) for d in todo) * 24
    ctx.progress.update(items_total=n_total, items_done=0, force=True)
    b = get_bundle()
    summary = {}
    for d in todo:
        items = load_mcq(d)[: int(limit)] if limit else load_mcq(d)
        prompts, gold = [], []
        for it in items:
            for p in PERMS:
                prompts.append(format_prompt(it, permute=p))
                gold.append(p.index(it.answer))
        # sort by length for efficient batching, then restore order
        order = np.argsort([len(x) for x in prompts])
        base = ctx.progress.state["items_done"]
        probs, _, _ = score_prompts(b.model, [prompts[i] for i in order], bs, None,
                                    on_batch=lambda k: ctx.progress.update(items_done=base + k))
        pred = np.empty(len(prompts), dtype=int)
        pred[order] = probs.argmax(1)
        ok = (pred == np.array(gold)).reshape(len(items), 24).all(1)
        ids = [int(i) for i in np.where(ok)[0]]
        rec = {"dataset": d, "ids": ids, "n_items": len(items), "n_correct_all24": len(ids),
               "batch_size": bs, "smoke": ctx.smoke}
        for case in ("bio", "cyber"):
            leg = paths.legacy_artifacts(case) / "data" / "question_ids" / "all" / f"{legacy_name(d)}_correct.csv"
            if leg.exists() and not limit:
                L = set(np.atleast_1d(np.genfromtxt(leg, dtype=int)).tolist())
                rec[f"legacy_{case}"] = {"n": len(L), "jaccard": len(L & set(ids)) / max(1, len(L | set(ids)))}
        atomic_write_json(out_dir / f"{d}.json", rec)
        summary[d] = len(ids)
    ctx.write_metrics({"datasets": summary, "out_dir": str(out_dir)})
    ctx.finish({"view": "dsg-subset-ids", "forget": None})
    return summary
