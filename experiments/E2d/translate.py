"""E2d translate task (GPU): WMDP-Bio DEV items into Hindi with NLLB-200-distilled-600M + back-translation chrF,
the same pipeline and cache as B3 (dsgx.data.translate; WMDP translations live only under $DSG_PRIVATE).
B3 translated the TEST split only; dev records are appended (existing records are never rewritten, and items
already cached are skipped, so a restart resumes). Counts only are logged."""
from dsgx.data.mcq import load_mcq
from dsgx.data.splits import get_split
from dsgx.data.translate import Translator, load_cache, translate_items


def task(ctx):
    a = ctx.args
    ds, lang, split = a.get("dataset", "wmdp-bio"), a.get("lang", "hi"), a.get("split", "dev")
    if split != "dev":
        raise ValueError("E2d is DEV only")
    ids = get_split(ds, split)
    if a.get("limit"):
        ids = ids[: int(a["limit"])]
    items = load_mcq(ds)
    todo = [items[i] for i in ids]
    ctx.write_config()
    ctx.progress.update(items_total=len(todo), items_done=0, force=True)
    tr = Translator(batch_size=int(a.get("batch_size", 16)))
    new = translate_items(todo, lang, tr, progress=lambda k: ctx.progress.advance(k))
    cache = load_cache(ds, lang)
    have = [cache[it.item_id] for it in todo if it.item_id in cache]
    kept = sum(r["chrf"] >= float(a.get("min_chrf", 40)) for r in have)
    m = {"dataset": ds, "lang": lang, "split": split, "n_items": len(todo), "new_records": new,
         "cached": len(have), "chrf_ge_min": int(kept), "min_chrf": float(a.get("min_chrf", 40))}
    ctx.write_metrics(m)
    ctx.finish({"view": "translation-cache", "forget": None, "kept": int(kept)})
    return m
