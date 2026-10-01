"""B3 translation cache task: translate forget + utility test items into 8 languages with NLLB,
back-translation chrF filter, persistent cache (WMDP -> $DSG_PRIVATE). GPU, CPU-light otherwise."""
from dsgx.data.mcq import load_mcq
from dsgx.data.splits import get_split
from dsgx.data.translate import LANGS, Translator, translate_items


def task(ctx):
    a = ctx.args
    datasets = a.get("datasets", ["wmdp-bio"])
    langs = a.get("langs", list(LANGS))
    split = a.get("split", "test")
    limit = a.get("limit")
    tr = Translator(batch_size=int(a.get("batch_size", 16)))
    items_by_ds = {}
    for d in datasets:
        ids = get_split(d, split)
        if limit:
            ids = ids[: int(limit)]
        allit = load_mcq(d)
        items_by_ds[d] = [allit[i] for i in ids]
    total = sum(len(v) for v in items_by_ds.values()) * len(langs)
    ctx.progress.update(items_total=total, items_done=0, force=True)
    out = {}
    for lang in langs:
        n = 0
        for d, items in items_by_ds.items():
            n += translate_items(items, lang, tr, progress=lambda k: ctx.progress.advance(k))
        out[lang] = n
    ctx.write_metrics({"datasets": datasets, "langs": langs, "new_records": out})
    ctx.finish({"view": "translation-cache", "forget": None})
    return out
