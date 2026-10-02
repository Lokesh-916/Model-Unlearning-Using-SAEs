"""N10 Audit Card (should): a template plus auto-filled cards for DSG and the best fixes, generated
from $DSG_RESULTS/summary.json if present, else from whatever run metrics exist. Markdown + JSON per
method under $DSG_RESULTS/audit_cards/."""
import json

from dsgx import paths

TEMPLATE = """# Audit Card: {name}

| Field | Value |
|---|---|
| Method | {name} |
| Model / SAE | {model} |
| Forget accuracy (DSG-subset test) | {forget} |
| Utility (full MMLU, pooled) | {utility} |
| Benign FPR (target 5%) | {fpr} |
| Hazard FNR | {fnr} |
| Attack success, dilution (max) | {dilution} |
| Attack success, cross-lingual (max) | {crosslingual} |
| Knowledge retained internally (probe acc, best layer) | {probe} |
| Resists LoRA relearning | {relearn} |
| Latency overhead / token | {latency} |
| Source runs | {runs} |

_Generated {date} from {source}. Numbers carry n and 95% CI in the linked run metrics. Not published._
"""


def _fmt(ci):
    if not isinstance(ci, dict) or ci.get("mean") is None:
        return "n/a"
    lo, hi, n = ci.get("lo"), ci.get("hi"), ci.get("n")
    return f"{ci['mean']:.3f} [{lo:.3f}, {hi:.3f}] (n={n})" if lo is not None else f"{ci['mean']:.3f} (n={n})"


def task(ctx):
    from dsgx.util import now_iso

    sj = paths.results_dir() / "summary.json"
    summary = json.loads(sj.read_text()) if sj.exists() else {}
    methods = ctx.args.get("methods") or list((summary.get("methods") or {}).keys()) or ["dsg-faithful"]
    out = paths.results_dir() / ("audit_cards_smoke" if ctx.smoke else "audit_cards")
    out.mkdir(parents=True, exist_ok=True)
    cards = {}
    for name in methods:
        m = (summary.get("methods") or {}).get(name, {})
        card = {"name": name, "model": m.get("model", "gemma-2-2b-it / gemma-scope-2b-pt-res L3"),
                "forget": _fmt(m.get("forget")), "utility": _fmt(m.get("utility")),
                "fpr": _fmt(m.get("benign_fpr")), "fnr": _fmt(m.get("hazard_fnr")),
                "dilution": _fmt(m.get("attack_dilution")), "crosslingual": _fmt(m.get("attack_crosslingual")),
                "probe": _fmt(m.get("probe_best")), "relearn": m.get("relearn_verdict", "n/a"),
                "latency": m.get("latency_overhead", "n/a"), "runs": ", ".join(m.get("runs", []) or ["(pending)"]),
                "date": now_iso(), "source": "summary.json" if summary else "(no summary.json yet; template only)"}
        (out / f"{name}.md").write_text(TEMPLATE.format(**card))
        (out / f"{name}.json").write_text(json.dumps(card, indent=1))
        cards[name] = str(out / f"{name}.md")
    ctx.write_metrics({"cards": cards, "from_summary": bool(summary), "out": str(out)})
    ctx.finish({"view": "n10-audit-cards", "forget": None})
    return cards
