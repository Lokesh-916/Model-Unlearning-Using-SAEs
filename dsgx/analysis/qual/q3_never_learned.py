"""Q3: does DSG look like a model that NEVER learned the forget set? Compare every TOFU condition with the
retain-only ("never learned") model on the forget metrics, from the logged TOFU runs.

    python -m dsgx.analysis.qual.q3_never_learned [--smoke] [--runs ROOT] [--out DIR]

Sources (whatever exists): A2/tofu-metrics (lab PC, LoRA models) and A2-tofu-full/tofu-metrics (gpuws, full
fine-tune; separate hardware: use --runs on dsg_results_cluster/runs), A5/mia (membership inference, gated vs
ungated). Output: Q3_NEVER_LEARNED.md, q3_never_learned.json, q3_truth_ratio.{png,pdf}.
Distance to the never-learned model = |metric(condition) - metric(retain model)| on the forget set; the
TOFU forget-quality KS p-value is the canonical test (high p = indistinguishable from never learned).
"""
import argparse
import json

import numpy as np

from dsgx.analysis.collect import fmt_ci, load_all
from dsgx.analysis.qual import out_dir
from dsgx.util import atomic_write_json, atomic_write_text

KEYS = ["truth_ratio_forget", "retain_answer_prob", "model_utility", "forget_quality_ks_p"]


def _val(x):
    return x.get("mean") if isinstance(x, dict) else x


def normalise(conds: dict) -> dict:
    """exp/A2 (flat keys) and tofu-full (nested per split) -> flat {cond: {metric: value|ci}}."""
    out = {}
    for c, v in conds.items():
        if "forget" in v and isinstance(v["forget"], dict) and "truth_ratio" in v["forget"]:
            out[c] = {"truth_ratio_forget": v["forget"]["truth_ratio"], "forget_answer_prob": v["forget"]["answer_prob"],
                      "forget_rougeL_recall": v["forget"]["rougeL_recall"], "model_utility": v.get("model_utility"),
                      "forget_quality_ks_p": v.get("forget_quality_ks_p"),
                      "truth_ratio_values": v["forget"].get("truth_ratio_values")}
        else:
            out[c] = {k: v.get(k) for k in ("truth_ratio_forget", "truth_ratio_retain", "retain_answer_prob",
                                            "model_utility", "forget_quality_ks_p")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir(a.out)
    runs = load_all(a.runs, smoke=a.smoke)
    srcs = {r.base_exp: normalise(r.metrics.get("conditions", {})) for r in runs
            if r.name == "tofu-metrics" and r.base_exp in ("A2", "A2-tofu-full")}
    mia = next((r.metrics for r in runs if r.base_exp == "A5" and r.name == "mia"), None)
    L = ["# Q3 — DSG vs the never-learned (retain-only) model", ""]
    res = {}
    for exp, conds in srcs.items():
        ref = conds.get("retain-model") or {}
        L += [f"## {exp}", "", "| condition | forget truth ratio | |Δ| to never-learned | forget quality (KS p) | model utility |",
              "|---|---|---|---|---|"]
        res[exp] = {}
        for c, v in conds.items():
            tr, rtr = _val(v.get("truth_ratio_forget")), _val(ref.get("truth_ratio_forget"))
            d = abs(tr - rtr) if tr is not None and rtr is not None else None
            res[exp][c] = {"truth_ratio_forget": tr, "delta_to_never_learned": d, "ks_p": v.get("forget_quality_ks_p"),
                           "model_utility": v.get("model_utility")}
            L.append(f"| {c} | {fmt_ci(v.get('truth_ratio_forget')) if isinstance(v.get('truth_ratio_forget'), dict) else tr} | "
                     f"{'' if d is None else f'{d:.3f}'} | {v.get('forget_quality_ks_p')} | {v.get('model_utility')} |")
        best = min((c for c in res[exp] if c != "retain-model" and res[exp][c]["delta_to_never_learned"] is not None),
                   key=lambda c: res[exp][c]["delta_to_never_learned"], default=None)
        L += ["", f"Closest to never-learned: **{best}**." if best else "", ""]
    if mia:
        L += ["## A5 membership inference (AUROC; 0.5 = indistinguishable)", "", "```", json.dumps(mia.get("auroc"), indent=1), "```", ""]
        res["A5_mia"] = mia.get("auroc")
    if not srcs:
        L.append("_No TOFU metrics yet (A2 tofu-metrics / A2-tofu-full)._")
    figs = []
    try:
        from dsgx.analysis.figures import PALETTE, save, setup

        plt = setup()
        for exp, conds in srcs.items():
            vals = {c: v.get("truth_ratio_values") for c, v in conds.items() if v.get("truth_ratio_values")}
            if not vals:
                continue
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for i, (c, x) in enumerate(vals.items()):
                ax.hist(np.log10(np.clip(x, 1e-6, None)), bins=30, alpha=0.5, color=PALETTE[i % 8], label=c)
            ax.set_xlabel("log10 truth ratio (forget set)")
            ax.set_ylabel("items")
            ax.set_title(f"Q3 {exp}: truth-ratio distributions vs never-learned")
            ax.legend(fontsize=7)
            figs.append(save(fig, out, f"q3_truth_ratio_{exp}"))
    except Exception as e:
        L.append(f"_figure skipped: {e}_")
    atomic_write_text(out / "Q3_NEVER_LEARNED.md", "\n".join(L) + "\n")
    atomic_write_json(out / "q3_never_learned.json", {"results": res, "figures": figs})
    print(f"wrote {out / 'Q3_NEVER_LEARNED.md'} ({len(srcs)} TOFU source(s), {len(figs)} figure(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
