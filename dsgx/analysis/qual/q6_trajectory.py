"""Q6 layer-by-layer trajectory figures: where along the depth does the answer appear, and what does the
gate (layer 3) change downstream?

    python -m dsgx.analysis.qual.q6_trajectory [--smoke] [--out DIR]

Panels per experiment (A4, D3), one line per captured condition:
  (a) logit-lens answer accuracy by layer (capture metrics),
  (b) linear-probe TEST accuracy by layer with the control task (probe metrics),
  (c) mean cosine between the condition's and base's residuals by layer, forget items vs benign items
      (residual captures) - the gate's footprint propagating through the stack.
Writes q6_trajectory_<exp>.{png,pdf}, Q6_TRAJECTORY.md and q6_trajectory.json.
"""
import argparse

import numpy as np

from dsgx.analysis.collect import load_all
from dsgx.analysis.qual import geometry as G
from dsgx.analysis.qual import out_dir
from dsgx.util import atomic_write_json, atomic_write_text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=["A4", "D3"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir(a.out)
    from dsgx.analysis.figures import INK2, PALETTE, MARKERS, save, setup

    runs = load_all(smoke=a.smoke, exps=a.exp)
    md, res, figs = ["# Q6 — layer-by-layer trajectories", ""], {}, []
    for exp in a.exp:
        cap_runs = {r.name.split("__", 1)[1]: r for r in runs if r.base_exp == exp and r.name.startswith("capture__")}
        probe_runs = {r.name.split("__", 1)[1]: r for r in runs if r.base_exp == exp and r.name.startswith("probe__")}
        caps = G.captures(exp + ("-smoke" if a.smoke else ""))
        if not (cap_runs or probe_runs or caps):
            md.append(f"- {exp}: nothing captured yet")
            continue
        plt = setup()
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
        r_exp = {}
        for i, tag in enumerate(sorted(set(cap_runs) | set(probe_runs) | set(caps))):
            st = {"color": PALETTE[i % 8], "marker": MARKERS[i % 8], "markersize": 3}
            row = {}
            if tag in cap_runs and cap_runs[tag].metrics.get("logit_lens_acc_by_layer"):
                ll = cap_runs[tag].metrics["logit_lens_acc_by_layer"]
                axs[0].plot(range(len(ll)), ll, label=tag, **st)
                row["logit_lens_best"] = (int(np.argmax(ll)), float(max(ll)))
            if tag in probe_runs and probe_runs[tag].metrics.get("layers"):
                lay = probe_runs[tag].metrics["layers"]
                Ls = sorted(int(k) for k in lay)
                axs[1].plot(Ls, [lay[str(l)]["probe"]["mean"] for l in Ls], label=tag, **st)
                axs[1].plot(Ls, [lay[str(l)]["control"]["mean"] for l in Ls], color=st["color"], linestyle=":", linewidth=1)
                row["probe_best"] = max(((l, lay[str(l)]["probe"]["mean"]) for l in Ls), key=lambda t: t[1])
            if tag in caps and tag != "base" and "base" in caps and caps[tag]["X"].shape == caps["base"]["X"].shape:
                f = G.is_forget(caps["base"]["meta"])
                nL = caps["base"]["X"].shape[1]
                cf = [G.mean_cos(G.layer_matrix(caps["base"], L)[f], G.layer_matrix(caps[tag], L)[f]) for L in range(nL)] if f.any() else []
                cb = [G.mean_cos(G.layer_matrix(caps["base"], L)[~f], G.layer_matrix(caps[tag], L)[~f]) for L in range(nL)] if (~f).any() else []
                if cf:
                    axs[2].plot(range(nL), cf, label=f"{tag} forget", **st)
                if cb:
                    axs[2].plot(range(nL), cb, label=f"{tag} benign", color=st["color"], linestyle="--", linewidth=1.2)
                row["cos_forget_min"] = float(min(cf)) if cf else None
                row["cos_benign_min"] = float(min(cb)) if cb else None
            r_exp[tag] = row
        for ax, t in zip(axs, ["logit-lens answer accuracy", "probe accuracy (dotted: control)", "cos(base, condition)"]):
            ax.set_title(t)
            ax.set_xlabel("layer")
            ax.axvline(3, color=INK2, linewidth=0.8, linestyle=":")
        axs[0].legend(fontsize=6)
        axs[2].legend(fontsize=6)
        figs.append(save(fig, out, f"q6_trajectory_{exp}"))
        res[exp] = r_exp
        md += [f"## {exp}", "", "| condition | logit lens best (layer, acc) | probe best (layer, acc) | min cos forget | min cos benign |",
               "|---|---|---|---|---|"]
        for t, r in r_exp.items():
            md.append(f"| {t} | {r.get('logit_lens_best', '')} | {r.get('probe_best', '')} | {r.get('cos_forget_min', '')} | {r.get('cos_benign_min', '')} |")
        md += ["", "Dotted vertical line: layer 3 (the DSG hook).", ""]
    atomic_write_text(out / "Q6_TRAJECTORY.md", "\n".join(md) + "\n")
    atomic_write_json(out / "q6_trajectory.json", {"results": res, "figures": figs})
    print(f"wrote {out / 'Q6_TRAJECTORY.md'} ({len(res)} experiment(s), {len(figs)} figure(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
