"""Q5 representation geometry: what does the gate change in the residual stream, layer by layer?

    python -m dsgx.analysis.qual.q5_geometry [--exp A4] [--smoke] [--out DIR] [--layer L]

For every captured condition (A4: base, dsg; D3: base, d1, d2, ...) versus base, per layer:
  CKA(base, cond)                 global similarity of the item representations;
  mean cosine(base_i, cond_i)     per-item direction change;
  answer separability             Fisher ratio of the gold-letter classes (knowledge present = separable);
  forget/benign axis              cosine between the base and cond forget-minus-benign mean directions and
                                  the norm ratio (if benign items were captured).
Writes Q5_GEOMETRY.md, q5_geometry.json, q5_geometry.{png,pdf} (CKA / separability by layer) and
q5_pca_L<layer>.{png,pdf} (2-D PCA at one layer, coloured by condition; points are items, no text).
"""
import argparse

import numpy as np

from dsgx.analysis.qual import geometry as G
from dsgx.analysis.qual import out_dir
from dsgx.util import atomic_write_json, atomic_write_text


def analyse(caps: dict) -> dict:
    if "base" not in caps:
        return {}
    base = caps["base"]
    nL = base["X"].shape[1]
    yb = np.asarray(base["meta"]["gold"])
    fb = G.is_forget(base["meta"])
    res = {}
    for tag, cap in caps.items():
        if cap["X"].shape[:2] != base["X"].shape[:2] or cap["meta"]["items"] != base["meta"]["items"]:
            res[tag] = {"error": "item sets differ from base"}
            continue
        rows = []
        for L in range(nL):
            Xb, Xc = G.layer_matrix(base, L), G.layer_matrix(cap, L)
            r = {"layer": L, "cka": G.linear_cka(Xb, Xc), "mean_cos": G.mean_cos(Xb, Xc),
                 "fisher_answer": G.fisher_ratio(Xc, yb)}
            if fb.any() and (~fb).any():
                db = Xb[fb].mean(0) - Xb[~fb].mean(0)
                dc = Xc[fb].mean(0) - Xc[~fb].mean(0)
                r["forget_axis_cos"] = float(db @ dc / (np.linalg.norm(db) * np.linalg.norm(dc) + 1e-9))
                r["forget_axis_norm_ratio"] = float(np.linalg.norm(dc) / (np.linalg.norm(db) + 1e-9))
            rows.append(r)
        res[tag] = rows
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=["A4", "D3"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer", type=int, default=None, help="PCA layer (default: largest base-vs-dsg change)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir(a.out)
    from dsgx.analysis.figures import PALETTE, MARKERS, save, setup

    allres, figs = {}, []
    L_md = ["# Q5 — representation geometry (residual stream, last prompt token)", ""]
    for exp in a.exp:
        e = exp + ("-smoke" if a.smoke else "")
        caps = G.captures(e)
        res = analyse(caps)
        if not res:
            L_md.append(f"- {e}: no captures with a base condition")
            continue
        allres[e] = res
        plt = setup()
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for i, (tag, rows) in enumerate((t, r) for t, r in res.items() if isinstance(r, list)):
            x = [r["layer"] for r in rows]
            st = {"color": PALETTE[i % 8], "marker": MARKERS[i % 8], "markersize": 3}
            axs[0].plot(x, [r["cka"] for r in rows], label=tag, **st)
            axs[1].plot(x, [r["fisher_answer"] for r in rows], label=tag, **st)
        axs[0].set_title("CKA vs base")
        axs[0].set_xlabel("layer")
        axs[1].set_title("answer separability (Fisher ratio)")
        axs[1].set_xlabel("layer")
        axs[1].legend(fontsize=7)
        figs.append(save(fig, out, f"q5_geometry_{e}"))
        other = [t for t in res if t != "base" and isinstance(res[t], list)]
        if other:
            rows = res[other[0]]
            L = a.layer if a.layer is not None else int(np.argmin([r["mean_cos"] for r in rows]))
            fig, ax = plt.subplots(figsize=(4.2, 3.8))
            X = np.concatenate([G.layer_matrix(caps[t], L) for t in ["base"] + other])
            P = G.pca2(X)
            n = caps["base"]["X"].shape[0]
            f = G.is_forget(caps["base"]["meta"])
            for i, t in enumerate(["base"] + other):
                Pi = P[i * n:(i + 1) * n]
                ax.scatter(Pi[f, 0], Pi[f, 1], s=10, color=PALETTE[i % 8], marker="o", label=f"{t} forget")
                if (~f).any():
                    ax.scatter(Pi[~f, 0], Pi[~f, 1], s=10, color=PALETTE[i % 8], marker="x", label=f"{t} benign")
            ax.set_title(f"{e}: PCA of layer {L}")
            ax.legend(fontsize=6)
            figs.append(save(fig, out, f"q5_pca_{e}_L{L}"))
        L_md += [f"## {e}", "", "| condition | min CKA (layer) | min mean cos (layer) | max answer Fisher (layer) |", "|---|---|---|---|"]
        for t, rows in res.items():
            if not isinstance(rows, list):
                L_md.append(f"| {t} | {rows.get('error')} | | |")
                continue
            c = min(rows, key=lambda r: r["cka"])
            m = min(rows, key=lambda r: r["mean_cos"])
            fz = max(rows, key=lambda r: r["fisher_answer"] if r["fisher_answer"] == r["fisher_answer"] else -1)
            L_md.append(f"| {t} | {c['cka']:.3f} (L{c['layer']}) | {m['mean_cos']:.3f} (L{m['layer']}) | "
                        f"{fz['fisher_answer']:.4f} (L{fz['layer']}) |")
        L_md.append("")
    atomic_write_text(out / "Q5_GEOMETRY.md", "\n".join(L_md) + "\n")
    atomic_write_json(out / "q5_geometry.json", {"results": allres, "figures": figs})
    print(f"wrote {out / 'Q5_GEOMETRY.md'} ({len(allres)} experiment(s), {len(figs)} figure(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
