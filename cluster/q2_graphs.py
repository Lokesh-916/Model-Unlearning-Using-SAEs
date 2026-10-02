"""Q2: attribution graphs (circuit-tracer 0.5.0) for WMDP-Bio and benign MMLU items on gemma-2-2b-it.

Question: is the knowledge DSG gates on computed THROUGH the layer-3 features DSG selects, or around
them? For each item: attribute(prompt -> top answer logits) with the Gemma Scope transcoders
(transcoder set "gemma" = mwhanna/gemma-scope-transcoders, trained on gemma-2-2b PT; used here on the IT
model exactly as DSG uses PT SAEs on the IT model), prune (node 0.8 / edge 0.98), and record
  * graph replacement / completeness scores;
  * influence share of each layer's feature nodes on the logits (pruned graph);
  * DSG overlap: the 20 DSG residual-SAE features (layer 3) are matched to layer-3 transcoder features by
    decoder cosine (>= MATCH_COS); share of logit influence carried by matched nodes, and whether any
    matched node survives pruning.
Environment: a separate venv env/q2 (system site packages of mechunlearn2 + the 29 overlay wheels in
wheels/q2: circuit-tracer 0.5.0 needs transformers <= 4.57.3). Created by q2_graphs.sbatch, offline.
Graphs contain prompt tokens (hazardous for WMDP): only --keep-graphs per dataset are saved, ONLY under
$DSG_PRIVATE/jobs/q2-graphs/, never fetched; results hold item ids, hashes and metrics.
"""
import argparse
import json

import numpy as np
import torch

NAME = "q2-graphs"
MATCH_COS = 0.5


def layer_of_nodes(active_features, n_layers, n_pos, n_logits):
    """Layer index per node in circuit-tracer order: features, errors (layer-major), tokens (-1), logits (n_layers)."""
    nf = active_features.shape[0]
    lay = [int(x) for x in active_features[:, 0].tolist()]
    lay += [l for l in range(n_layers) for _ in range(n_pos)]
    lay += [-1] * n_pos + [n_layers] * n_logits
    return np.array(lay), nf


def graph_metrics(adjacency, active_features, n_layers, n_pos, logit_probs, node_mask=None, matched=None):
    """Influence of every node on the logits (probability-weighted), aggregated per layer and for the
    DSG-matched feature nodes. adjacency[target, source] = direct effect (circuit-tracer convention).
    matched: bool per active feature (True = equivalent of a DSG feature)."""
    A = np.abs(np.asarray(adjacency, dtype=np.float64))
    n = A.shape[0]
    n_logits = len(logit_probs)
    lay, nf = layer_of_nodes(np.asarray(active_features), n_layers, n_pos, n_logits)
    assert len(lay) == n, (len(lay), n)
    rs = A.sum(1, keepdims=True)
    An = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    w = np.zeros(n)
    w[n - n_logits:] = np.asarray(logit_probs, dtype=np.float64)
    infl, cur = np.zeros(n), w.copy()
    for _ in range(n_layers + 2):  # propagate back through the DAG
        cur = cur @ An
        infl += cur
        if cur.sum() < 1e-9:
            break
    keep = np.ones(n, bool) if node_mask is None else np.asarray(node_mask, bool)
    feat = np.zeros(n, bool)
    feat[:nf] = True
    tot = infl[: n - n_logits].sum() or 1.0
    per_layer = {int(L): float(infl[(lay == L) & feat & keep].sum() / tot) for L in range(n_layers)}
    out = {"n_nodes": int(n), "n_features": int(nf), "n_kept": int(keep.sum()), "feature_influence_share": float(infl[feat].sum() / tot),
           "error_influence_share": float(infl[(~feat) & (lay >= 0) & (lay < n_layers)].sum() / tot),
           "token_influence_share": float(infl[lay == -1].sum() / tot), "per_layer_share": per_layer,
           "share_layers_le3": float(sum(v for k, v in per_layer.items() if k <= 3))}
    if matched is not None:
        m = np.zeros(n, bool)
        m[:nf] = np.asarray(matched, bool)
        out["dsg_matched_nodes"] = int(m.sum())
        out["dsg_matched_kept"] = int((m & keep).sum())
        out["dsg_influence_share"] = float(infl[m].sum() / tot)
    return out


def dsg_matches(sae_wdec, tc_wdec, feats, thr=MATCH_COS):
    """{dsg feature -> (best transcoder feature, cosine)} and the matched transcoder ids (cos >= thr)."""
    a = torch.nn.functional.normalize(torch.as_tensor(sae_wdec[feats], dtype=torch.float32), dim=-1)
    b = torch.nn.functional.normalize(torch.as_tensor(tc_wdec, dtype=torch.float32), dim=-1)
    cos = a @ b.T
    best = cos.max(1)
    pairs = {int(f): (int(i), float(c)) for f, i, c in zip(feats, best.indices.tolist(), best.values.tolist())}
    return pairs, sorted({i for i, c in pairs.values() if c >= thr})


def _sae_wdec():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("google/gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142/params.npz")
    return np.load(p)["W_dec"]


def _tc_wdec(model, layer=3):
    tc = model.transcoders
    for get in (lambda: tc[layer].W_dec, lambda: tc.W_dec[layer], lambda: tc.transcoders[layer].W_dec):
        try:
            w = get()
            return w.detach().float().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
        except Exception:
            continue
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="WMDP-Bio TEST items (and the same number of benign items)")
    ap.add_argument("--benign", default="high_school_geography")
    ap.add_argument("--max-feature-nodes", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--keep-graphs", type=int, default=3, help="graphs saved (private) per dataset, for case studies")
    ap.add_argument("--plan", action="store_true", help="imports + item plan only (CPU)")
    a = ap.parse_args(argv)
    from cluster import jobcommon as jc
    from dsgx import paths
    from dsgx.data.mcq import format_prompt
    from dsgx.util import atomic_write_json

    items = [("wmdp-bio", it) for it in jc.mcq_items("wmdp-bio", "test", a.n)] + \
            [(a.benign, it) for it in jc.mcq_items(a.benign, "test", a.n)]
    feats, tau = jc.dsg_features() if not a.plan else (list(range(20)), None)
    jc.log(NAME, f"{len(items)} items; DSG features {len(feats)}")
    if a.plan:
        import circuit_tracer  # noqa: F401  (the overlay env must import)

        print(json.dumps({"n_items": len(items), "circuit_tracer": circuit_tracer.__name__}))
        return 0
    jc.require_gpu(30)
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_graph_scores, prune_graph

    model = ReplacementModel.from_pretrained("google/gemma-2-2b-it", "gemma", dtype=torch.bfloat16)
    tcw = _tc_wdec(model)
    pairs, matched_ids = dsg_matches(_sae_wdec(), tcw, feats) if tcw is not None else ({}, [])
    priv = jc.private_dir(NAME)
    out_dir = paths.runs_dir() / "Q2-graphs"
    rows, kept = [], {}
    for ds, it in items:
        rec_p = out_dir / "items" / f"{ds}__{it.item_id}.json"
        if rec_p.exists():
            rows.append(json.loads(rec_p.read_text()))
            continue
        prompt = format_prompt(it)
        g = attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.95, batch_size=a.batch_size,
                      max_feature_nodes=a.max_feature_nodes, offload="cpu")
        if kept.get(ds, 0) < a.keep_graphs:  # ~250 MB each: keep only a few case studies
            g.to_pt(str(priv / f"{ds}__{it.item_id}.pt"))
            kept[ds] = kept.get(ds, 0) + 1
        pr = prune_graph(g, 0.8, 0.98)
        rep, comp = compute_graph_scores(g)
        af = g.selected_features if hasattr(g, "selected_features") else None
        act = g.active_features[af] if af is not None else g.active_features
        m = None
        if matched_ids:
            m = ((act[:, 0] == 3) & torch.isin(act[:, 2], torch.tensor(matched_ids))).cpu().numpy()
        met = graph_metrics(g.adjacency_matrix.float().cpu().numpy(), act.cpu().numpy(), model.cfg.n_layers,
                            len(g.input_tokens), g.logit_probabilities.float().cpu().numpy(),
                            pr.node_mask.cpu().numpy(), m)
        rec = {"dataset": ds, "item_id": int(it.item_id), "prompt_hash": jc.text_hash(prompt), "replacement_score": float(rep),
               "completeness_score": float(comp), **met}
        atomic_write_json(rec_p, rec)
        rows.append(rec)
        del g
        torch.cuda.empty_cache()

    def agg(ds, k):
        v = [r[k] for r in rows if r["dataset"] == ds and r.get(k) is not None]
        from dsgx.eval import stats

        return stats.bootstrap_ci(v) if v else None

    res = {ds: {k: agg(ds, k) for k in ("replacement_score", "completeness_score", "share_layers_le3", "dsg_influence_share",
                                        "feature_influence_share")} for ds in ("wmdp-bio", a.benign)}
    atomic_write_json(out_dir / "graphs" / "metrics.json", {"per_dataset": res, "dsg_matches": pairs, "matched_ids": matched_ids,
                                                            "match_cos": MATCH_COS, "n": len(rows)})
    atomic_write_json(out_dir / "graphs" / "DONE", {"headline": {"dsg_influence_share_wmdp": res["wmdp-bio"]["dsg_influence_share"]}})
    jc.summary(NAME, {"per_dataset": res, "n_matched_dsg_features": len(matched_ids), "n_items": len(rows)})
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
