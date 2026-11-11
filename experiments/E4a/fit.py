"""E4a fit task (GPU): the hazard-vs-benign-biology detectors, trained on DOCUMENTS only (no question data).

(a) a-sae-cusum, per SAE: token fire rates p (forget), q_b (benign biology), q_w (WikiText) from the docs task.
    Eligible: p >= P_MIN. Rank by log((p + EPS) / (max(q_b, q_w) + EPS)), top N_FEATURES ("frequent on forget, rare
    on benign biology AND WikiText"). Token LLR = Bernoulli naive Bayes of the 20-feature fire pattern, forget p vs
    the null q = max(q_b, q_w) (rates clipped to [EPS, 1 - EPS], as dsgx.methods.gates); score = CUSUM max (drift 0).
(b) per layer, logistic regression on the raw residual, forget (1) vs benign biology (0), class-balanced, inputs
    standardised on the training docs, L2 lambda from PROBE_LAMBDAS by CV_FOLDS-fold document-level CV AUROC:
    b-probe-mean  on the mean-pooled residual of each document; question score = the probe logit of its mean residual
    b-probe-cusum on TOK_PER_DOC sampled token residuals per document (folds grouped by document); question score =
                  CUSUM max of the per-token logits (positions > 0).
Outputs: $DSG_CACHE/E4a/<exp>/fit/<tag>.json (a) and probe_L<l>.npz (b); a layer already fitted is reused.
"""
import json

import numpy as np
import torch

from dsgx.util import atomic_write_bytes, atomic_write_json

from experiments.E4a.common import (CV_FOLDS, EPS, LAYERS, N_FEATURES, P_MIN, PROBE_LAMBDAS, SAES, sha, work_dir)
from experiments.E4a.docs import load_all


def select(p, qb, qw, k=N_FEATURES, p_min=P_MIN):
    p, q = np.asarray(p, float), np.maximum(np.asarray(qb, float), np.asarray(qw, float))
    score = np.log((p + EPS) / (q + EPS))
    score[p < p_min] = -np.inf
    order = np.argsort(-score, kind="stable")
    order = [int(i) for i in order if np.isfinite(score[i])][:k]
    return order, score


def llr(p, q):
    p, q = np.clip(p, EPS, 1 - EPS), np.clip(q, EPS, 1 - EPS)
    return np.log(p / q), np.log((1 - p) / (1 - q))


def auroc(pos, neg) -> float:
    """Mann-Whitney AUROC (ties = 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks of ties
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logreg(X, y, lam, iters=200):
    """Balanced L2 logistic regression on standardised X (torch, full-batch LBFGS). -> (w, b)."""
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    wt = torch.where(y > 0, 0.5 / n1, 0.5 / n0)
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = (wt * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() \
            + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return w.detach(), b.detach()


def standardise(X):
    mu = X.mean(0)
    sd = X.std(0).clamp_min(1e-6)
    return mu, sd


def cv_probe(X, y, groups, folds=CV_FOLDS, lambdas=PROBE_LAMBDAS, seed=0):
    """-> (best lambda, {lambda: mean doc-level CV AUROC}). Folds by document (groups)."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % folds))
    f = torch.tensor([fold_of[g] for g in groups], device=X.device)
    res = {}
    for lam in lambdas:
        aucs = []
        for k in range(folds):
            tr, te = f != k, f == k
            mu, sd = standardise(X[tr])
            w, b = fit_logreg((X[tr] - mu) / sd, y[tr], lam)
            z = (((X[te] - mu) / sd) @ w + b).cpu().numpy()
            yt = y[te].cpu().numpy()
            aucs.append(auroc(z[yt > 0], z[yt == 0]))
        res[lam] = float(np.mean(aucs))
    best = max(lambdas, key=lambda l: (res[l], -l))
    return best, res


def task(ctx):
    exp = ctx.args.get("docs_exp", ctx.exp_id)
    fdir = work_dir(ctx.exp_id, "fit")
    D = {c: load_all(exp, c) for c in ("forget", "benign", "wiki")}
    ntok = {c: float(D[c]["ntok"].sum()) for c in D}
    ctx.write_config(docs_exp=exp, n_docs={c: int(len(D[c]["ntok"])) for c in D}, ntok=ntok)
    ctx.progress.update(items_total=len(SAES) + 2 * len(LAYERS), items_done=0, force=True)
    out = {"a": {}, "b": {}}
    # (a)
    for k, (tag, layer, sid) in enumerate(SAES):
        p, qb, qw = (D[c]["fire"][k] / ntok[c] for c in ("forget", "benign", "wiki"))
        feats, score = select(p, qb, qw)
        q = np.maximum(qb, qw)
        w1, w0 = llr(p[feats], q[feats])
        rec = {"tag": tag, "layer": layer, "sae_id": sid, "features": feats,
               "p_forget": [float(x) for x in p[feats]], "q_benign": [float(x) for x in qb[feats]],
               "q_wiki": [float(x) for x in qw[feats]], "w1": [float(x) for x in w1], "w0": [float(x) for x in w0],
               "n_eligible": int(np.isfinite(score).sum()), "fp": sha([exp, ntok, tag])}
        atomic_write_json(fdir / f"{tag}.json", rec)
        out["a"][tag] = {"n_eligible": rec["n_eligible"], "min_log_ratio": float(score[feats[-1]]) if feats else None}
        ctx.progress.advance(1)
    # (b)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_f, n_b = len(D["forget"]["ntok"]), len(D["benign"]["ntok"])
    for li, layer in enumerate(LAYERS):
        f = fdir / f"probe_L{layer:02d}.npz"
        fp = sha([exp, ntok, layer, PROBE_LAMBDAS, CV_FOLDS])
        if f.exists() and str(np.load(f)["fp"]) == fp:
            with np.load(f) as z:
                out["b"][f"L{layer:02d}"] = json.loads(str(z["info"]))
            ctx.progress.advance(2)
            continue
        info, arrs = {}, {}
        # mean-pooled
        X = torch.tensor(np.concatenate([D["forget"]["mean"][:, li], D["benign"]["mean"][:, li]]),
                         dtype=torch.float32, device=dev)
        y = torch.cat([torch.ones(n_f), torch.zeros(n_b)]).to(dev)
        g = np.arange(n_f + n_b)
        lam, cv = cv_probe(X, y, g)
        mu, sd = standardise(X)
        w, b = fit_logreg((X - mu) / sd, y, lam)
        info["mean"] = {"lambda": lam, "cv_auroc": cv, "n_pos": n_f, "n_neg": n_b}
        arrs.update(mean_mu=mu.cpu().numpy(), mean_sd=sd.cpu().numpy(), mean_w=w.cpu().numpy(), mean_b=b.cpu().numpy())
        del X
        ctx.progress.advance(1)
        # token-level
        xs, ys, gs = [], [], []
        for c, lab, off in (("forget", 1.0, 0), ("benign", 0.0, n_f)):
            tp = D[c]["tpos"]
            ok = tp >= 0
            xs.append(D[c]["tok"][:, :, li][ok])
            ys.append(np.full(int(ok.sum()), lab))
            gs.append((np.arange(len(tp))[:, None] + off).repeat(tp.shape[1], 1)[ok])
        X = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=dev)
        y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=dev)
        g = np.concatenate(gs)
        lam, cv = cv_probe(X, y, g)
        mu, sd = standardise(X)
        w, b = fit_logreg((X - mu) / sd, y, lam)
        info["tok"] = {"lambda": lam, "cv_auroc_token": cv, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        arrs.update(tok_mu=mu.cpu().numpy(), tok_sd=sd.cpu().numpy(), tok_w=w.cpu().numpy(), tok_b=b.cpu().numpy())
        del X
        import io

        buf = io.BytesIO()
        np.savez(buf, fp=np.array(fp), info=np.array(json.dumps(info)), **arrs)
        atomic_write_bytes(f, buf.getvalue())
        out["b"][f"L{layer:02d}"] = info
        print(f"[E4a fit] L{layer:02d} mean cv {info['mean']['cv_auroc']} tok cv {info['tok']['cv_auroc_token']}", flush=True)
        ctx.progress.advance(1)
        torch.cuda.empty_cache()
    ctx.write_metrics(out)
    ctx.finish({"view": "e4a-fit", "forget": None})
    return out
