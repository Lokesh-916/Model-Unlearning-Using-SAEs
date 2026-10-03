"""Q2: attribution graphs (circuit-tracer 0.5.0, Gemma Scope transcoders) from question to answer.

Mode `tofu` (default; roadmap Q2 "attribution graphs before and after"): for the same TOFU fact, the graph
of the model that knows it and of two unlearned versions, plus one attack case:
  * base  = the TOFU fine-tuned gemma-2-2b-it of tofu-full ($DSG_CACHE/models/A2-tofu-full/full; it knows the
            fictitious authors; the original gemma-2-2b-it does not);
  * dsg   = base + DSG at layer 3 with tofu-full's TOFU features and tau (runs/A2-tofu-full/tofu-metrics/
            partial/full+dsg.json; recomputed with tofu-full's rule if missing). The clamp enters the graph as a
            fixed vector added to the residual stream after layer 3 (for a given prompt it is a constant), so
            the graph shows what the model computes downstream of the clamp; the layer-3 features themselves
            still fire ("the path exists but is clamped");
  * d2    = base with experiments/D2's null-space edit applied to the same TOFU features (MLP down-projections
            of layers 0..3; retain keys from 48 TOFU retain90 QA texts; eps 0.05; ridge 1e-3), i.e. D2's recipe
            on the TOFU features ("baked": no hook);
  * attack: the DSG model on the same fact with the question in French (NLLB-200 600M translation made on the
            lab PC, cluster/data/q2_tofu_forget10_fr.json, back-translation chrF stored) - cross-lingual B3.
Facts: among the first 40 forget10 items, the 3 whose key answer token the base model predicts with the highest
probability. Prompt = chat(question) + the gold answer up to its first "key" token (alphabetic, >= 4 characters,
not in the question, not a stop word); target = that key token. For every fact x condition x language (no graph)
we also record P(key), rho and the gate decision; graphs are drawn for 3 facts x {base, dsg, d2} + 1 attack.
Per graph: replacement / completeness scores, P(key), per-layer influence shares, error / token shares, the DSG
-matched layer-3 transcoder nodes (decoder cosine >= MATCH_COS to a DSG SAE feature), a compact JSON of the
pruned graph (nodes, top edges, token strings) and a figure (PDF + PNG); a 2 x 2 panel for the first fact.
TOFU is fictitious (not hazardous), so token strings and figures are saved and fetched; full graphs (.pt,
~0.25 GB each) are kept for the 4 panel graphs only.

Mode `wmdp`: the earlier aggregate job (WMDP-Bio vs benign MCQ items on gemma-2-2b-it, metrics only; graphs
private, never fetched).

Environment: overlay venv env/q2 (system site packages of mechunlearn2 + wheels/q2; circuit-tracer 0.5.0 needs
transformers <= 4.57.3), created offline by q2-graphs.sbatch. Resumable: finished graphs are skipped; a budget
(DSG_BUDGET_MIN, slurm/later.sh) is checked before each graph.

    python cluster/q2_graphs.py [--mode tofu|wmdp] [--budget-min M] [--plan]
    python cluster/q2_graphs.py --translate      # LAB PC, mechunlearn2 env, CPU: writes the French questions
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

NAME = "q2-graphs"
EXP = "Q2-graphs"
MATCH_COS = 0.5
N_SCAN, N_FACTS = 40, 3
GRAPH_MIN = 10            # budget for one graph (attribution + pruning + figure), with margin
FR_FILE = Path(__file__).resolve().parent / "data" / "q2_tofu_forget10_fr.json"
STOP = {"this", "that", "with", "from", "which", "have", "been", "were", "their", "they", "there", "also", "into",
        "about", "would", "could", "should", "these", "those", "when", "where", "while", "known", "born", "author",
        "authors", "book", "books", "work", "works", "name", "being", "such", "many", "some", "very", "often"}


# ----------------------------------------------------------------------------- graph metrics (both modes)
def layer_of_nodes(active_features, n_layers, n_pos, n_logits):
    """Layer index per node in circuit-tracer order: features, errors (layer-major), tokens (-1), logits (n_layers)."""
    nf = active_features.shape[0]
    lay = [int(x) for x in active_features[:, 0].tolist()]
    lay += [l for l in range(n_layers) for _ in range(n_pos)]
    lay += [-1] * n_pos + [n_layers] * n_logits
    return np.array(lay), nf


def node_positions(active_features, n_layers, n_pos, n_logits):
    nf = active_features.shape[0]
    pos = [int(x) for x in active_features[:, 1].tolist()]
    pos += [p for _ in range(n_layers) for p in range(n_pos)]
    pos += list(range(n_pos)) + [n_pos - 1] * n_logits
    return np.array(pos)


def influence(adjacency, n_logits, logit_probs, n_layers):
    A = np.abs(np.asarray(adjacency, dtype=np.float64))
    rs = A.sum(1, keepdims=True)
    An = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    w = np.zeros(A.shape[0])
    w[A.shape[0] - n_logits:] = np.asarray(logit_probs, dtype=np.float64)
    infl, cur = np.zeros(A.shape[0]), w.copy()
    for _ in range(n_layers + 2):  # propagate back through the DAG
        cur = cur @ An
        infl += cur
        if cur.sum() < 1e-9:
            break
    return infl


def graph_metrics(adjacency, active_features, n_layers, n_pos, logit_probs, node_mask=None, matched=None):
    """Influence of every node on the logits (probability-weighted), aggregated per layer and for the
    DSG-matched feature nodes. adjacency[target, source] = direct effect (circuit-tracer convention).
    matched: bool per active feature (True = equivalent of a DSG feature)."""
    n = np.asarray(adjacency).shape[0]
    n_logits = len(logit_probs)
    lay, nf = layer_of_nodes(np.asarray(active_features), n_layers, n_pos, n_logits)
    assert len(lay) == n, (len(lay), n)
    infl = influence(adjacency, n_logits, logit_probs, n_layers)
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
    a = torch.nn.functional.normalize(torch.as_tensor(np.asarray(sae_wdec)[feats], dtype=torch.float32), dim=-1)
    b = torch.nn.functional.normalize(torch.as_tensor(tc_wdec, dtype=torch.float32), dim=-1)
    cos = a @ b.T
    best = cos.max(1)
    pairs = {int(f): (int(i), float(c)) for f, i, c in zip(feats, best.indices.tolist(), best.values.tolist())}
    return pairs, sorted({i for i, c in pairs.values() if c >= thr})


def compact_graph(g, prune, n_layers, tok, matched=None, max_edges=1500) -> dict:
    """Pruned graph as JSON: nodes (kind, layer, pos, feature, influence), strongest kept edges, token strings."""
    A = g.adjacency_matrix.float().cpu().numpy()
    act = (g.active_features[g.selected_features] if getattr(g, "selected_features", None) is not None else g.active_features).cpu().numpy()
    n_pos, n_logits = len(g.input_tokens), len(g.logit_probabilities)
    lay, nf = layer_of_nodes(act, n_layers, n_pos, n_logits)
    pos = node_positions(act, n_layers, n_pos, n_logits)
    infl = influence(A, n_logits, g.logit_probabilities.float().cpu().numpy(), n_layers)
    keep = prune.node_mask.cpu().numpy().astype(bool)
    em = prune.edge_mask.cpu().numpy().astype(bool) & keep[:, None] & keep[None, :]
    kinds = ["feature"] * nf + ["error"] * (n_layers * n_pos) + ["token"] * n_pos + ["logit"] * n_logits
    idx = np.flatnonzero(keep)
    nodes = [{"id": int(i), "kind": kinds[i], "layer": int(lay[i]), "pos": int(pos[i]),
              "feature": int(act[i, 2]) if i < nf else None, "influence": float(infl[i]),
              "dsg_match": bool(matched[i]) if (matched is not None and i < nf) else False} for i in idx]
    t, s = np.nonzero(em)
    order = np.argsort(-np.abs(A[t, s]))[:max_edges]
    edges = [{"src": int(s[k]), "tgt": int(t[k]), "w": float(A[t[k], s[k]])} for k in order]
    toks = [tok.decode([int(x)]) for x in g.input_tokens.tolist()]
    tgt = [getattr(lt, "token_str", str(lt)) for lt in g.logit_targets]
    return {"n_layers": n_layers, "n_pos": n_pos, "tokens": toks, "targets": tgt,
            "target_probs": [float(x) for x in g.logit_probabilities.float().cpu().tolist()], "nodes": nodes, "edges": edges}


# ----------------------------------------------------------------------------- figures
def draw_graph(cg: dict, ax, title: str, clamp_layer: int | None = 3, max_edges=250):
    """Static attribution-graph picture: x = token position, y = layer; edge width ~ |w|; DSG-matched in orange."""
    import matplotlib.pyplot as plt  # noqa: F401

    L, P = cg["n_layers"], cg["n_pos"]
    by = {n["id"]: n for n in cg["nodes"]}
    rng = np.random.default_rng(0)
    xy = {}
    for n in cg["nodes"]:
        jitter = 0.0 if n["kind"] in ("token", "logit") else rng.uniform(-0.3, 0.3)
        y = -1 if n["kind"] == "token" else (L if n["kind"] == "logit" else n["layer"])
        xy[n["id"]] = (n["pos"] + jitter, y + (0 if n["kind"] != "error" else 0.25))
    E = [e for e in cg["edges"] if e["src"] in xy and e["tgt"] in xy][:max_edges]
    wmax = max([abs(e["w"]) for e in E] or [1.0])
    for e in reversed(E):
        (x0, y0), (x1, y1) = xy[e["src"]], xy[e["tgt"]]
        ax.plot([x0, x1], [y0, y1], color="#2b6cb0" if e["w"] > 0 else "#c53030", alpha=0.15 + 0.6 * abs(e["w"]) / wmax,
                lw=0.3 + 2.2 * abs(e["w"]) / wmax, zorder=1)
    for kind, mk, col, sz in (("feature", "o", "#4a5568", 18), ("error", "s", "#a0aec0", 12), ("token", "^", "#2f855a", 30), ("logit", "D", "#6b46c1", 45)):
        pts = [n for n in cg["nodes"] if n["kind"] == kind]
        if pts:
            ax.scatter([xy[n["id"]][0] for n in pts], [xy[n["id"]][1] for n in pts], marker=mk, s=sz, c=col, zorder=2, linewidths=0)
    m = [n for n in cg["nodes"] if n.get("dsg_match")]
    if m:
        ax.scatter([xy[n["id"]][0] for n in m], [xy[n["id"]][1] for n in m], marker="o", s=40, facecolors="none",
                   edgecolors="#dd6b20", linewidths=1.4, zorder=3, label="DSG-matched feature")
    if clamp_layer is not None:
        ax.axhline(clamp_layer + 0.5, ls="--", lw=0.8, color="#dd6b20")
        ax.text(P - 0.5, clamp_layer + 0.6, "DSG clamp", ha="right", va="bottom", fontsize=6, color="#dd6b20")
    ax.set_xticks(range(P))
    ax.set_xticklabels([t.replace("\n", "\\n")[:10] for t in cg["tokens"]], rotation=90, fontsize=5)
    ax.set_yticks([-1] + list(range(0, L, 4)) + [L])
    ax.set_yticklabels(["emb"] + [str(i) for i in range(0, L, 4)] + ["logit"], fontsize=6)
    ax.set_xlim(-0.8, P - 0.2)
    ax.set_ylim(-1.6, L + 0.6)
    tgt = cg["targets"][0] if cg["targets"] else "?"
    ax.set_title(f"{title}\ntarget {tgt!r}, p = {cg['target_probs'][0]:.2f}" if cg["target_probs"] else title, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save_figure(cg, path: Path, title, clamp_layer):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4.0, 0.16 * cg["n_pos"] + 1.5), 4.2))
    draw_graph(cg, ax, title, clamp_layer)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def save_panel(cgs: dict, path: Path, titles: dict, clamp_layer: int = 3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("base", "dsg", "d2", "attack-fr") if k in cgs]
    if not keys:
        return
    P = max(cgs[k]["n_pos"] for k in keys)
    fig, axs = plt.subplots(2, 2, figsize=(max(7.0, 0.3 * P + 3), 7.0), squeeze=False)
    for ax, k in zip(axs.flat, keys):
        draw_graph(cgs[k], ax, titles[k], None if k == "d2" else clamp_layer)
    for ax in list(axs.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- TOFU facts
def key_split(tok, question: str, answer: str):
    """(prefix answer ids, key token id, key token string) - first informative answer token not in the question."""
    qwords = {w.lower() for w in re.findall(r"[A-Za-z]+", question)}
    ids = tok(answer, add_special_tokens=False)["input_ids"]
    for j, t in enumerate(ids):
        s = tok.decode([t]).strip()
        if j >= 1 and s.isalpha() and len(s) >= 4 and s.lower() not in qwords and s.lower() not in STOP \
                and s.lower() not in question.lower():  # also skips pieces of question words ("siao" of "Hsiao")
            return ids[:j], t, s
    return ids[:1], ids[1] if len(ids) > 1 else ids[0], tok.decode(ids[1:2]).strip()


def local_tokenizer(name="google/gemma-2-2b-it"):
    """Tokenizer from the local HF snapshot path. transformers 4.57.3 (overlay env) calls the Hub API
    (model_info, for its Mistral regex patch) when a tokenizer is loaded by repo id, which fails offline on
    gpuws; a local path skips that call."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot_download(name, local_files_only=True))


def chat_ids(tok, q):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=False)["input_ids"]


class DSGClamp:
    """TransformerLens hook on blocks.<L>.hook_resid_post: DSG decision on the whole input (rho = fraction of
    tokens on which any selected feature fires, BOS excluded from firing as in tofu-full's fire_stats), clamp
    the selected features to -mult at firing positions; applied as a CONSTANT delta (computed without grad), so
    the attribution stays linear and the clamp is a fixed vector in the residual stream."""

    def __init__(self, sae, feats, tau, mult=500.0):
        self.sae, self.tau, self.mult = sae, float(tau), float(mult)
        self.f = torch.tensor(feats)
        self.enabled = False
        self.last = {}

    def __call__(self, acts, hook):
        if not self.enabled:
            return acts
        with torch.no_grad():
            x = acts.to(self.sae.W_dec.dtype)
            a = self.sae.encode(x)
            f = self.f.to(a.device)
            delta = torch.zeros_like(x)
            for b in range(x.shape[0]):
                fire = (a[b][:, f] > 0).any(-1)
                rho = float(fire.float().mean())
                on = rho > self.tau
                self.last = {"rho": rho, "fired": bool(on)}
                if on and fire.any():
                    a2 = a[b].clone()
                    pos = fire.nonzero().flatten()
                    a2[pos[:, None], f[None, :]] = -self.mult
                    delta[b] = self.sae.decode(a2) - self.sae.decode(a[b])
        return acts + delta.to(acts.dtype)


def d2_edit_tl(model, sae_wdec_feats, keys_by_layer, l_star=3, eps=0.05, ridge=1e-3):
    """experiments/D2 (exp/D2-nullspace edit.py) on TransformerLens weights: W_out = down_proj.T."""
    edited = []
    for L in range(l_star + 1):
        mlp = model.blocks[L].mlp
        mlp = getattr(mlp, "old_mlp", mlp)
        W = mlp.W_out  # [d_mlp, d_model]
        dev = W.device
        U = torch.as_tensor(sae_wdec_feats, dtype=torch.float32, device=dev)
        U = U / U.norm(dim=1, keepdim=True)
        down = W.data.float().T  # [d_model, d_mlp]
        K = keys_by_layer[L].to(dev, torch.float32)
        C = (K.T @ K) / max(1, K.shape[0])
        evals, evecs = torch.linalg.eigh(C + ridge * torch.eye(C.shape[0], device=dev))
        keep = evals < evals.max() * eps
        Pn = evecs[:, keep] @ evecs[:, keep].T
        down_new = (torch.eye(down.shape[0], device=dev) - U.T @ U) @ down
        delta = (down_new - down) @ Pn
        W.data = (down + delta).T.to(W.dtype).contiguous()
        edited.append({"layer": L, "null_dim": int(keep.sum()), "delta_fro": float(delta.norm())})
    return edited


@torch.no_grad()
def mlp_keys(model, token_lists, l_star=3):
    """Inputs of W_out (post-activation of the gated MLP) at layers 0..l_star, all tokens of each text."""
    names = [f"blocks.{L}.mlp.old_mlp.hook_post" for L in range(l_star + 1)]
    if names[0] not in model.hook_dict:
        names = [f"blocks.{L}.mlp.hook_post" for L in range(l_star + 1)]
    out = {L: [] for L in range(l_star + 1)}
    for ids in token_lists:
        _, cache = model.run_with_cache(torch.tensor([ids], device=model.cfg.device), names_filter=lambda n: n in names)
        for L, n in enumerate(names):
            out[L].append(cache[n][0].float().cpu())
    return {L: torch.cat(v) for L, v in out.items()}


@torch.no_grad()
def target_prob(model, ids, key):
    logits = model(torch.tensor([ids], device=model.cfg.device))[0, -1].float()
    return float(logits.softmax(-1)[key])


# ----------------------------------------------------------------------------- translation (lab PC)
def translate(n=N_SCAN, out=FR_FILE):
    """LAB PC (mechunlearn2 env, CPU): French versions of the first n TOFU forget10 questions with NLLB-200 600M
    (greedy), back-translation chrF; TOFU is fictitious, so the text is stored in the repo."""
    import sacrebleu
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = [dict(r) for r in load_dataset("locuslab/TOFU", "forget10", split="train")][:n]
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    m = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

    def tr(texts, src, tgt):
        tok.src_lang = src
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            g = m.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=256, num_beams=1)
        return tok.batch_decode(g, skip_special_tokens=True)
    qs = [r["question"] for r in rows]
    fr = tr(qs, "eng_Latn", "fra_Latn")
    back = tr(fr, "fra_Latn", "eng_Latn")
    recs = [{"index": i, "question_fr": f, "chrf_back": float(sacrebleu.sentence_chrf(b, [q]).score)}
            for i, (q, f, b) in enumerate(zip(qs, fr, back))]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": "locuslab/TOFU forget10 (first %d questions)" % n, "translator": name,
                               "decoding": "greedy", "items": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} ({len(recs)} questions; chrF median {np.median([r['chrf_back'] for r in recs]):.1f})")
    return 0


# ----------------------------------------------------------------------------- models
def load_replacement(jc, tok, hf_dir=None):
    from circuit_tracer import ReplacementModel

    if jc.TINY:
        return _tiny_replacement(jc, tok)
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16) if hf_dir else None
    kw = {"hf_model": hf} if hf is not None else {}
    m = ReplacementModel.from_pretrained("google/gemma-2-2b-it", "gemma", dtype=torch.bfloat16, tokenizer=tok, **kw)
    del hf
    return m


def _tiny_replacement(jc, tok):
    """Random tiny Gemma-2-shaped TransformerLens model + random per-layer JumpReLU transcoders (CPU smoke)."""
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
    from transformer_lens import HookedTransformerConfig

    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=4, d_model=64, d_head=32, n_heads=2, n_key_value_heads=1, d_mlp=128,
                                  d_vocab=256000, n_ctx=256, act_fn="gelu_pytorch_tanh", normalization_type="RMS",
                                  positional_embedding_type="rotary", gated_mlp=True, original_architecture="Gemma2ForCausalLM",
                                  use_normalization_before_and_after=True, dtype=torch.float32,
                                  device="cpu", default_prepend_bos=False)
    tcs = {}
    for L in range(cfg.n_layers):
        t = SingleLayerTranscoder(64, 256, JumpReLU(torch.zeros(256), 0.1), L, device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            t.W_enc.normal_(0, 0.2)
            t.W_dec.normal_(0, 0.1)
        tcs[L] = t
    ts = TranscoderSet(tcs, feature_input_hook="ln2.hook_normalized", feature_output_hook="hook_mlp_out", scan="tiny")
    m = ReplacementModel.from_config(cfg, ts, tokenizer=tok)
    for p in m.parameters():
        if p.dim() > 1 and float(p.abs().sum()) == 0:
            torch.nn.init.normal_(p, 0, 0.05)
    return m


def tc_wdec(model, layer=3):
    tc = model.transcoders
    for get in (lambda: tc[layer].W_dec, lambda: tc.transcoders[layer].W_dec, lambda: tc.W_dec[layer]):
        try:
            w = get()
            return w.detach().float().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
        except Exception:
            continue
    return None


def tofu_gate(jc, paths):
    """(features, tau, source) of tofu-full's DSG on the TOFU model; None if the file is missing."""
    p = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "partial" / "full+dsg.json"
    g = (jc.read_json(p, {}) or {}).get("gate") or {}
    if g.get("features") and g.get("threshold") is not None:
        return [int(f) for f in g["features"]], float(g["threshold"]), str(p.relative_to(paths.results_dir()))
    return None


@torch.no_grad()
def recompute_gate(model, sae, f_ids, r_ids, layer=3):
    """tofu-full's rule on TransformerLens activations: top-20 by forget/retain fire ratio (>= 1% forget tokens),
    tau = 95th percentile of rho on retain texts."""
    name = f"blocks.{layer}.hook_resid_post"

    def fires(ids_list):
        tot, n, per = None, 0, []
        for ids in ids_list:
            _, c = model.run_with_cache(torch.tensor([ids], device=model.cfg.device), names_filter=name)
            a = sae.encode(c[name][0].to(sae.W_dec.dtype))
            a[0] = 0
            fr = (a > 0).float().sum(0)
            tot = fr if tot is None else tot + fr
            n += a.shape[0]
            per.append(a > 0)
        return (tot / max(n, 1)).float().cpu().numpy(), per
    pf, _ = fires(f_ids)
    pr, per = fires(r_ids)
    feats = [int(i) for i in np.argsort(-np.where(pf >= 0.01, (pf + 1e-4) / (pr + 1e-4), 0))[:20]]
    rhos = [float(p[:, feats].any(-1).float().mean()) for p in per]
    return feats, float(np.percentile(rhos, 95))


# ----------------------------------------------------------------------------- TOFU mode
def run_tofu(a, jc, paths):
    from circuit_tracer import attribute
    from circuit_tracer.graph import compute_graph_scores, prune_graph
    from datasets import load_dataset

    from dsgx.util import atomic_write_json

    budget = jc.Budget(a.budget_min)
    out = paths.runs_dir() / EXP / "tofu"
    gdir, fdir, pdir = out / "graphs", out / "figures", out / "pt"
    for d in (gdir, fdir, pdir):
        d.mkdir(parents=True, exist_ok=True)
    tok = local_tokenizer()
    f10 = [dict(r) for r in load_dataset("locuslab/TOFU", "forget10", split="train")][:N_SCAN]
    r90 = [dict(r) for r in load_dataset("locuslab/TOFU", "retain90", split="train")][:48]
    if jc.TINY:
        f10, r90 = f10[:4], r90[:4]
    fr = {r["index"]: r for r in json.loads(FR_FILE.read_text())["items"]} if FR_FILE.exists() else {}
    hf_dir = paths.cache_dir() / "models" / "A2-tofu-full" / "full"
    if not jc.TINY and not (hf_dir / "config.json").exists():
        jc.log(NAME, f"ABORT: TOFU model {hf_dir} is missing (tofu-full cleaned up too early?); nothing run")
        return 2
    model = load_replacement(jc, tok, hf_dir)
    model.eval()
    sae = jc.load_sae()
    layer = 1 if jc.TINY else 3
    fact_rows = []
    for i, r in enumerate(f10):
        pre, key, ks = key_split(tok, r["question"], r["answer"])
        fact_rows.append({"index": i, "q_ids": chat_ids(tok, r["question"]), "prefix": pre, "key": int(key), "key_str": ks})
    # gate: tofu-full's features and tau (else recomputed with its rule)
    g = tofu_gate(jc, paths)
    if g is None:
        f_ids = [chat_ids(tok, r["question"]) + tok(r["answer"], add_special_tokens=False)["input_ids"] for r in f10]
        r_ids = [chat_ids(tok, r["question"]) + tok(r["answer"], add_special_tokens=False)["input_ids"] for r in r90]
        feats, tau = recompute_gate(model, sae, f_ids, r_ids, layer)
        gsrc = "recomputed with tofu-full's rule (TransformerLens activations)"
    else:
        feats, tau, gsrc = g
    clamp = DSGClamp(sae, feats, tau)
    model.hook_dict[f"blocks.{layer}.hook_resid_post"].add_hook(clamp, is_permanent=True)
    # facts: highest base P(key) among the first N_SCAN
    for fr_ in fact_rows:
        fr_["p_base"] = target_prob(model, fr_["q_ids"] + fr_["prefix"], fr_["key"])
    facts = sorted(fact_rows, key=lambda x: -x["p_base"])[: (2 if jc.TINY else N_FACTS)]
    # D2 edit needs the retain keys of the unedited model
    r_texts = [chat_ids(tok, r["question"]) + tok(r["answer"], add_special_tokens=False)["input_ids"] for r in r90]
    sae_wdec = sae.W_dec.detach().float().cpu().numpy()
    tcw = tc_wdec(model, layer)
    pairs, matched_ids = dsg_matches(sae_wdec, tcw, feats) if tcw is not None else ({}, [])
    keys = mlp_keys(model, r_texts, layer)

    def conds():
        yield "base", False
        yield "dsg", True

    def ids_for(f, lang):
        if lang == "en":
            return f["q_ids"] + f["prefix"]
        q = fr.get(f["index"], {}).get("question_fr")
        return (chat_ids(tok, q) + f["prefix"]) if q else None

    def graph(tag, ids, f, keep_pt):
        gp = gdir / f"{tag}.json"
        if gp.exists():
            return json.loads(gp.read_text())
        if not budget.fits(GRAPH_MIN):
            return None
        gr = attribute(torch.tensor(ids), model, attribution_targets=torch.tensor([f["key"]]), batch_size=a.batch_size,
                       max_feature_nodes=a.max_feature_nodes, offload=None if jc.TINY else "cpu")
        gate = dict(clamp.last) if clamp.enabled else None
        pr = prune_graph(gr, 0.8, 0.98)
        rep, comp = compute_graph_scores(gr)
        act = gr.active_features[gr.selected_features] if getattr(gr, "selected_features", None) is not None else gr.active_features
        m = ((act[:, 0] == layer) & torch.isin(act[:, 2], torch.tensor(matched_ids, dtype=act.dtype))).cpu().numpy() if matched_ids else None
        met = graph_metrics(gr.adjacency_matrix.float().cpu().numpy(), act.cpu().numpy(), model.cfg.n_layers, len(gr.input_tokens),
                            gr.logit_probabilities.float().cpu().numpy(), pr.node_mask.cpu().numpy(), m)
        cg = compact_graph(gr, pr, model.cfg.n_layers, tok, None if m is None else np.concatenate([m, np.zeros(len(pr.node_mask) - len(m), bool)]))
        rec = {"tag": tag, "fact_index": f["index"], "key_token": f["key_str"], "replacement_score": float(rep),
               "completeness_score": float(comp), "p_key": float(gr.logit_probabilities.float()[0]), "gate": gate, **met, "graph": cg}
        if keep_pt:
            gr.to_pt(str(pdir / f"{tag}.pt"))
        save_figure(cg, fdir / tag, tag, None if tag.endswith("-d2") else layer)
        atomic_write_json(gp, rec)
        jc.log(NAME, f"graph {tag}: p_key {rec['p_key']:.3f}, replacement {rep:.3f}, gate {gate}")
        del gr
        torch.cuda.empty_cache()
        return rec

    # 1) base + dsg graphs (same weights), and the attack case
    recs, table = {}, []
    for k, f in enumerate(facts):
        for cname, on in conds():
            clamp.enabled = on
            recs[f"fact{k}-{cname}"] = graph(f"fact{k}-{cname}", ids_for(f, "en"), f, keep_pt=(k == 0))
    f0 = facts[0]
    clamp.enabled = True
    ids_fr = ids_for(f0, "fr")
    if ids_fr is not None:
        recs["fact0-attack-fr"] = graph("fact0-attack-fr", ids_fr, f0, keep_pt=True)
    # P(key), rho and gate decision for every fact x {base, dsg} x {en, fr} (no graph)
    for f in facts:
        for lang in ("en", "fr"):
            ids = ids_for(f, lang)
            if ids is None:
                continue
            for cname, on in conds():
                clamp.enabled = on
                p = target_prob(model, ids, f["key"])
                table.append({"fact_index": f["index"], "lang": lang, "condition": cname, "p_key": p,
                              "gate": dict(clamp.last) if on else None})
    # 2) D2: edit the weights in place (after every base/dsg use), no hook
    clamp.enabled = False
    edit = d2_edit_tl(model, sae_wdec[feats], keys, l_star=layer)
    for k, f in enumerate(facts):
        recs[f"fact{k}-d2"] = graph(f"fact{k}-d2", ids_for(f, "en"), f, keep_pt=(k == 0))
        for lang in ("en", "fr"):
            ids = ids_for(f, lang)
            if ids is not None:
                table.append({"fact_index": f["index"], "lang": lang, "condition": "d2", "p_key": target_prob(model, ids, f["key"]), "gate": None})
    done = all(v is not None for v in recs.values())
    if recs.get("fact0-base") and recs.get("fact0-dsg") and recs.get("fact0-d2"):
        save_panel({k.split("-", 1)[1]: v["graph"] for k, v in recs.items() if v and k.startswith("fact0-")}, fdir / "panel_fact0",
                   {"base": "TOFU model (knows the fact)", "dsg": f"+ DSG (clamp after layer {layer})", "d2": "D2 edit (baked)",
                    "attack-fr": "+ DSG, question in French"}, clamp_layer=layer)
    summ = {"mode": "tofu", "facts": [{k: v for k, v in f.items() if k in ("index", "key_str", "p_base")} for f in facts],
            "gate": {"features": feats, "tau": tau, "source": gsrc, "layer": layer}, "dsg_matches": pairs, "matched_ids": matched_ids,
            "match_cos": MATCH_COS, "d2_edit": edit, "prob_table": table, "complete": done,
            "graphs": {k: ({kk: vv for kk, vv in v.items() if kk != "graph"} if v else None) for k, v in recs.items()},
            "french_file": FR_FILE.name if FR_FILE.exists() else None}
    atomic_write_json(out / "metrics.json", summ)
    if done:
        atomic_write_json(out / "DONE", {"headline": {k: (v["p_key"], (v.get("gate") or {}).get("fired")) for k, v in recs.items()}})
    jc.summary(NAME, {k: v for k, v in summ.items() if k not in ("prob_table",)})
    jc.log(NAME, "tofu: done" if done else "tofu: partial (budget); resubmit q2-graphs.sbatch to continue")
    return 0


# ----------------------------------------------------------------------------- WMDP mode (aggregate, private graphs)
def run_wmdp(a, jc, paths):
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_graph_scores, prune_graph

    from dsgx.data.mcq import format_prompt
    from dsgx.util import atomic_write_json

    items = [("wmdp-bio", it) for it in jc.mcq_items("wmdp-bio", "test", a.n)] + \
            [(a.benign, it) for it in jc.mcq_items(a.benign, "test", a.n)]
    feats, tau = jc.dsg_features()
    model = ReplacementModel.from_pretrained("google/gemma-2-2b-it", "gemma", dtype=torch.bfloat16, tokenizer=local_tokenizer())
    tcw = tc_wdec(model)
    sae = jc.load_sae()
    pairs, matched_ids = dsg_matches(sae.W_dec.detach().float().cpu().numpy(), tcw, feats) if tcw is not None else ({}, [])
    priv = jc.private_dir(NAME)
    out_dir = paths.runs_dir() / EXP
    rows, kept = [], {}
    budget = jc.Budget(a.budget_min)
    for ds, it in items:
        rec_p = out_dir / "items" / f"{ds}__{it.item_id}.json"
        if rec_p.exists():
            rows.append(json.loads(rec_p.read_text()))
            continue
        if not budget.fits(GRAPH_MIN):
            break
        prompt = format_prompt(it)
        g = attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.95, batch_size=a.batch_size,
                      max_feature_nodes=a.max_feature_nodes, offload="cpu")
        if kept.get(ds, 0) < a.keep_graphs:  # graphs contain prompt tokens: private only
            g.to_pt(str(priv / f"{ds}__{it.item_id}.pt"))
            kept[ds] = kept.get(ds, 0) + 1
        pr = prune_graph(g, 0.8, 0.98)
        rep, comp = compute_graph_scores(g)
        act = g.active_features[g.selected_features] if hasattr(g, "selected_features") else g.active_features
        m = ((act[:, 0] == 3) & torch.isin(act[:, 2], torch.tensor(matched_ids))).cpu().numpy() if matched_ids else None
        met = graph_metrics(g.adjacency_matrix.float().cpu().numpy(), act.cpu().numpy(), model.cfg.n_layers,
                            len(g.input_tokens), g.logit_probabilities.float().cpu().numpy(), pr.node_mask.cpu().numpy(), m)
        rec = {"dataset": ds, "item_id": int(it.item_id), "prompt_hash": jc.text_hash(prompt), "replacement_score": float(rep),
               "completeness_score": float(comp), **met}
        atomic_write_json(rec_p, rec)
        rows.append(rec)
        del g
        torch.cuda.empty_cache()

    def agg(ds, k):
        from dsgx.eval import stats

        v = [r[k] for r in rows if r["dataset"] == ds and r.get(k) is not None]
        return stats.bootstrap_ci(v) if v else None
    res = {ds: {k: agg(ds, k) for k in ("replacement_score", "completeness_score", "share_layers_le3", "dsg_influence_share",
                                        "feature_influence_share")} for ds in ("wmdp-bio", a.benign)}
    atomic_write_json(out_dir / "graphs" / "metrics.json", {"per_dataset": res, "dsg_matches": pairs, "matched_ids": matched_ids,
                                                            "match_cos": MATCH_COS, "n": len(rows)})
    if len(rows) == len(items):
        atomic_write_json(out_dir / "graphs" / "DONE", {"headline": {"dsg_influence_share_wmdp": res["wmdp-bio"]["dsg_influence_share"]}})
    jc.summary(NAME, {"mode": "wmdp", "per_dataset": res, "n_matched_dsg_features": len(matched_ids), "n_items": len(rows)})
    return 0


def plan(a, jc, paths):
    """CPU, no weights loaded: imports, offline tokenizer, offline transcoder file resolution, inputs present."""
    import yaml
    from huggingface_hub import hf_hub_download
    from circuit_tracer.utils.hf_utils import resolve_transcoder_paths

    import circuit_tracer  # noqa: F401  (the overlay env must import)

    rep = {"mode": a.mode, "french_file": FR_FILE.exists(), "tokenizer": type(local_tokenizer()).__name__}
    if not jc.TINY:
        cfg = yaml.safe_load(open(hf_hub_download("mwhanna/gemma-scope-transcoders", "config.yaml")))
        cfg.update({"repo_id": "mwhanna/gemma-scope-transcoders", "revision": None, "subfolder": None})
        tp = resolve_transcoder_paths(cfg)
        rep["transcoder_layers"] = len(tp)
        rep["transcoders_present"] = all(Path(p).exists() for p in tp.values())
        rep["tofu_model"] = (paths.cache_dir() / "models" / "A2-tofu-full" / "full" / "config.json").exists()
        rep["tofu_gate_file"] = tofu_gate(jc, paths) is not None
    print(json.dumps(rep))
    ok = rep.get("transcoder_layers", 26) == 26 and rep.get("transcoders_present", True) and rep["french_file"]
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tofu", "wmdp"], default="tofu")
    ap.add_argument("--n", type=int, default=40, help="wmdp mode: WMDP-Bio TEST items (and as many benign items)")
    ap.add_argument("--benign", default="high_school_geography")
    ap.add_argument("--max-feature-nodes", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--keep-graphs", type=int, default=3, help="wmdp mode: graphs saved (private) per dataset")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--plan", action="store_true", help="imports only (CPU)")
    ap.add_argument("--translate", action="store_true", help="LAB PC: write the French TOFU questions")
    a = ap.parse_args(argv)
    if a.translate:
        return translate()
    from cluster import jobcommon as jc
    from dsgx import paths

    if a.plan:
        return plan(a, jc, paths)
    jc.require_gpu(30)
    return run_tofu(a, jc, paths) if a.mode == "tofu" else run_wmdp(a, jc, paths)


if __name__ == "__main__":
    raise SystemExit(main())
