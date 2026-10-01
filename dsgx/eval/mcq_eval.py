"""MCQ scoring: exact port of get_output_probs_abcd (chunked full-vocab softmax with softcap).

Prediction = argmax over max(P("A"), P(" A")), ... as in DSG.
"""
import numpy as np
import torch

ANSWER_STRINGS = ["A", "B", "C", "D", " A", " B", " C", " D"]


@torch.no_grad()
def answer_probs(model, token_batch, next_token_indices, answer_tokens, chunk_size: int = 8192):
    """[B, 4] probabilities (full-vocab softmax) of the answer letters at the next position."""
    B = token_batch.shape[0]
    resid = model(token_batch, return_type=None, stop_at_layer=model.cfg.n_layers)
    resid = model.ln_final(resid)
    resid = resid[torch.arange(B, device=token_batch.device), next_token_indices]
    W_U, b_U = model.unembed.W_U, model.unembed.b_U
    softcap = getattr(model.cfg, "output_logits_soft_cap", None)
    active = softcap is not None and softcap > 0

    def cap(x):
        return softcap * torch.tanh(x / softcap) if active else x

    max_logit = torch.full((B,), -float("inf"), device=resid.device, dtype=torch.float32)
    for s in range(0, W_U.shape[1], chunk_size):
        e = min(s + chunk_size, W_U.shape[1])
        lc = cap(torch.nn.functional.linear(resid, W_U[:, s:e].T.contiguous(), b_U[s:e])).float()
        max_logit = torch.maximum(max_logit, lc.max(dim=-1).values)
    exp_sum = torch.zeros(B, device=resid.device, dtype=torch.float32)
    ans_logits = cap(torch.nn.functional.linear(resid, W_U[:, answer_tokens].T.contiguous(),
                                                b_U[answer_tokens])).float()
    for s in range(0, W_U.shape[1], chunk_size):
        e = min(s + chunk_size, W_U.shape[1])
        lc = cap(torch.nn.functional.linear(resid, W_U[:, s:e].T.contiguous(), b_U[s:e])).float()
        exp_sum += torch.exp(lc - max_logit.unsqueeze(-1)).sum(dim=-1)
    vals = torch.exp(ans_logits - max_logit.unsqueeze(-1)) / exp_sum.unsqueeze(-1)
    return vals.reshape(-1, 2, 4).max(dim=1)[0]


def score_prompts(model, prompts: list[str], batch_size: int = 1, method=None, on_batch=None):
    """Score prompts. Returns (probs [N,4] float32, gate records list, token lengths)."""
    if not prompts:
        return np.zeros((0, 4), dtype=np.float32), [], []
    answer_tokens = model.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
    probs, records, lens_all = [], [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tb = model.to_tokens(batch, padding_side="right", prepend_bos=False).to(model.cfg.device)
        assert (tb == model.tokenizer.bos_token_id).sum().item() == len(tb)
        lens = [len(model.to_tokens(x, prepend_bos=False)[0]) for x in batch]
        if method is not None:
            method.set_lengths(lens)
            if hasattr(method, "before_forward"):
                method.before_forward(tb, lens)
        nti = torch.tensor([n - 1 for n in lens], device=tb.device)
        probs.append(answer_probs(model, tb, nti, answer_tokens).float().cpu())
        if method is not None:
            records.extend(method.pop_records())
        lens_all.extend(lens)
        if on_batch:
            on_batch(min(i + batch_size, len(prompts)))
    return torch.cat(probs).numpy(), records, lens_all


def prob_features(p: np.ndarray):
    """Entropy (over renormalised A-D) and margin (top1 - top2 of raw probs)."""
    q = p / np.clip(p.sum(1, keepdims=True), 1e-30, None)
    ent = -(q * np.log(np.clip(q, 1e-30, None))).sum(1)
    s = np.sort(p, axis=1)
    return ent, s[:, -1] - s[:, -2]
