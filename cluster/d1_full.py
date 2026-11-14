"""D1-full: UNDO-style distillation of the DSG-guarded gemma-2-2b-it into a NOISED FULL-PARAMETER student.

Server version of exp/D1-local (which is LoRA on 16 GB). For each noise alpha in ALPHAS:
  student <- shrink-and-perturb copy of the base weights: theta = (1 - a) * theta + a * eps,
             eps ~ N(0, std(theta)) per tensor (UNDO, Lee et al. 2025);
  loss    = KL(student || DSG-guarded teacher) on forget passages (all tokens)
          + KL(student || unguarded teacher) on retain passages (wikitext-2 train + MMLU dev prompts);
  bf16 weights, AdamW8bit, lr 1e-5 (warmup 5%), grad checkpointing, batch 4 x 512 tokens, STEPS steps;
  checkpoint (trainer state) every 100 steps -> a Slurm requeue resumes; the final student is saved as
  HF weights to $DSG_CACHE/models/D1-full/undo_a<alpha>.
Then harness TEST evaluation (bs=1, both views, WMDP-Bio + full-MMLU utility) of base, DSG (paper config)
and every student, all on gpuws (exp id D1-full). The fixed-budget relearning test is job a6-full.
Writes $DSG_RESULTS/jobs/d1-full/summary.json. Hazardous passages are never printed or written.
"""
import argparse
import gc
import shutil

import torch
import torch.nn.functional as F

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json

NAME = "d1-full"
ALPHAS = [0.1, 0.3, 0.5]
STEPS, BS, MAXLEN, LR, CKPT = 2000, 4, 512, 1e-5, 100


def noise_(model, alpha, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 0:
                continue
            eps = torch.randn(p.shape, generator=g).to(p.device, torch.float32) * p.float().std().clamp_min(1e-8)
            p.copy_(((1 - alpha) * p.float() + alpha * eps).to(p.dtype))


def kl_tokens(s_logits, t_logits, mask):
    ls = F.log_softmax(s_logits.float(), -1)
    pt = F.softmax(t_logits.float(), -1)
    kl = (pt * (torch.log(pt.clamp_min(1e-12)) - ls)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1)


def train_alpha(alpha, a, forget, retain, tok, out_root):
    from dsgx.train.core import Trainer

    tag = f"undo_a{alpha}"
    final = out_root / tag
    if (final / "config.json").exists():
        jc.log(NAME, f"{tag}: final weights exist, skip training")
        return final
    teacher = jc.load_lm(jc.HF_2B)
    teacher.eval().requires_grad_(False)
    hook, handle, info = jc.dsg_guard(teacher)
    student = jc.load_lm(jc.HF_2B, train=True)
    noise_(student, alpha, seed=int(alpha * 1000))
    g = torch.Generator().manual_seed(0)
    dev = next(student.parameters()).device

    def batch(src):
        idx = torch.randint(0, len(src), (a.bs,), generator=g).tolist()
        enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen)
        return {k: v.to(dev) for k, v in enc.items()}

    def step_fn(step):
        fb, rb = batch(forget), batch(retain)
        with torch.no_grad():
            hook.enabled = True
            tf = teacher(**fb).logits
            hook.enabled = False
            tr = teacher(**rb).logits
        kf = kl_tokens(student(**fb).logits, tf, fb["attention_mask"])
        kr = kl_tokens(student(**rb).logits, tr, rb["attention_mask"])
        return {"loss": kf + kr, "kl_forget": kf.detach(), "kl_retain": kr.detach()}

    work = jc.job_dir(NAME) / tag
    tr = Trainer(student, student.parameters(), work, lr=a.lr, steps=a.steps, ckpt_every=a.ckpt,
                 optimizer="adamw" if jc.TINY else "adamw8bit")
    tr.run(step_fn)
    final.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final)
    tok.save_pretrained(final)
    atomic_write_json(final / "d1_full.json", {"alpha": alpha, "steps": a.steps, "dsg": info, "last": tr.log[-1] if tr.log else None})
    shutil.rmtree(work / "last", ignore_errors=True)  # trainer state (~10 GB) no longer needed
    handle.remove()
    del teacher, student, tr
    gc.collect()
    torch.cuda.empty_cache()
    return final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="*", default=ALPHAS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--ckpt", type=int, default=CKPT)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    forget = jc.forget_passages("bio", n=None if not jc.TINY else 16)
    from dsgx.data.mcq import format_prompt

    retain = jc.retain_passages(n=None if not jc.TINY else 16)
    retain += [format_prompt(it) for it in jc.mcq_items("high_school_geography", "dev")[: (8 if jc.TINY else None)]]
    jc.log(NAME, f"forget passages {len(forget)}, retain passages {len(retain)}")
    out_root = paths.cache_dir() / "models" / "D1-full"
    finals = {alpha: str(train_alpha(alpha, a, forget, retain, tok, out_root)) for alpha in a.alphas}
    res = {"alphas": a.alphas, "checkpoints": finals, "steps": a.steps, "runs": {}}
    if not a.no_eval and not jc.TINY:
        res["runs"]["base"] = str(jc.harness_eval("D1-full", "base", {"name": "base"}))
        res["runs"]["dsg_paper"] = str(jc.harness_eval("D1-full", "dsg-paper", {"name": "dsg-faithful", "n_features": 20,
                                                                                 "retain_pct": 95, "multiplier": 500}))
        for alpha, ck in finals.items():
            res["runs"][f"undo_a{alpha}"] = str(jc.harness_eval("D1-full", f"undo-full-a{alpha}", {"name": "base"}, weights=ck))
    res["headline"] = jc.headlines(res["runs"])
    jc.summary(NAME, res)
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
