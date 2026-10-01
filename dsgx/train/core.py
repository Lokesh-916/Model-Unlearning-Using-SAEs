"""Training utilities shared by A2 (TOFU LoRA), A6 (relearning), D1 (distillation), B5 (soft prompts).

* load_hf(name, weights=None, dtype) -> (model, tokenizer) on GPU
* add_dsg_hook_hf(model, sae, features, multiplier, tau, layer): the DSG clamp on an HF model
  (forward hook on decoder layer `layer`'s output == TransformerLens blocks.<layer>.hook_resid_post);
  differentiable, so it can stay on during training ("DSG with hook").
* mcq_probs_hf(model, tok, prompts, bs): same scoring rule as the harness (full-vocab softmax,
  max of "A"/" A" ...), for periodic evaluation inside training loops.
* Trainer: step loop with checkpoint every N steps + resume, train_log.parquet, progress heartbeat.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from dsgx.util import atomic_write_json


def load_hf(name="google/gemma-2-2b-it", weights=None, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(weights or name, torch_dtype=dtype,
                                                 attn_implementation="eager").to(device)
    return model, tok


def decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if hasattr(m, "language_model"):
        m = m.language_model
    return m.layers


class HFDSGHook:
    def __init__(self, sae, features, multiplier, tau, faithful=True):
        from dsgx.methods.dsg import DSGHook

        self.h = DSGHook(sae, features, multiplier, tau, faithful=faithful, record=False)
        self.enabled = True
        self.fired = []

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        new = self.h(hs.to(self.h.sae.W_dec.dtype)).to(hs.dtype)
        return (new, *output[1:]) if isinstance(output, tuple) else new


def add_dsg_hook_hf(model, sae, features, multiplier, tau, layer, faithful=True):
    hook = HFDSGHook(sae, features, multiplier, tau, faithful)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    return hook, handle


@torch.no_grad()
def mcq_probs_hf(model, tok, prompts, bs=8):
    letters = ["A", "B", "C", "D", " A", " B", " C", " D"]
    ans = torch.tensor([tok(x, add_special_tokens=False)["input_ids"][0] for x in letters], device=model.device)
    out = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        p = logits.softmax(-1)[:, ans].reshape(-1, 2, 4).max(1).values
        out.append(p.cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 4))


def mcq_accuracy_hf(model, tok, items, bs=8):
    from dsgx.data.mcq import format_prompt

    p = mcq_probs_hf(model, tok, [format_prompt(it) for it in items], bs)
    return (p.argmax(1) == np.array([it.answer for it in items])).astype(float)


def lm_loss(model, tok, texts, max_len=512, mask_prefix=None):
    """Mean next-token loss; mask_prefix: list of prefix strings whose tokens are excluded."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=False).to(model.device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    if mask_prefix:
        for i, p in enumerate(mask_prefix):
            n = len(tok(p, add_special_tokens=False)["input_ids"])
            labels[i, :n] = -100
    return model(**enc, labels=labels).loss


class Trainer:
    """Minimal resumable loop. step_fn(step) -> dict(loss=tensor, **floats) does forward+loss."""

    def __init__(self, model, params, out_dir: Path, lr=1e-4, steps=100, ckpt_every=50, progress=None,
                 eval_fn=None, eval_every=None, optimizer="adamw", save_fn=None, grad_clip=1.0):
        self.model, self.out = model, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.params = [p for p in params if p.requires_grad]
        if optimizer == "adamw8bit":
            import bitsandbytes as bnb

            self.opt = bnb.optim.AdamW8bit(self.params, lr=lr)
        else:
            self.opt = torch.optim.AdamW(self.params, lr=lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)))
        self.steps, self.ckpt_every, self.progress = steps, ckpt_every, progress
        self.eval_fn, self.eval_every, self.save_fn, self.grad_clip = eval_fn, eval_every, save_fn, grad_clip
        self.log = []
        self.start = 0
        self._resume()

    def _state_path(self):
        return self.out / "last"

    def _resume(self):
        sp = self._state_path() / "trainer.pt"
        if sp.exists():
            st = torch.load(sp, weights_only=False)
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            sd = st["params"]
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in sd:
                        p.copy_(sd[n].to(p.device, p.dtype))
            self.opt.load_state_dict(st["opt"])
            self.sched.load_state_dict(st["sched"])
            self.start = st["step"]
            self.log = st.get("log", [])
            print(f"[train] resumed at step {self.start} ({len(trainable)} trainable tensors)", flush=True)

    def checkpoint(self, step, tag="last"):
        d = self.out / tag
        d.mkdir(parents=True, exist_ok=True)
        sd = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        tmp = d / "trainer.pt.tmp"
        torch.save({"step": step, "params": sd, "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                    "log": self.log}, tmp)
        tmp.replace(d / "trainer.pt")
        if self.save_fn:
            self.save_fn(d)

    def run(self, step_fn):
        import pandas as pd

        self.model.train()
        t0 = time.time()
        for step in range(self.start, self.steps):
            out = step_fn(step)
            loss = out.pop("loss")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip))
            self.opt.step()
            self.sched.step()
            row = {"step": step + 1, "loss": float(loss), "lr": self.sched.get_last_lr()[0], "grad_norm": gn,
                   "time": time.time() - t0,
                   "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                   **{k: float(v) for k, v in out.items()}}
            if self.eval_fn and self.eval_every and ((step + 1) % self.eval_every == 0 or step + 1 == self.steps):
                self.model.eval()
                row.update(self.eval_fn(step + 1))
                self.model.train()
            self.log.append(row)
            if self.progress:
                self.progress.update(step=step + 1, steps_total=self.steps, current_metric=f"loss {float(loss):.3f}")
            if (step + 1) % self.ckpt_every == 0 or step + 1 == self.steps:
                self.checkpoint(step + 1)
                pd.DataFrame(self.log).to_parquet(self.out / "train_log.parquet", index=False)
        self.model.eval()
        pd.DataFrame(self.log).to_parquet(self.out / "train_log.parquet", index=False)
        atomic_write_json(self.out / "train_summary.json", {"steps": self.steps, "last": self.log[-1] if self.log else None})
        return self.log
