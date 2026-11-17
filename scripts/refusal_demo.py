#!/usr/bin/env python
"""StreamGuard v2 / v2-R live demo (lab PC GPU; read-only: nothing is written to $DSG_RESULTS).

Loads gemma-2-2b-it, the FROZEN StreamGuard v2 gate (L08 SAE-CUSUM, threshold 7.084529876708984, exp/V2-test cc6f375)
and the FROZEN v2-R refusal setting (E5b arm B: refusal direction at layer 10, 0.45 x mean residual norm, added on the
prompt and the first 8 generated tokens; exp/V2R-test 16416f7). The code is imported from the exp/V2R-test worktree
(--code-root), and the selection + direction sha256 are checked exactly as V2R's frozen guard does.

For each prompt: gate score on the prompt, threshold, FIRED (on the prompt or at generated token k, the gate is checked
after every token as in v2-R) or NOT FIRED; then three answers: base (ungated), v2 clamp, v2-R self-refusal.

  python scripts/refusal_demo.py --prompt "Explain how a refrigerator works." --force-fire --sample 3
  python scripts/refusal_demo.py --file prompts.txt
  python scripts/refusal_demo.py --wmdp-dev-index 12

--wmdp-dev-index K (WMDP-Bio open-ended DEV item K, as in V2R's prompts): the question, the base answer and the
clamp answer are NEVER printed; only the gate line and the self-refusal answer if it is a refusal (E5's refusal
grader), else "[withheld]". Typed prompts (--prompt / --file) are printed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("DSG_WORKTREES", "/home/amaloch/projects/mechunlearn-project/dsg_worktrees")) / "exp" / "V2R-test"
FROZEN = {"layer": 10, "s": 0.45, "mode": "start8", "limit": 8,
          "direction_sha256": "f7d9a17e803a2ed4b526db1a4db70360e00229049b77893971b153ab1bb0fc0b"}
STEP1_EXP = "E5b"
TEMPERATURE = 0.7


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("inputs (at least one)")
    src.add_argument("--prompt", action="append", default=[], help="typed prompt (repeatable)")
    src.add_argument("--file", type=Path, help="one prompt per line (blank lines and lines starting with # skipped)")
    src.add_argument("--wmdp-dev-index", type=int, action="append", default=[], metavar="K",
                     help="WMDP-Bio open DEV item K (repeatable); question / base / clamp never printed")
    p.add_argument("--force-fire", action="store_true", help="apply the responses even if the gate does not fire")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help=f"also N sampled self-refusal answers (temperature {TEMPERATURE}, refusal arm only)")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--code-root", type=Path, default=DEFAULT_ROOT, help="exp/V2R-test worktree (frozen code)")
    a = p.parse_args(argv)
    if not (a.prompt or a.file or a.wmdp_dev_index):
        p.error("give --prompt, --file or --wmdp-dev-index")
    if a.sample < 0 or a.max_new_tokens < 1:
        p.error("--sample must be >= 0 and --max-new-tokens >= 1")
    if any(k < 0 for k in a.wmdp_dev_index):
        p.error("--wmdp-dev-index must be >= 0")
    return a


def read_prompts(a) -> list[str]:
    out = list(a.prompt)
    if a.file:
        out += [l.rstrip("\n") for l in a.file.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return out


def _gen_rows(model, seq, n_left, rows, clamp, layer, temperature=None, generator=None):
    """Greedy (temperature None) or sampled continuation; same hooks as experiments.E5.batchgen.continue_batch."""
    import torch
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    from experiments.E5.batchgen import BatchClamp, _set_t, _steer_hook, end_ids

    if temperature is None:
        from experiments.E5.batchgen import continue_batch
        return [o for o, _ in continue_batch(model, seq, n_left, rows, clamp, layer)]
    dev = model.cfg.device
    end = end_ids(model)
    R = len(rows)
    hooks = []
    model.reset_hooks()
    try:
        if any(r["clamp"] for r in rows):
            sae, feats, mult, hook_name = clamp
            model.add_hook(hook_name, BatchClamp(sae, feats, torch.tensor([r["clamp"] for r in rows], device=dev), mult))
        if any(r["vec"] is not None for r in rows):
            d = model.cfg.d_model
            V = torch.stack([r["vec"].float().to(dev) if r["vec"] is not None else torch.zeros(d, device=dev) for r in rows])
            h = _steer_hook(V, [r.get("limit") for r in rows])
            model.add_hook(f"blocks.{layer}.hook_resid_post", h)
            hooks.append(h)
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, R)
        last = model(seq.repeat(R, 1), past_kv_cache=cache)[:, -1]
        outs, live = [[] for _ in range(R)], [True] * R
        for k in range(1, n_left + 1):
            probs = torch.softmax(last.float() / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).view(-1)
            for i, t in enumerate(nxt.tolist()):
                if live[i]:
                    outs[i].append(t)
                    if t in end:
                        live[i] = False
            if not any(live):
                break
            _set_t(hooks, k)
            last = model(nxt.view(R, 1), past_kv_cache=cache)[:, -1]
    finally:
        model.reset_hooks()
    return outs


class Demo:
    def __init__(self, code_root: Path):
        sys.path.insert(0, str(code_root))
        import torch

        from dsgx.models.loader import get_bundle
        from experiments.E5.common import cache_path
        from experiments.E5.gen import load_gate
        from experiments.E5.refusal import load_dir
        from experiments.V2.common import DSG_FEATURES, THRESHOLD

        self.torch = torch
        try:
            self.commit = subprocess.run(["git", "-C", str(code_root), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()
        except OSError:
            self.commit = "?"
        s1 = json.loads((Path(os.environ["DSG_RESULTS"]) / "runs" / STEP1_EXP / "refusal" / "metrics.json").read_text())
        sel = s1["selection"]
        got = (int(sel["layer"]), float(sel["s"]), sel.get("mode", "all"))
        if got != (FROZEN["layer"], FROZEN["s"], FROZEN["mode"]):
            raise SystemExit(f"E5b selection {got} != frozen v2-R setting")
        h = hashlib.sha256(cache_path(STEP1_EXP, "base").read_bytes()).hexdigest()
        if h != FROZEN["direction_sha256"]:
            raise SystemExit(f"refusal direction sha256 {h} != frozen")
        self.b = get_bundle()
        self.m = self.b.model
        self.gate = load_gate(self.b)
        self.threshold = THRESHOLD
        d, norm = load_dir(STEP1_EXP, "base", FROZEN["layer"])
        self.norm = norm
        self.vec = FROZEN["s"] * norm * torch.tensor(d, dtype=torch.float32, device=self.b.device)
        self.clamp = (self.b.sae, DSG_FEATURES, 500.0, self.b.hook_name)

    def run(self, prompt_text: str, force: bool, max_new: int, n_sample: int, seed: int) -> dict:
        from experiments.E5.batchgen import L08Detector, phase1
        from experiments.E5.common import is_refusal

        torch, m = self.torch, self.m
        toks = m.to_tokens(prompt_text, prepend_bos=False).to(self.b.device)
        det = L08Detector(*self.gate)
        base, _, fire, fscore, pscore = phase1(m, toks, "blocks.8.hook_resid_post", det, max_new)
        start = fire if fire is not None else (0 if force else None)
        r = {"prompt_score": pscore, "fire_step": fire, "fire_score": fscore, "forced": fire is None and force,
             "base": base, "clamp": None, "refusal": None, "samples": []}
        dec = lambda t: m.tokenizer.decode(t, skip_special_tokens=True).strip()  # noqa: E731
        if start is None:
            r["clamp"] = r["refusal"] = list(base)
        else:
            prefix = base[:start]
            seq = torch.cat([toks, torch.tensor([prefix], dtype=torch.long, device=self.b.device)], 1) if prefix else toks
            rows = [{"clamp": True, "vec": None}, {"clamp": False, "vec": self.vec, "limit": FROZEN["limit"]}]
            a_out, b_out = _gen_rows(m, seq, max_new - len(prefix), rows, self.clamp, FROZEN["layer"])
            r["clamp"], r["refusal"] = prefix + a_out, prefix + b_out
            if n_sample:
                g = torch.Generator(device=self.b.device).manual_seed(seed)
                srows = [{"clamp": False, "vec": self.vec, "limit": FROZEN["limit"]}] * n_sample
                r["samples"] = [prefix + o for o in _gen_rows(m, seq, max_new - len(prefix), srows, self.clamp,
                                                              FROZEN["layer"], TEMPERATURE, g)]
        for k in ("base", "clamp", "refusal"):
            r[k] = dec(r[k])
        r["samples"] = [dec(s) for s in r["samples"]]
        r["is_refusal"] = is_refusal(r["refusal"])
        r["samples_refusal"] = [is_refusal(s) for s in r["samples"]]
        return r

    def gate_line(self, r) -> str:
        t = f"gate score (prompt) {r['prompt_score']:.4f} | threshold {self.threshold:.4f} | "
        if r["fire_step"] == 0:
            return t + "FIRED (on the prompt)"
        if r["fire_step"] is not None:
            return t + f"FIRED (at generated token {r['fire_step']}, score {r['fire_score']:.4f})"
        return t + "NOT FIRED" + (" (--force-fire: responses applied from the prompt)" if r["forced"] else "")


def main(argv=None):
    a = parse_args(argv)
    prompts = read_prompts(a)
    demo = Demo(a.code_root)
    print(f"# StreamGuard v2 gate (frozen, threshold {demo.threshold}) + v2-R self-refusal (L{FROZEN['layer']}, "
          f"{FROZEN['s']} x norm {demo.norm:.2f}, prompt + first {FROZEN['limit']} tokens); code {demo.commit}; "
          f"max_new {a.max_new_tokens}" + (f"; {a.sample} samples at T {TEMPERATURE}, seed {a.seed}" if a.sample else ""))
    from experiments.E5.data import chat

    for i, p in enumerate(prompts):
        r = demo.run(chat(p), a.force_fire, a.max_new_tokens, a.sample, a.seed)
        print(f"\n=== prompt {i + 1}: {p}")
        print(demo.gate_line(r))
        print(f"--- base (ungated):\n{r['base']}")
        print(f"--- v2 clamp:\n{r['clamp']}")
        print(f"--- v2-R self-refusal (greedy):\n{r['refusal']}")
        for j, s in enumerate(r["samples"]):
            print(f"--- v2-R self-refusal sample {j + 1} (T {TEMPERATURE}):\n{s}")
    if a.wmdp_dev_index:
        from dsgx.eval.openqa import item_set, open_prompt

        items = item_set("wmdp-bio-open", "dev")
        for k in a.wmdp_dev_index:
            if k >= len(items):
                raise SystemExit(f"--wmdp-dev-index {k}: only {len(items)} DEV items")
            it = items[k]
            r = demo.run(open_prompt(it.question), a.force_fire, a.max_new_tokens, a.sample, a.seed)
            print(f"\n=== WMDP-Bio DEV item {k} (id {it.item_id}; question, base and clamp answers withheld)")
            print(demo.gate_line(r))
            print("--- v2-R self-refusal (greedy):\n" + (r["refusal"] if r["is_refusal"] else "[withheld]"))
            for j, (s, ok) in enumerate(zip(r["samples"], r["samples_refusal"])):
                print(f"--- v2-R self-refusal sample {j + 1} (T {TEMPERATURE}):\n" + (s if ok else "[withheld]"))


if __name__ == "__main__":
    main()
