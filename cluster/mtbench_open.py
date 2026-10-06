"""MT-Bench with an OPEN judge that fits 48 GB, fully offline (decision 9: one fixed judge for all
comparisons; absolute scores are not comparable with the paper's GPT-4-judged 7.78).

Phase gen (gemma-2-2b-it, TransformerLens, greedy, max 1024 new tokens per turn, chat template):
  base        no gate
  dsg         canonical streaming DSG gate (dsgx.gen.stream, paper config N=20 / 95 / x500, rho > tau)
  window-w16  our window gate (same 20 features and clamp; score = max fire fraction over 16-token windows,
              gates.score_window; threshold = gates.calibrate on MMLU DEV prompts at 5% FPR, as A7's best fix),
              re-evaluated at every generated token like dsg
  cusum       the final method (X1 combined gate, session 22): DSG's 20 features + clamp, CUSUM detector
              (gates.stream_gate: per-token LLR, score = max CUSUM), threshold = gates.calibrate at 5% FPR on MMLU DEV
              (the X1 MCQ gate's cached threshold); paired vs dsg and vs base in summary.json
  80 questions x 2 turns; turn 2 sees the model's own turn-1 answer.
Phase judge (FIXED, never change between conditions):
  judge model  JUDGE = google/gemma-2-9b-it (bf16, ~18 GB; user decision 2026-10-04, BM3). Same model family
               as the judged gemma-2-2b-it -> self-preference risk; labelled `same_family_judge` (DEVIATIONS).
               (unsloth/Qwen2.5-32B-Instruct-bnb-4bit was the original choice; not downloaded.)
  prompts      FastChat single-answer grading (mtbench/data/judge_prompts.jsonl: single-v1 / single-math-v1
               (+ -multi-turn), reference answers mtbench/data/reference_answer/gpt-4.jsonl for math, reasoning,
               coding), system prompt as given, judge chat template, greedy, max 512 new tokens,
               score = [[x]] (fallback [x]); unparsable -> None (counted, reported).
Resumable: answers_<mode>.jsonl / judgments_<mode>__<judge>.jsonl rows already present are skipped; each sbatch
stops starting new work when DSG_BUDGET_MIN (slurm/later.sh) runs low and prints "partial" (chain several copies).
Outputs: $DSG_RESULTS/jobs/mtbench/ (answers are benign MT-Bench text) and summary.json with mean score
per condition and turn, 95% bootstrap CIs over questions, and the paired difference dsg - base.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.eval import stats

NAME = "mtbench"
JUDGE = "google/gemma-2-9b-it"
MODES = ["base", "dsg", "window-w16", "cusum"]
GEN_MIN, JUDGE_MIN = 2.0, 0.5   # minutes reserved per answer / per judgment before starting it (budget check)
NEED_REF = {"math", "reasoning", "coding"}
DATA = paths.REPO_ROOT / "mtbench" / "data"


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def append(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj) + "\n")


def window_gate(b, w=16):
    """(features, threshold, score_fn) of the window gate: harness Gate + calibrate (cached threshold)."""
    from dsgx.methods import gates

    g = gates.Gate({"type": "window", "w": w, "n_features": 20, "retain_pct": 95, "case": "bio"}, b)
    cal = gates.calibrate(b, g, "bio", 0.05, 1000, "mmlu-dev")
    return g.features, float(cal["threshold"]), (lambda fires: gates.score_window(torch.tensor(fires), len(fires), w)), cal


def cusum_gate(b):
    """(features, threshold, score_fn, token_fn, calib) of the X1 combined gate (CUSUM detector) for streaming."""
    from dsgx.methods import gates

    g = gates.Gate({"type": "cusum", "n_features": 20, "retain_pct": 95, "case": "bio"}, b)
    cal = gates.calibrate(b, g, "bio", 0.05, 1000, "mmlu-dev")
    return (*gates.stream_gate(g), cal)


def generate_all(modes, max_new, limit, budget):
    out = jc.job_dir(NAME)
    qs = rows(DATA / "question.jsonl")[: limit or None]
    if jc.TINY:
        model = jc.load_lm()
        tok = jc.load_tok()

        def gen(text, mode):
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        from dsgx.gen.stream import generate
        from dsgx.models.loader import get_bundle

        b = get_bundle()
        tok = b.model.tokenizer
        feats, tau = jc.dsg_features()
        win, cus = None, None

        def gen(text, mode):
            nonlocal win, cus
            if mode == "cusum":
                if cus is None:
                    cus = cusum_gate(b)
                    jc.log(NAME, f"cusum gate: {len(cus[0])} features, threshold {cus[1]:.4f} ({cus[4].get('source')}, "
                                 f"empirical FPR {cus[4].get('empirical_fpr')})")
                r = generate(b.model, text, b, cus[0], 500.0, cus[1], "stream", max_new, score_fn=cus[2], token_fn=cus[3])
            elif mode.startswith("window-w"):
                if win is None:
                    win = window_gate(b, int(mode.split("-w")[1]))
                    jc.log(NAME, f"window gate: {len(win[0])} features, threshold {win[1]:.4f} ({win[3].get('source')}, "
                                 f"empirical FPR {win[3].get('empirical_fpr')})")
                r = generate(b.model, text, b, win[0], 500.0, win[1], "stream", max_new, score_fn=win[2])
            else:
                r = generate(b.model, text, b, feats, 500.0, tau, "none" if mode == "base" else "stream", max_new)
            return tok.decode(r.tokens, skip_special_tokens=True)
    complete = True
    for mode in modes:
        p = out / f"answers_{mode}.jsonl"
        done = {(r["question_id"], r["turn"]) for r in rows(p)}
        prev = {(r["question_id"], r["turn"]): r["answer"] for r in rows(p)}
        for q in qs:
            msgs = []
            for t, turn in enumerate(q["turns"], start=1):
                msgs.append({"role": "user", "content": turn})
                key = (q["question_id"], t)
                if key not in done:
                    if not budget.fits(GEN_MIN):
                        complete = False
                        break
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    ans = gen(text, mode)
                    append(p, {"mode": mode, "question_id": q["question_id"], "category": q["category"], "turn": t, "answer": ans})
                    prev[key] = ans
                msgs.append({"role": "assistant", "content": prev[key]})
            if not complete:
                break
        jc.log(NAME, f"answers {mode}: {len(rows(p))} rows")
        if not complete:
            break
    return complete


def build(q, turn, ans, refs, prompts):
    use_ref = q["category"] in NEED_REF and q["question_id"] in refs
    name = ("single-math-v1" if use_ref else "single-v1") + ("-multi-turn" if turn == 2 else "")
    pr = prompts[name]
    ref = refs.get(q["question_id"], ["", ""]) if use_ref else ["", ""]
    if turn == 1:
        user = pr["prompt_template"].format(question=q["turns"][0], answer=ans[(q["question_id"], 1)], ref_answer_1=ref[0])
    else:
        user = pr["prompt_template"].format(question_1=q["turns"][0], question_2=q["turns"][1], answer_1=ans[(q["question_id"], 1)],
                                            answer_2=ans[(q["question_id"], 2)], ref_answer_1=ref[0], ref_answer_2=ref[1])
    return name, pr["system_prompt"], user


def judge_all(modes, judge, limit, budget):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = jc.job_dir(NAME)
    qs = {q["question_id"]: q for q in rows(DATA / "question.jsonl")}
    refs = {r["question_id"]: r["choices"][0]["turns"] for r in rows(DATA / "reference_answer" / "gpt-4.jsonl")}
    prompts = {p["name"]: p for p in rows(DATA / "judge_prompts.jsonl")}
    if jc.TINY:
        model, tok = jc.load_lm(), jc.load_tok()
    else:
        tok = AutoTokenizer.from_pretrained(judge)
        model = AutoModelForCausalLM.from_pretrained(judge, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tag = judge.split("/")[-1]
    complete = True
    for mode in modes:
        ans = {(r["question_id"], r["turn"]): r["answer"] for r in rows(out / f"answers_{mode}.jsonl")}
        p = out / f"judgments_{mode}__{tag}.jsonl"
        done = {(r["question_id"], r["turn"]) for r in rows(p)}
        todo = [k for k in sorted(ans) if k not in done and (k[1] == 1 or (k[0], 1) in ans)][: limit or None]
        for qid, turn in todo:
            if not budget.fits(JUDGE_MIN):
                complete = False
                break
            name, system, user = build(qs[qid], turn, ans, refs, prompts)
            try:  # judges without a system role (e.g. Gemma) get the system prompt folded into the user turn
                text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                               tokenize=False, add_generation_prompt=True)
            except Exception:
                text = tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}], tokenize=False,
                                               add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=512 if not jc.TINY else 8, do_sample=False)
            jt = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\[\[(\d+\.?\d*)\]\]", jt) or re.search(r"\[(\d+\.?\d*)\]", jt)
            append(p, {"question_id": qid, "turn": turn, "category": qs[qid]["category"], "judge": judge, "prompt": name,
                       "score": float(m.group(1)) if m else None, "judgment": jt})
        jc.log(NAME, f"judged {mode}: {len(rows(p))} rows")
        if not complete:
            break
    return tag, complete


def summarise(modes, tag):
    out = jc.job_dir(NAME)
    res = {}
    sc = {}
    for mode in modes:
        r = rows(out / f"judgments_{mode}__{tag}.jsonl")
        sc[mode] = {(x["question_id"], x["turn"]): x["score"] for x in r}
        vals = [v for v in sc[mode].values() if v is not None]
        res[mode] = {"all": stats.bootstrap_ci(vals), "unparsed": sum(v is None for v in sc[mode].values())}
        for t in (1, 2):
            res[mode][f"turn{t}"] = stats.bootstrap_ci([v for (q, tt), v in sc[mode].items() if tt == t and v is not None])
    for m in [x for x in sc if x != "base"]:
        if "base" in sc:
            keys = [k for k in sc["base"] if sc["base"][k] is not None and sc[m].get(k) is not None]
            res[f"{m}_minus_base"] = stats.paired_bootstrap([sc[m][k] for k in keys], [sc["base"][k] for k in keys])
    for m in [x for x in sc if x not in ("base", "dsg")]:
        if "dsg" in sc:
            keys = [k for k in sc["dsg"] if sc["dsg"][k] is not None and sc[m].get(k) is not None]
            res[f"{m}_minus_dsg"] = stats.paired_bootstrap([sc[m][k] for k in keys], [sc["dsg"][k] for k in keys])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "judge", "all"], default="all")
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    jc.require_gpu(40)
    budget = jc.Budget(a.budget_min)
    if a.phase in ("gen", "all"):
        ok = generate_all(a.modes, a.max_new if not jc.TINY else 4, a.limit, budget)
        torch.cuda.empty_cache()
        from dsgx.models.loader import clear

        clear()
        if not ok:
            jc.log(NAME, "partial: answers incomplete (budget); resubmit mtbench.sbatch to continue")
            return 0
    if a.phase in ("judge", "all"):
        tag, ok = judge_all(a.modes, a.judge, a.limit, budget)
        if not ok:
            jc.log(NAME, "partial: judgments incomplete (budget); resubmit mtbench.sbatch to continue")
            return 0
        res = summarise(a.modes, tag)
        jc.summary(NAME, {"judge": a.judge, "settings": {"temperature": 0, "max_new_judge": 512, "max_new_answer": a.max_new,
                                                       "prompts": "FastChat single-v1 / single-math-v1"},
                          "same_family_judge": "gemma" in a.judge, "scores": res,
                          "note": "absolute scores are not comparable with the paper's GPT-4-judged 7.78"})
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
