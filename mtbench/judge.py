"""Score MT-Bench answers with an LLM judge through OpenRouter (single-answer grading, 1-10).

Uses FastChat's own judge prompts (reference-guided for math / reasoning / coding). Every call is logged
with its token usage and cost, and the run stops before a hard spend cap is exceeded. Resumable: rows
already in mtbench/results/judgments_<mode>.jsonl are skipped, and the cap counts spend from earlier runs.

The API key is read from $OPENROUTER_API_KEY or ~/.openrouter_key and is never written anywhere.
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["base", "dsg"], required=True)
ap.add_argument("--judge", default="openai/gpt-4o")
ap.add_argument("--price_in", type=float, default=2.50, help="USD per 1M input tokens (fallback if API omits cost)")
ap.add_argument("--price_out", type=float, default=10.0, help="USD per 1M output tokens")
ap.add_argument("--cap", type=float, default=2.25, help="hard cap in USD across ALL judge runs (both modes)")
ap.add_argument("--limit", type=int, default=None, help="only first N answers (smoke test)")
ap.add_argument("--max_tokens", type=int, default=500)
args = ap.parse_args()

key = os.environ.get("OPENROUTER_API_KEY") or (
    open(os.path.expanduser("~/.openrouter_key")).read().strip() if os.path.exists(os.path.expanduser("~/.openrouter_key")) else None)
if not key:
    sys.exit("No OpenRouter key: set OPENROUTER_API_KEY or put it in ~/.openrouter_key")

R = "mtbench/results"
qs = {q["question_id"]: q for q in map(json.loads, open("mtbench/data/question.jsonl"))}
refs = {r["question_id"]: r["choices"][0]["turns"] for r in map(json.loads, open("mtbench/data/reference_answer/gpt-4.jsonl"))}
prompts = {p["name"]: p for p in map(json.loads, open("mtbench/data/judge_prompts.jsonl"))}
answers = {}
for r in map(json.loads, open(f"{R}/answers_{args.mode}.jsonl")):
    answers[(r["question_id"], r["turn"])] = r["answer"]
NEED_REF = {"math", "reasoning", "coding"}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def total_spent():
    t = 0.0
    for m in ("base", "dsg"):
        p = f"{R}/judgments_{m}.jsonl"
        if os.path.exists(p):
            t += sum(json.loads(l)["cost_usd"] for l in open(p))
    return t

def build(qid, turn):
    q = qs[qid]; use_ref = q["category"] in NEED_REF and qid in refs
    name = ("single-math-v1" if use_ref else "single-v1") + ("-multi-turn" if turn == 2 else "")
    pr = prompts[name]; ref = refs.get(qid, ["", ""]) if use_ref else ["", ""]
    if turn == 1:
        user = pr["prompt_template"].format(question=q["turns"][0], answer=answers[(qid, 1)], ref_answer_1=ref[0])
    else:
        user = pr["prompt_template"].format(question_1=q["turns"][0], question_2=q["turns"][1],
                                            answer_1=answers[(qid, 1)], answer_2=answers[(qid, 2)],
                                            ref_answer_1=ref[0], ref_answer_2=ref[1])
    return name, pr["system_prompt"], user

def call(system, user):
    body = json.dumps({"model": args.judge, "temperature": 0, "max_tokens": args.max_tokens,
                       "usage": {"include": True},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code in (401, 402, 403):
                sys.exit(f"Stopping: HTTP {code} from OpenRouter (key/credit problem)")
            log(f"  call failed ({e}); retry {attempt + 1}/4"); time.sleep(3 * (attempt + 1))
    raise RuntimeError("judge call failed 4 times")

out_path = f"{R}/judgments_{args.mode}.jsonl"
done = {(d["question_id"], d["turn"]) for d in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
todo = [k for k in sorted(answers) if k not in done]
if args.limit:
    todo = todo[: args.limit]
log(f"mode={args.mode} judge={args.judge} to_judge={len(todo)} already_done={len(done)} "
    f"spent_so_far=${total_spent():.4f} cap=${args.cap:.2f}")

t0 = time.time(); n = 0
with open(out_path, "a") as f:
    for qid, turn in todo:
        name, system, user = build(qid, turn)
        worst = (len(system + user) / 3.5) / 1e6 * args.price_in + args.max_tokens / 1e6 * args.price_out
        if total_spent() + worst > args.cap:
            log(f"STOP: next call could push spend past the cap (${total_spent():.4f} + ${worst:.4f} > ${args.cap:.2f})")
            break
        t1 = time.time(); resp = call(system, user)
        text = resp["choices"][0]["message"]["content"]
        m = re.search(r"\[\[(\d+\.?\d*)\]\]", text) or re.search(r"\[(\d+\.?\d*)\]", text)
        score = float(m.group(1)) if m else None
        u = resp.get("usage", {}); pin, pout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = u.get("cost")
        if cost is None:
            cost = pin / 1e6 * args.price_in + pout / 1e6 * args.price_out
        row = dict(mode=args.mode, question_id=qid, category=qs[qid]["category"], turn=turn, judge=args.judge,
                   prompt_name=name, score=score, prompt_tokens=pin, completion_tokens=pout,
                   cost_usd=round(cost, 6), seconds=round(time.time() - t1, 2), judgment=text)
        f.write(json.dumps(row) + "\n"); f.flush(); n += 1
        log(f"q{qid} t{turn} {qs[qid]['category']:<11} score={score} in={pin} out={pout} cost=${cost:.4f} "
            f"| {n}/{len(todo)} total_spent=${total_spent():.4f} elapsed={(time.time() - t0) / 60:.1f}min")
log(f"DONE mode={args.mode} judged={n} spent_total=${total_spent():.4f} minutes={(time.time() - t0) / 60:.1f}")
