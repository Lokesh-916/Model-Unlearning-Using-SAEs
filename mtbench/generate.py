"""Generate MT-Bench answers with gemma-2-2b-it, without ('base') or with ('dsg') the DSG guardrail.

Writes one JSON line per (question, turn) to mtbench/results/answers_<mode>.jsonl and resumes if
the file already exists. Greedy decoding, max 300 new tokens per turn.

DSG mode uses the Bio configuration of the main reproduction (20 features, clamp 500, gate threshold
from the cached metrics file). The gate is decided once on the prompt (the conversation so far), as in the
paper's evaluation. If it is on, the repo's clamp is applied to the prompt tokens and the same 20 features
are clamped on every generated token; if off, nothing is changed. Generation uses the KV cache.
"""
import argparse, contextlib, io, json, os, pickle, sys, time
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))
from transformer_lens import HookedTransformer
from sae_lens import SAE
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["base", "dsg"], required=True)
ap.add_argument("--max_new", type=int, default=300)
ap.add_argument("--limit", type=int, default=None, help="only first N questions (smoke test)")
args = ap.parse_args()

PKL = ("artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it/gemma-scope-2b-pt-res_layer_3/"
       "width_16k/average_l0_142/results/metrics/"
       "clamp_feature_activation_multiplier500_nfeatures20_layer3_retainthres95_seed0.pkl")
p = pickle.load(open(PKL, "rb"))["ablate_params"]
FEATS, MULT, TAU = list(p["features_to_ablate"]), p["multiplier"], p["activation_threshold"]

out_path = f"mtbench/results/answers_{args.mode}.jsonl"
os.makedirs("mtbench/results", exist_ok=True)
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        d = json.loads(l); done.add((d["question_id"], d["turn"]))

questions = [json.loads(l) for l in open("mtbench/data/question.jsonl")]
if args.limit:
    questions = questions[: args.limit]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"mode={args.mode} questions={len(questions)} already_done={len(done)} tau={TAU:.4f}")
model = HookedTransformer.from_pretrained("gemma-2-2b-it", device="cuda", dtype=torch.bfloat16)
tok = model.tokenizer
END = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
stats = {}

if args.mode == "dsg":
    sae, _, _ = SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id="layer_3/width_16k/average_l0_142", device="cuda")
    HOOK = sae.cfg.metadata.hook_name

    def hook(resid, hook):
        with torch.no_grad():
            if resid.shape[1] > 1:  # prompt pass: decide the gate, clamp like the paper's code
                acts = sae.encode(resid); acts[:, 0, :] = 0
                mask = (acts[:, :, FEATS] > 0).sum(2) > 0
                rate = (mask.sum(1) / mask.shape[1]).item()
                stats["rate"] = rate
                stats["gate_on"] = rate > TAU
                if not stats["gate_on"]:
                    return resid
                with contextlib.redirect_stdout(io.StringIO()):
                    return anthropic_clamp_resid_SAE_features(resid, hook, sae, FEATS, MULT, TAU)
            if not stats.get("gate_on"):  # generated token, gate off
                return resid
            acts = sae.encode(resid)      # generated token, gate on: clamp the target features
            error = resid - sae.decode(acts)
            tgt = acts[:, :, FEATS]
            acts[:, :, FEATS] = torch.where(tgt > 0, torch.full_like(tgt, -MULT), tgt)
            stats["clamped_tokens"] = stats.get("clamped_tokens", 0) + int((tgt > 0).any(2).sum().item())
            return sae.decode(acts) + error

@torch.no_grad()
def respond(messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    toks = model.to_tokens(text, prepend_bos=False)
    n_prompt = toks.shape[1]
    stats.clear()
    if args.mode == "dsg":
        with model.hooks(fwd_hooks=[(HOOK, hook)]):
            out = model.generate(toks, max_new_tokens=args.max_new, temperature=0.0, stop_at_eos=True,
                                 eos_token_id=END, verbose=False, return_type="tokens")
    else:
        out = model.generate(toks, max_new_tokens=args.max_new, temperature=0.0, stop_at_eos=True,
                             eos_token_id=END, verbose=False, return_type="tokens")
    new = out[0, n_prompt:]
    reply = tok.decode(new, skip_special_tokens=True).strip()
    return reply, len(new)

t_start = time.time()
n = 0
with open(out_path, "a") as f:
    for q in questions:
        msgs = []
        for turn, user in enumerate(q["turns"], start=1):
            if (q["question_id"], turn) in done:
                # reload earlier answer to keep the conversation history consistent
                prev = [json.loads(l) for l in open(out_path)]
                reply = next(d["answer"] for d in prev if d["question_id"] == q["question_id"] and d["turn"] == turn)
                msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
                continue
            msgs.append({"role": "user", "content": user})
            t0 = time.time()
            reply, n_tok = respond(msgs)
            msgs.append({"role": "assistant", "content": reply})
            toks_ids = tok.encode(reply, add_special_tokens=False)
            row = dict(mode=args.mode, question_id=q["question_id"], category=q["category"], turn=turn,
                       answer=reply, new_tokens=n_tok, seconds=round(time.time() - t0, 2),
                       distinct_token_ratio=round(len(set(toks_ids)) / max(1, len(toks_ids)), 4),
                       gate_on=bool(stats.get("gate_on", False)),
                       activation_rate=round(stats["rate"], 4) if "rate" in stats else None,
                       clamped_tokens=stats.get("clamped_tokens", 0))
            f.write(json.dumps(row) + "\n"); f.flush()
            n += 1
            el = time.time() - t_start
            log(f"q{q['question_id']} t{turn} {q['category']:<11} {n_tok:>3} tok {row['seconds']:>5.1f}s "
                f"gate_on={row['gate_on']} | {n} answers, {el/60:.1f} min elapsed")
log(f"DONE mode={args.mode} new_answers={n} total_minutes={(time.time()-t_start)/60:.1f}")
