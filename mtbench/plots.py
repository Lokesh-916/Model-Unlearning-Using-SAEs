"""Build summary numbers and figures from the stored MT-Bench answers and judgments (no API calls)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "mtbench/results", "mtbench/figures"
os.makedirs(F, exist_ok=True)
PAPER = 7.78
load = lambda p: [json.loads(l) for l in open(p)]
J = {m: {(r["question_id"], r["turn"]): r for r in load(f"{R}/judgments_{m}.jsonl") if r["score"] is not None} for m in ("base", "dsg")}
A = {m: {(r["question_id"], r["turn"]): r for r in load(f"{R}/answers_{m}.jsonl")} for m in ("base", "dsg")}
keys = sorted(set(J["base"]) & set(J["dsg"]))
cats = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
col = {"base": "#4C78A8", "dsg": "#E45756"}
lab = {"base": "Base gemma-2-2b-it", "dsg": "With DSG guardrail"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

sc = lambda m, ks: np.array([J[m][k]["score"] for k in ks])
mean = {m: sc(m, keys).mean() for m in J}
sem = {m: sc(m, keys).std(ddof=1) / np.sqrt(len(keys)) for m in J}
by_cat = {m: [np.mean([J[m][k]["score"] for k in keys if J[m][k]["category"] == c]) for c in cats] for m in J}
by_turn = {m: [np.mean([J[m][k]["score"] for k in keys if k[1] == t]) for t in (1, 2)] for m in J}
diff = sc("dsg", keys) - sc("base", keys)
summary = dict(n_judgments=len(keys), judge=next(iter(J["base"].values()))["judge"], paper_dsg_score=PAPER,
               mean=mean, sem=sem, by_category=dict(zip(cats, zip(by_cat["base"], by_cat["dsg"]))),
               by_turn=by_turn, answers_changed_score=int((diff != 0).sum()),
               gate_on_answers=int(sum(A["dsg"][k]["gate_on"] for k in keys)),
               cost_usd=sum(r["cost_usd"] for m in J for r in J[m].values()))
json.dump(summary, open(f"{R}/summary.json", "w"), indent=2)

# 1 overall
fig, ax = plt.subplots(figsize=(5, 4))
for i, m in enumerate(("base", "dsg")):
    ax.bar(i, mean[m], yerr=1.96 * sem[m], color=col[m], capsize=5)
    ax.text(i, mean[m] + 0.25, f"{mean[m]:.2f}", ha="center", fontweight="bold")
ax.axhline(PAPER, ls="--", color="gray"); ax.text(1.45, PAPER + 0.08, f"paper DSG {PAPER}", ha="right", color="gray")
ax.set_xticks([0, 1], [lab["base"], lab["dsg"]]); ax.set_ylim(0, 10); ax.set_ylabel("MT-Bench score (1-10)")
ax.set_title("Overall MT-Bench score (error bars: 95% CI)"); fig.tight_layout(); fig.savefig(f"{F}/1_overall_score.png"); plt.close(fig)

# 2 per category
fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(cats)); w = 0.38
for i, m in enumerate(("base", "dsg")):
    ax.bar(x + (i - 0.5) * w, by_cat[m], w, label=lab[m], color=col[m])
ax.set_xticks(x, cats, rotation=20); ax.set_ylabel("mean score"); ax.set_ylim(0, 10); ax.legend(frameon=False, ncol=2)
ax.set_title("Score by MT-Bench category"); fig.tight_layout(); fig.savefig(f"{F}/2_score_by_category.png"); plt.close(fig)

# 3 per turn
fig, ax = plt.subplots(figsize=(5, 4)); x = np.arange(2)
for i, m in enumerate(("base", "dsg")):
    ax.bar(x + (i - 0.5) * w, by_turn[m], w, label=lab[m], color=col[m])
ax.set_xticks(x, ["Turn 1", "Turn 2 (follow-up)"]); ax.set_ylim(0, 10); ax.set_ylabel("mean score"); ax.legend(frameon=False)
ax.set_title("First question vs follow-up"); fig.tight_layout(); fig.savefig(f"{F}/3_score_by_turn.png"); plt.close(fig)

# 4 per-answer score change
fig, ax = plt.subplots(figsize=(6, 4))
vals, cnt = np.unique(diff, return_counts=True)
ax.bar(vals, cnt, width=0.8, color=["#59A14F" if v > 0 else "#E45756" if v < 0 else "#9E9E9E" for v in vals])
ax.set_xlabel("score with DSG minus score without (per answer)"); ax.set_ylabel("number of answers")
ax.set_title(f"How much DSG changed each answer's score ({(diff == 0).sum()} of {len(diff)} unchanged)")
fig.tight_layout(); fig.savefig(f"{F}/4_score_change_histogram.png"); plt.close(fig)

# 5 gate activation vs score change
tau = 0.5458
rate = np.array([A["dsg"][k]["activation_rate"] or 0 for k in keys])
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(rate, diff + np.random.default_rng(0).normal(0, 0.05, len(diff)), s=18, alpha=0.6, color=col["dsg"])
ax.axvline(tau, ls="--", color="gray"); ax.text(tau + 0.01, ax.get_ylim()[1] * 0.9, "gate threshold", color="gray")
ax.set_xlabel("hazard-feature activation rate of the prompt"); ax.set_ylabel("score change with DSG (small jitter added)")
ax.set_title("The guardrail only acts on prompts right of the line"); fig.tight_layout(); fig.savefig(f"{F}/5_gate_activation_vs_score_change.png"); plt.close(fig)
print(json.dumps(summary, indent=1)); print("figures saved to", F)
