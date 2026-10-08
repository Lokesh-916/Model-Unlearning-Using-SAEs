# Dynamic SAE Guardrails

Official implementation of "SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs" (COLM 2025)

<p align="center">
  <img src="figures/SAE_plot.png" alt="Dynamic SAE Guardrails Overview" width="600">
</p>

## Overview

Dynamic SAE Guardrails is a novel approach for targeted knowledge unlearning in language models using Sparse Autoencoders (SAEs). Unlike static feature ablation, our method dynamically intervenes only when feature activation patterns exceed specified thresholds, preserving model capabilities while effectively removing unwanted knowledge.

### Key Features
- **Dynamic intervention**: Activates only when necessary, minimizing impact on general capabilities
- **Precision unlearning**: Achieves state-of-the-art results on WMDP benchmarks
- **Minimal side effects**: Maintains >99% performance on MMLU while unlearning targeted knowledge

### Supported Datasets
- **WMDP (Weapons of Mass Destruction Proxy)**: Full unlearning evaluation for bio/cyber hazardous knowledge
- **MUSE (Machine Unlearning Six-dataset Ensemble)**: Feature identification for books/news datasets


## Setup

### Requirements

- Python 3.8+
- NVIDIA GPU with 80GB+ memory (A100 recommended)
- PyTorch 2.0+

### Data Prerequisites

For WMDP bio evaluation:
1. Request the forget corpus from [this form](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform)
2. Place `bio-forget-corpus.jsonl` in `dynamic_sae_guardrails/evals/unlearning/data/`

For WMDP cyber evaluation, the corpus is automatically downloaded from HuggingFace.

### Installation

```bash
# Clone the repository
git clone https://github.com/aashiqmuhamed/DynamicSAEGuardrails.git
cd DynamicSAEGuardrails

# Install in editable mode
pip install -e ./dynamic_sae_guardrails
```

## Usage

### WMDP Unlearning Evaluation

Evaluate unlearning performance on WMDP bio/cyber datasets:

```bash
python dynamic_sae_guardrails/evals/unlearning/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_3/width_16k/average_l0_142" \
    --model_name gemma-2-2b-it \
    --llm_batch_size 1 \
    --llm_dtype float32 \
    --force_rerun \
    --random_seed 0 \
    --case bio  # or 'cyber' for cyber evaluation
```

### MUSE Feature Identification

Identify important features for MUSE datasets (books/news):

```bash
python dynamic_sae_guardrails/evals/unlearning/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_3/width_16k/average_l0_142" \
    --model_name gemma-2-2b-it \
    --llm_batch_size 1 \
    --llm_dtype float32 \
    --force_rerun \
    --random_seed 0 \
    --case news  # or 'books' for books dataset
```

## Output Structure

```
artifacts_dynamic_bs1_{case}/
└── unlearning/
    └── {model_name}/
        └── {sae_name}/
            └── results/
                ├── sparsities/         # Feature activation patterns
                │   ├── feature_sparsity_forget.txt
                │   └── feature_sparsity_retain.txt
                └── metrics/            # Evaluation metrics (WMDP only)
                    └── *.pkl

eval_results/
└── unlearning_dynamic_bs1/
    └── {sae_name}.json         # Final unlearning score (WMDP only)
```

## Key Parameters

- **`retain_thresholds`**: Percentile thresholds for filtering features (default: [95])
- **`n_features_list`**: Number of features to intervene on (default: [20, 50, 100, 200])
- **`multipliers`**: Negative values for feature clamping (default: [500])

## Notes

- Requires instruct-tuned models (e.g., `gemma-2-2b-it`)
- The unlearning score is calculated as: `1.0 - min(WMDP accuracy)` where MMLU ≥ 99%
- For MUSE datasets, this codebase only identifies features. Full evaluation requires [open-unlearning](https://github.com/locuslab/open-unlearning)

## How to run D1 v3 on gpuws without Claude

D1 v3 is a **prepared** follow-up to D1 v2 (2026-10-08). It has not been submitted. Run everything from the lab PC in
`~/projects/mechunlearn-project/prep` after the "Every new terminal" block (runbook section 0). The gpuws rules in
CLAUDE.md still apply: Slurm only, ~/dsg_cluster only, at least 50 GB free, ours under 100 GB, log every command.

**What it does** (`cluster/d1_v3.py`; the rule is fixed in its docstring before any result):

| part | setting |
|---|---|
| distillation data | WMDP bio forget corpus (DSG-guarded teacher) + WikiText-2 (unguarded teacher). **No benchmark questions** (v2 used geography DEV prompts) |
| corpus split | fixed shuffle (seed 0): 1000 passages held out **only** for the relearning attack; 23,432 **only** for distillation |
| students | noise alpha 0.1 (v2's DEV pick), 4000 steps, **seeds 0, 1, 2** (noise + batch order). No selection: report the 3-seed mean ± sd |
| MCQ utility | **full MMLU** = `@utility` (48 subjects) **plus** the 9 bio hazard-adjacent subjects reported separately; DEV bs 16, TEST bs 1 |
| open-ended | `dsgx.eval.openqa`, greedy, 64 tokens: WMDP-bio leakage (476 items, match rate, lower is better) and benign hazard-adjacent MMLU (352 items, higher is better), for base / DSG (stream gate) / each student. Generations stay in `~/dsg_cluster/private` (never fetched, never opened) |
| relearning | same controls under **both** protocols: dsg-hook, dsg-nohook, rmu-v2, plus the 3 students. Full fine-tune (A6-full: lr 1e-5, 200 steps) and LoRA r 8 / 64 (exp/A6: lr 1e-4, 1000 steps), k = 10 / 100 / 1000 held-out passages, the same eval sets (300 WMDP-bio TEST, 200 `@utility` TEST). 18 full + 36 LoRA cells |
| outputs | `runs/D1-v3`, `runs/D1-v3-relearn-full`, `runs/D1-v3-relearn-lora`, `jobs/d1-v3/{summary.json,SUMMARY.md}` (hardware label gpuws) |

**Cost:** about 21 h of GPU time as 12 chained 3 h sbatch jobs after `validate` (train ×4, test ×2, open ×2,
relearn ×4). Every step resumes, so a budget stop exits 0 and the next copy continues. Peak disk on gpuws is
**32 GB**: 3 students × 5.2, one trainer state of 10, RMU-v2 4.9, corpus 0.7, MiniLM 0.9. Staging copies about 7 GB.
**Server `/` needs ≥ 82 GB free** (50 GB floor + 32 GB). On 2026-10-08 it had 63 GB (other users), so wait.
Never lower the limits.

**Step 1: dry run (submits nothing; about 5 min):**

    scripts/d1_v3_dryrun.sh

You should see: `5 passed` (tiny CPU tests); `PLAN OK` with `"tau": 0.5458`, `"distill": 23432`, `"holdout": 1000`,
`"leak": 476`, `"benign": 352`, `"estimate_hours" ... "total": 21.3` and `"problems": []`; the server plan line
`at peak: free N GB (must stay >= 50)` (N must be ≥ 50, else stop here); and a dry `CHAIN:` line with 13 jobs.

**Step 2: one command (plan, confirm, stage, submit):**

    git status --short          # must be clean: the code snapshot refuses uncommitted changes
    cluster/server.sh run d1-v3 # answer y; if our other jobs are queued, instead run:
                                #   cluster/server.sh stage d1-v3 && cluster/server.sh submit d1-v3 --after-any <last id>

You should see: `staged d1-v3; server free N GB`, `snapshot code-d1v3 = prep/later-runs@<hash>`, `import check ... OK`,
then 13 `submitted ... as job <id>` lines and `chain submitted; last job <id>`. Optional: `scripts/d1_v3_dryrun.sh --server`
repeats the CPU plan on the gpuws login node against the staged snapshot. It must also print `PLAN OK`.

**Step 3: watch (any time):** `cluster/server.sh check d1-v3`. Look for `VALIDATE sanity-gpuws EXACT`, then
`[d1-v3] undo_a0.1_s0: steps 0->4000`, `budget used ... next sbatch resumes`, `trained: {0: True, 1: True, 2: True}`,
`open-leak-...: match {...}`, `relearn-...: forget 0.xxx -> 0.yyy`, and `relearning done; skipped {}`.
If `validate` fails, everything after it is cancelled (afterok) and the inputs stay staged. Fix the cause, then
`cluster/server.sh submit d1-v3`. If the chain ends with work left (the log ends at `budget used`), resubmit only
the unfinished part, e.g. `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh d1-v3-relearn.sbatch'` (runbook 5.1).

**Step 4: fetch, verify, clean (the staged-storage cycle):**

    cluster/server.sh fetch d1-v3 && cluster/server.sh verify d1-v3
    cat $P/dsg_results_cluster/jobs/d1-v3/SUMMARY.md
    cluster/server.sh cleanup d1-v3          # dry run: lists students, RMU-v2, corpus, MiniLM, private/D1-v3, code-d1v3
    cluster/server.sh cleanup d1-v3 --yes

The student weights (15.6 GB) are **not** fetched: lab disk is near its 60 GB line. Copy one only if a later
experiment needs it and the lab disk allows: `rsync --bwlimit=50000 --partial -a gpuws:dsg_cluster/data/dsg_cache/models/D1-v3/undo_a0.1_s0 $P/dsg_results_cluster/checkpoints/D1-v3/`.

**Reporting:** label every number "gpuws". Compare only with gpuws runs: D1 v2, d1-full, A6-full, RMU-v2. Never
mix them with lab-PC numbers. D1 v3 is a follow-up, not a claim input. Using it for C-H6 needs a DEVIATIONS row
and a team decision first. Commits touching D1 carry the Amar060 trailer (CLAUDE.md commit rule 2).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{muhamed2025saes,
  title={SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs},
  author={Muhamed, Aashiq and Bonato, Jacopo and Diab, Mona and Smith, Virginia},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025},
  url={https://arxiv.org/abs/2504.08192}
}
```

## Acknowledgments

This codebase builds upon [SAEBench](https://github.com/adamkarvonen/SAEBench). We thank the authors for their foundational work.