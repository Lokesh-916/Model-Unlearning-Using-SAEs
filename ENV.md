# Environment (decision 10)

Created 2026-10-01 for the v2 harness.

## Conda env `mechunlearn2`

```
conda create -y --clone mechunlearn -n mechunlearn2
/home/amaloch/miniconda3/envs/mechunlearn2/bin/python -m pip install --no-deps -e dynamic_sae_guardrails
/home/amaloch/miniconda3/envs/mechunlearn2/bin/python -m pip install pytest
```

- `--no-deps` is deliberate. `dynamic_sae_guardrails/pyproject.toml` pins `numpy<2.0` and lists
  packages we do not use (`collectibles`, `openai`, plotting/dev tools). Installing them would
  downgrade numpy under torch 2.11 / pandas 3 / sae-lens 6.50. Decision 10 says keep numpy
  unless it breaks something; nothing broke (the sanity gate reproduces bit-exactly), so numpy
  stays at 2.4.6. Unpickling the legacy metric pickles prints `numpy.core` deprecation warnings;
  they are harmless.
- The original env `mechunlearn` is untouched.
- Exact package versions: `requirements-lock.txt` (`pip freeze`).

| package | version |
|---|---|
| Python | 3.11.15 |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| transformers | 5.16.1 |
| transformer-lens | 3.8.1 |
| sae-lens | 6.50.0 |
| numpy | 2.4.6 |
| datasets | 5.0.1 |
| pandas / pyarrow | 3.0.5 / 25.0.1 |
| pytest | 9.1.1 (added) |

## Hardware

NVIDIA RTX 2000 Ada (16 GB), driver 580.173.02, 62 GB RAM, 36 CPUs.

## Shared locations (MASTER_PLAN 3.2)

| variable | path |
|---|---|
| `DSG_CACHE` | `/home/amaloch/projects/mechunlearn-project/dsg_cache` |
| `DSG_RESULTS` | `/home/amaloch/projects/mechunlearn-project/dsg_results` |
| `DSG_PRIVATE` | `/home/amaloch/projects/mechunlearn-project/dsg_private` |
| `DSG_WORKTREES` | `/home/amaloch/projects/mechunlearn-project/dsg_worktrees` |

`scripts/env.sh` exports these plus `DSGX_PY` (the env's python) and, by default, `HF_HUB_OFFLINE=1`
and `HF_DATASETS_OFFLINE=1` (everything the harness needs is in `~/.cache/huggingface`; offline mode
avoids minutes of dataset revalidation per job). Set `DSGX_OFFLINE=0` for jobs that must download.

## Notes

- `dsgx` is **not** pip-installed: every job imports it from its own worktree (`python -m dsgx...`
  with the worktree as cwd), so jobs run exactly the pinned commit.
- `[DSG DEBUG]` prints in the library's clamp hook are off unless `DSG_DEBUG=1`.
- Model load peaks at about 19 GB host RAM (TransformerLens loads the HF weights on CPU first);
  steady state is about 7 GB VRAM for gemma-2-2b-it + the layer-3 SAE in bf16.
