# StreamGuard v2: single TEST evaluation (lab PC)

**Status: RUNNING (queued 2026-11-13 11:46 IST; ETA ~2026-11-14 05:00-06:00 IST).** This file is replaced by
`python -m dsgx.analysis.v2_test --out docs/V2_TEST_RESULT.md` when every V2-devcheck and V2-test job is DONE
(`scripts/v2_test_when_done.sh` does this, commits and pushes; it never re-runs anything).

**Frozen before any TEST result** (2026-11-13T11:46:09+05:30): branch `exp/V2-test` commit
cc6f375e944e65420099f9d12d699c15d5e9d26e (pushed); DEVIATIONS row 2026-11-13.

- StreamGuard v2 = E4a L08 SAE-CUSUM gate, op5, DSG's -500 clamp at layer 3, two-pass.
- Threshold moved from the tie 7.06202507019043 to the midpoint **7.084529876708984** (next DEV score
  7.107034683227539); all 11,582 logged DEV decisions unchanged (offline); online DEV smoke 183 / 183.
- Queue: V2-devcheck (full DEV replay; its check task blocks TEST on any decision change), then V2-test (31 jobs,
  186 runs, ~148k items, one GPU job at a time, bs 1, item-level resume).
- Claims: `claims.evaluate_v2` V2 and V3 unchanged; V1 does not apply (response = clamp); V4 not run.
