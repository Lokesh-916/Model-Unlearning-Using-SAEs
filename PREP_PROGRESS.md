# PREP_PROGRESS — branch prep/later-runs

Goal: everything needed to finish the project without Claude Code after 2026-10-08.
Rules followed: no GPU, no ssh to gpuws, no writes to $DSG_RESULTS / dsg_worktrees / cluster worktree,
no large downloads, no hazardous text.

| # | item | status |
|---|---|---|
| 1 | NO_CLAUDE_RUNBOOK.md + doctor / wave_check / reboot / server.sh | done (commands of items 2-6 are referenced; verified at item 7) |
| 2 | `python -m dsgx.analysis.final_report` | in progress |
| 3 | `python -m dsgx.combine` (combination wave) | next |
| 4 | server job scripts + SERVER_JOBS_MANIFEST.md | next |
| 5 | qualitative track (Q1–Q8) tools | next |
| 6 | `python -m dsgx.analysis.paper_assets` | next |
| 7 | CLAUDE.md, merge notes, final PREP_PROGRESS | next |
