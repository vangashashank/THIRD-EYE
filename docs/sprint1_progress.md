# Sprint 1 Progress

Status: IN PROGRESS. Last verified baseline: local `main` at `6700ed5`.

## Checklist

| Requirement | Status | Evidence / next action |
| --- | --- | --- |
| LoRA harness and Qwen3-4B pilot | VERIFIED smoke test | Job 25765 completed 2026-10-06, exit 0:0, 154 seconds; original log and saved labels inspected. |
| NF4/bf16 QLoRA | IMPLEMENTED / GPU PENDING | Configurable NF4/double-quant/bf16 loader and k-bit preparation; offline loader tests pass. |
| Adapter save/reload, rollback, promotion | PARTIAL | Saved adapters exist; add numerical prediction comparisons and reject accidental checkpoint overwrites. |
| Same-parent K=3, H=2, finite complete labels | VERIFIED old smoke test / PENDING new metadata | Three candidate and descendant labels verified; new run must exercise proposal recording and isolation. |
| Pilot settings and measured memory/runtime | PARTIAL | Old run: batch 1, learning rate 1e-4, rank 8; peak GPU memory was not measured. |
| Forecaster checkpoint and schema | IMPLEMENTED / GPU LABELS PENDING | Schema-aware save/reload and synthetic prediction round-trip tested. Old run saved no forecaster weights. |
| Documentation, evidence, GitHub push | IN PROGRESS | Preserve interrupted README/report edits; add original redacted log, actual metrics and environment lock. |

## Execution State

- SSH control socket: `~/.ssh/third-eye-%C`; use existing authentication.
- Server project: `Third-Eye-4B-Smoke`; original results in `runs/qwen3_4b_smoke_25765/`.
- The old run preceded the 2026-10-09 synchronization of `6700ed5`; its exact Git revision is unknown.
- Existing unrelated benchmark jobs must remain untouched.
- New computation must use `cpu_student` or `gpu_student`, never the login node.
- Storage limit: 50 GB; query usage on a compute node before installing optional dependencies or saving new adapters.
- At most one Sprint 1 GPU job at a time and two corrective retries per failed stage.
- CPU preflight job `26945`: COMPLETED, exit `0:0`, 3 seconds, 2026-10-09.
  Compute-node home usage: 43,415,261,184 bytes against the declared 50 GB limit.
  Cached Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
  The existing environment has no bitsandbytes; use an isolated optional dependency overlay.
- No new GPU job submitted yet. Nine offline unit tests, trajectory isolation, Python AST syntax
  parsing (44 files), Bash syntax, diff whitespace and credential-pattern checks passed.
- Old job 25765 evidence is complete; no new measurements are attributed to that run.

## Resume

Inspect this file, Git status, Slurm queue and any recorded job logs before submitting a job.
Use immutable source snapshots for jobs and retain code/config/environment manifests.
Do not declare Sprint 1 complete until every pending requirement has recorded execution evidence.
