# Sprint 1 Progress

Status: technical target COMPLETE. Final GitHub publication verification is recorded in the handoff.
Baseline `6700ed5`; new run source `833da0c95e421ece4632174a71ed10e6d8caf129`.

## Checklist

| Requirement | Status | Evidence / next action |
| --- | --- | --- |
| LoRA harness and Qwen3-4B pilot | VERIFIED smoke test | Job 25765 completed 2026-10-06, exit 0:0, 154 seconds; original log and saved labels inspected. |
| NF4/bf16 QLoRA | VERIFIED | Job 26946 completed, NF4/double quantization/bf16 compute. |
| Adapter save/reload, rollback, promotion | VERIFIED | All three probe-logit max absolute differences 0.0; adapter digests match, parent unchanged; writes reject overwrites. |
| Same-parent K=3, H=2, finite complete labels | VERIFIED | Job 26946: all initial adapter digests match, proposals before training, actual outcomes, isolated workspace and finite consistent labels. |
| Pilot settings and measured memory/runtime | VERIFIED | Batch 1, lr 1e-4, rank 8; 315 s Slurm / 253.655 s Python; peak allocated 3.950179 GiB, reserved 5.308594 GiB. |
| Forecaster checkpoint and schema | VERIFIED | New fit saved/reloaded on server; prediction difference 0.0. Constant-output baseline, no held-out accuracy. Old run saved no forecaster weights. |
| Documentation, evidence, GitHub push | EVIDENCE COMPLETE / FINAL PUBLICATION | Both reports, actual redacted logs, recorded metrics, dependency locks, README and ledger; compare final HEAD with origin/main after push. |

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
- GPU pilot job `26946` submitted using immutable source commit
  `833da0c95e421ece4632174a71ed10e6d8caf129`; inspect queue and log before any retry.
  Server source snapshot: `.code-snapshots/833da0c95e421ece4632174a71ed10e6d8caf129/`.
  Result root: `runs/sprint1_qlora_26946/`; log: `logs/sprint1_pilot_26946.log`.
  COMPLETED `2026-10-09T16:23:43Z` to `16:28:58Z`, exit `0:0`, 315 seconds;
  one A100 MIG `2g.10gb`, 4 CPUs, 32 GiB host memory.
  The pending job's GRES was resized with `scontrol update JobId=26946 Gres=gpu:a100_2g.10gb:1`
  after the scheduler estimated a multi-day wait for `3g.20gb`. No duplicate job was submitted.
  Home usage before/after: 43,417,399,296 / 43,994,603,520 bytes. Existing `.venv` was not modified.
- Ten offline unit tests (including real PEFT synthetic CPU LoRA), trajectory isolation, Python AST syntax
  parsing (44 files), Bash syntax, diff whitespace and credential-pattern checks passed.
- Old job 25765 evidence is complete; no new measurements are attributed to that run.
- No Sprint 1 jobs remain running or queued. No corrective retries were needed.

## Resume

Inspect this file, Git status, Slurm queue and any recorded job logs before submitting a job.
Use immutable source snapshots for jobs and retain code/config/environment manifests.
All technical requirements now have evidence in `sprint1_report.md` and the two experiment folders.
Verify publication with `git fetch origin` and `git rev-parse HEAD origin/main` before relying on remote state.
Next: independent multi-state Sprint 2 evaluation protocol; no full study has been started.
