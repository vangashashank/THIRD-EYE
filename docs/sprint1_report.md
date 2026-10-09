# Sprint 1 Completion Report

**Technical target COMPLETE: verified pipeline infrastructure, not research accuracy.**

## Evidence Checklist

| Requirement | Verified evidence |
| --- | --- |
| HF/PEFT LoRA harness | Original Qwen3-4B job 25765 completed; new offline real-PEFT synthetic CPU training/save/reload test passes. LoRA remains the default. |
| Configurable NF4/bf16 QLoRA | Job 26946 completed with NF4/double quantization/bf16 compute, cached pinned model, and no ordinary quantized-model `.to()` calls. |
| Save/reload, rollback, promotion | Job 26946: adapter digests match, parent file unchanged, and all three probe-logit comparisons have max absolute difference 0.0. |
| Qwen3-4B bounded pilot/settings | Batch 1, lr 1e-4, rank 8, alpha 16, dropout .05, max length 128; suitable for this smoke pipeline, not optimized. |
| Same-parent K=3 / H=2 | Three initialized parent-state digests match; pre-update proposals and actual outcomes recorded; three finite complete label rows with identical evaluation settings/hashes. |
| Isolation/history/no leakage | Fresh trajectory, duplicate-workspace protection, applied t+1-only history, and forecaster inputs restricted to pre-update parent metrics/proposals. |
| Forecaster checkpoint/schema | New 200-step, three-row fit in job 26946; saved/reloaded on server with prediction max absolute difference 0.0. Synthetic history round-trip separately unit-tested. |
| Reproducible small evidence | Actual reviewed/redacted logs, JSON/CSV metrics, model/config/source hashes, dependency locks, README and experiment ledger. |

## Jobs and Measurements

| Job | Date (UTC) | Status / Exit | Elapsed | Allocation / Result |
| --- | --- | --- | --- | --- |
| 25765 | 2026-10-06 | COMPLETED / 0:0 | 154 s | Qwen3-4B LoRA, one A100 MIG 3g.20gb; original smoke test reused. |
| 26945 | 2026-10-09 | COMPLETED / 0:0 | 3 s | CPU preflight; home usage 43,415,261,184 bytes, cached model verified. |
| 26946 | 2026-10-09 | COMPLETED / 0:0 | 315 s | NF4/bf16, one A100 MIG 2g.10gb; Python pilot 253.655 s; peak allocated 3.950179 GiB, reserved 5.308594 GiB. |

No corrective retries were needed. No Sprint 1 jobs remain running or queued.
Unrelated university jobs were not changed. No computation/training workload ran on the login node.
Home usage after the GPU job: 43,994,603,520 bytes under the declared 50 GB quota.

- [Original LoRA report and evidence](experiments/qwen3_4b_smoke_25765/report.md).
- [New QLoRA report and evidence](experiments/sprint1_qlora_26946/report.md).
- [Experiment ledger](../experiments/experiment_ledger.csv).
- [Resumable progress record](sprint1_progress.md).

## Changes and Checks

Added configurable quantized loading, quantization-aware adapter preparation/placement,
non-overwriting checkpoint writes, same-parent adapter digests, pre-update proposal and actual
training metadata, immutable Slurm pilot/preflight scripts, and schema-aware forecaster persistence.
The optional bitsandbytes dependency is isolated from the existing server environment.
Ignore rules now cover runtime snapshots/caches/runs and leave source scripts trackable.

Checks: 10 offline unit tests, including a real PEFT LoRA round-trip on a tiny synthetic CPU
model; existing trajectory-isolation script; AST syntax for all 44 project Python files;
Bash syntax for both new Slurm scripts; Git diff whitespace checks; credential-pattern scan
and manual full-log review; recorded-label arithmetic; all 56 run source hashes against the
run's Git commit. Local tests used Python 3.9; server execution used Python 3.10.12.
Non-failing local LibreSSL and PEFT missing-repository-config warnings were retained.

## Version Provenance

The baseline Mac/GitHub commit was `6700ed5b0912aed67c1140b989120dddbf11530a`.
Job 25765 ran on October 6 with an unrecorded Git revision; the server synchronization
of `6700ed5` was on October 9, after that run. It is not claimed as the old run's source.

New job 26946 used immutable source commit `833da0c95e421ece4632174a71ed10e6d8caf129`.
The later evidence commit on `main` adds the actual results, a real-PEFT CPU regression test,
and changes the Slurm default to the verified `2g.10gb` allocation;
it must not be described as the commit used by job 26946. Git history records that publication
commit. The server has no `.git`; its immutable snapshot is the authoritative run source.
Root-code synchronization is tracked separately and preserves server-only Slurm files,
the server dependency lock, environments and existing outputs.

## Limits and Sprint 2

Both runs are tiny-data pipeline smoke tests, **not validated forecasting accuracy**.
The new forecaster training MSE `1.643511677684728e-05` approximately equals the constant-output
baseline `1.6435098586953245e-05`; the feature vectors and predictions are identical across
candidates. There is one state, no held-out evaluation, and the history encoder was unused
during this fit. No meaningful OOD/retention or ranking claim follows from these numbers.

Job 25765 saved **no forecaster weights**. Job 26946 saved a **new** checkpoint that remains
server-only, as do all adapters and model artifacts. No weights, environments, raw datasets,
credentials or large outputs are included in the commits.

Next Sprint 2 task: specify a small multi-state data/evaluation protocol with independent
trajectory-level train/validation/test splits and constant-output/ranking baselines, then
generate a bounded initial label corpus. Do not begin the full multi-model study until
that protocol and resource budget are reviewed.
