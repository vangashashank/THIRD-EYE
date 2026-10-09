# Qwen3-4B Smoke Test: Job 25765

**Completed tiny-data pipeline smoke test. Forecasting accuracy is not validated.**

## Run and Environment

- Date: 2026-10-06. Slurm reports 07:02:39-07:05:13 UTC
  (12:32:39-12:35:13 IST); elapsed time **154 seconds**, state `COMPLETED`, exit `0:0`.
- Model: `Qwen/Qwen3-4B`; logged total parameters 4,025,417,216,
  with 2,949,120 trainable LoRA parameters (logged 0.0733%).
- Allocation: `gpu_student`, one A100 MIG `3g.20gb`, four CPUs, 32 GiB host memory.
  CUDA reported `NVIDIA A100-SXM4-40GB MIG 3g.20gb` and 19.5 GiB device capacity.
  This capacity is not a peak GPU-memory measurement; peak GPU memory was not recorded.
- [Run-captured dependency lock](requirements.server.lock.txt): PyTorch `2.6.0+cu124`,
  Transformers `4.57.6`, PEFT `0.17.1`, Accelerate `1.10.1`.
  Python `3.10.12` was verified during the October 9 inspection; its version at run time
  was not logged. No driver version or exact model revision was recorded.
- Parent config: one epoch, batch size 1, learning rate `0.0001`, max length 128;
  LoRA rank 8, alpha 16, dropout 0.05, target modules `q_proj` and `v_proj`.
  Saved adapter metadata confirms these LoRA settings. Candidate seeds were 11, 22, 33.
  Every adapter-training fit logged one epoch and two steps. Separate candidate,
  descendant, and forecaster optimizer settings were not recorded in the run artifacts.

## Completed Stages

1. Trained a fresh Qwen3-4B LoRA parent; logged epoch-average training loss **1.8121**.
2. Registered its parent checkpoint at state 0.
3. Trained **K=3** candidate updates.
4. Evaluated t+1 losses and trained/evaluated **H=2** simulated descendants.
5. Saved three scalar-label rows and three target/OOD/retention label rows for one state.
6. Passed Direct forecaster architecture, forward/backward, no-history, and training checks.

The run directory contains the parent, candidate, and descendant adapters; inspection
verified 11 adapter-config/weight pairs, including copied parent/candidate adapters.
The log reports run storage as `292M`. No model artifacts or datasets are included here.

## Recorded Results

| Candidate | Seed | t+1 validation loss | t+2 validation loss | t+2 rank |
| --- | --- | --- | --- | --- |
| 0 | 11 | 1.775740 | 1.579895 | 3 |
| 1 | 22 | 1.776056 | 1.566801 | 2 |
| 2 | 33 | 1.763607 | 1.566566 | 1 |

The table rounds losses to six decimals. [JSON](metrics.json) and
[CSV](candidate_metrics.csv) retain recorded values. Multi-consequence labels are
parent minus t+2 full-text token-mean losses, ordered target, OOD, retention.
Each recorded evaluation set contains two arithmetic examples; these are smoke-test proxies.

The forecaster logged initial training MSE **0.10191228**, final training MSE
**0.00005565**, and a constant-output MSE floor of **0.00005566**.
The final fit approximately matches this baseline and does not demonstrate useful forecasting.
The history encoder was unused; there was one state and no held-out evaluation.
**No forecaster weights were saved**, and no separate original metrics file was found.
The metrics files in this evidence folder were extracted from the actual log and saved labels.

## Provenance and Redactions

The run's exact code/Git revision was not recorded, and the server project has no `.git`
metadata. A subsequent synchronization record dates the sync of `6700ed5` to
**2026-10-09 15:34:55 UTC**, after this run. **Do not attribute this run to `6700ed5`.**

Sources: Slurm accounting queried with `TZ=UTC`, `logs/qwen4b_smoke_25765.log`, and
the config, dependency lock, and labels under `runs/qwen3_4b_smoke_25765/`.
[The actual log](qwen4b_smoke_25765.log) marks the redacted private server-project path
(two occurrences) and internal compute hostname (one occurrence); measurements are unchanged.
`metrics.json` records SHA-256 hashes of the original source files and the redacted log.
No training was rerun to prepare this evidence.
