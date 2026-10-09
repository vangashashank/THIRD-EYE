# THIRD-EYE

A research prototype for LoRA candidate updates, short-horizon consequence
generation, and a Direct forecaster.

## Sprint 1 Complete

The HF/PEFT harness now supports default LoRA and configurable NF4/bf16 QLoRA.
Job **26946** completed on **2026-10-09** in **315 seconds** using one A100 MIG
`2g.10gb`: same-parent K=3 updates, H=2 labels, numerical save/reload/rollback/promotion,
applied-only history, and a new schema-aware forecaster checkpoint round-trip passed.
Peak PyTorch allocated GPU memory was **3.950179 GiB** (reserved **5.308594 GiB**).
Pilot settings: batch 1, learning rate `1e-4`, rank 8; suitable for this check, not optimal.

This remains a **tiny-data pipeline smoke test, not validated forecasting accuracy**.
The new fit's MSE (`1.643511677684728e-05`) matches the constant-output baseline
(`1.6435098586953245e-05`); no held-out evaluation was performed. The new forecaster
checkpoint and all adapters remain server-only. The original job below saved no forecaster weights.

- [Sprint 1 report and completed checklist](docs/sprint1_report.md)
- [New pilot report, actual log, metrics and provenance](docs/experiments/sprint1_qlora_26946/report.md)
- [Progress / resume record](docs/sprint1_progress.md)

Offline checks (no model downloads):

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m scripts.test_trajectory_isolation
```

The new GPU pilot is scheduler-only; use the immutable source and submission commands
in its report. Optional QLoRA dependency: `requirements.qlora.txt`.

## Original LoRA Smoke Test

Qwen3-4B job **25765** completed on **2026-10-06** in **154 seconds** on one
A100 MIG `3g.20gb` allocation. The run completed fresh parent training, three
candidate updates, H=2 consequences, label generation, and forecaster checks.

This was a **tiny-data pipeline smoke test, not validated forecasting accuracy**.
The forecaster's final training MSE (`0.00005565`) was approximately equal to
the constant-output baseline (`0.00005566`). Its history encoder was unused,
and **no forecaster weights were saved**.

The run's Git revision was not recorded. Commit `6700ed5` was synchronized to
the server on October 9, after the October 6 run; it is not claimed as the
revision that produced these results.

- [Run report](docs/experiments/qwen3_4b_smoke_25765/report.md)
- [Actual job log, with marked redactions](docs/experiments/qwen3_4b_smoke_25765/qwen4b_smoke_25765.log)
- [Recorded metrics](docs/experiments/qwen3_4b_smoke_25765/metrics.json)
- [Candidate metrics](docs/experiments/qwen3_4b_smoke_25765/candidate_metrics.csv)
- [Run-captured dependency lock](docs/experiments/qwen3_4b_smoke_25765/requirements.server.lock.txt)
- [Experiment ledger](experiments/experiment_ledger.csv)
