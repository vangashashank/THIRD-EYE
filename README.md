# THIRD-EYE

A research prototype for LoRA candidate updates, short-horizon consequence
generation, and a Direct forecaster.

## Verified Smoke Test

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
