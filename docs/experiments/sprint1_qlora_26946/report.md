# Sprint 1 Qwen3-4B NF4/bf16 Pilot: Job 26946

**Completed tiny-data pipeline verification, not validated forecasting accuracy.**

## Run

- Slurm: `gpu_student`, `COMPLETED`, exit `0:0`, 2026-10-09
  **16:23:43-16:28:58 UTC** (21:53:43-21:58:58 IST), **315 seconds** allocated runtime.
  The Python pilot measured **253.65520360902883 seconds** from its entry point,
  excluding dependency installation, process imports and surrounding shell commands.
- One `NVIDIA A100-SXM4-40GB MIG 2g.10gb`, four CPUs, 32 GiB requested host memory.
  CUDA-reported device capacity: **10,334,765,056 bytes / 9.625 GiB**.
  PyTorch peak allocated: **4,241,472,000 bytes / 3.950179 GiB**;
  peak reserved: **5,700,059,136 bytes / 5.308594 GiB** across the entire Python pilot.
  These are allocator measurements, not an NVML measurement of all device/driver overhead.
- Python `3.10.12`, PyTorch `2.6.0+cu124`, Transformers `4.57.6`, PEFT `0.17.1`,
  Accelerate `1.10.1`, bitsandbytes `0.46.1`, CUDA runtime `12.4`, driver `550.144.03`.
  [Actual runtime dependency lock](requirements.runtime.lock.txt).
- `Qwen/Qwen3-4B`, cached revision `1cfa9a7208912126459214e8b04321603b3df60c`;
  NF4, double quantization and bf16 compute. Offline model loading was enforced.
  Logged 4,025,417,216 total and 2,949,120 trainable LoRA parameters.
- Finalized **pilot** settings: batch **1**, learning rate **1e-4**, rank **8**;
  alpha 16, dropout 0.05, targets `q_proj`/`v_proj`, max length 128, one epoch.
  Parent and each candidate/descendant fit used two examples and two AdamW steps.
  Seeds: parent 42, candidates 11/22/33, descendants 1000/1001/1002.

These settings are suitable for this bounded pipeline check: finite updates, complete labels,
successful checkpoint comparisons, and substantial headroom in the 10 GB MIG slice.
They were inherited from the working LoRA pilot; no hyperparameter search was performed.
They are **not optimal settings** or a recommendation for research-scale training.

## Verified Stages

1. Trained and saved an NF4/bf16 parent adapter; mean training loss `2.5469970703125`.
2. Reloaded the adapter into a fresh quantized base. Adapter-state SHA-256 matched;
   full next-token probe logits had maximum absolute difference **0.0**.
3. Recorded three proposals before their training, then actual step losses, durations,
   and initialized adapter digests afterward. All three started from the same parent.
4. Generated three H=2 descendants and complete target/OOD/retention labels. All losses
   and labels are finite. Every evaluation used batch 1, max length 128 and the same
   two-example sets, with saved dataset hashes. Labels are parent minus t+2 token-mean loss.
5. Verified the parent file was unchanged. Rollback probe-logit difference: **0.0**.
   Promoted candidate **1**, selected by **t+1 target loss**, not future t+2 results;
   promoted versus candidate probe-logit difference: **0.0**.
6. Recorded only the applied state-0-to-1 t+1 transition in history. State-0 features remain
   empty after recording it; state-1 history contains actual loss reductions
   `[0.2251589298248291, 0.10474181175231934, 0.19967961311340332]`, not simulated t+2 labels.
7. Performed a **new** small Direct forecaster fit (seed 42, Adam, lr 0.01, 200 steps,
   three rows). Saved a schema-aware checkpoint on the server and reloaded it;
   prediction maximum absolute difference **0.0**. Inputs used only the pre-update
   parent measurements and proposed update settings. No future outcomes were input features.

| Candidate | Seed | t+1 target loss | t+2 target loss | Promoted |
| --- | --- | --- | --- | --- |
| 0 | 11 | 2.326694 | 2.074093 | No |
| 1 | 22 | 2.325388 | 2.072422 | Yes |
| 2 | 33 | 2.326373 | 2.061019 | No |

The table is rounded. [Metrics JSON](metrics.json) and [CSV](candidate_metrics.csv)
retain recorded numeric precision, including OOD/retention proxies and proposal/outcome metadata.

## Forecaster Limitations

Initial training MSE: `0.17838478088378906`.
Final training MSE: `1.643511677684728e-05`.
Constant-output baseline MSE: `1.6435098586953245e-05`.
All three predictions are identical: `[0.4813684821128845, 0.2294064611196518, 0.4349643290042877]`.
The three proposal feature vectors are identical; seeds are not forecaster inputs.
The fit therefore matches a constant-output baseline and does not establish candidate ranking.

This is one state with tiny arithmetic training data and proxy OOD/retention evaluations;
there is no held-out evaluation, validated forecasting accuracy, OOD generalization or retention
claim. The forecaster history encoder was unused during this fit. Synthetic unit fixtures are
separate from these run measurements. Original job 25765 saved **no forecaster weights**;
this job's newly fitted checkpoint is distinct and remains **server-only**.

## Provenance and Artifacts

Run source: **`833da0c95e421ece4632174a71ed10e6d8caf129`**, deployed as an immutable
code-only snapshot under `.code-snapshots/<commit>/`. All 56 captured source-file hashes
were compared with that Git commit; [provenance](provenance.json) records them, the exact
model revision and environment. [Run config](run_config.json) is the actual captured config.
The generic config's `output.adapter_path` is not used by this isolated pilot.

Submission, from the server project directory:

```bash
commit=833da0c95e421ece4632174a71ed10e6d8caf129
sbatch --chdir="$PWD" ".code-snapshots/$commit/scripts/slurm/sprint1_pilot.sbatch" "$PWD" "$commit"
scontrol update JobId=26946 Gres=gpu:a100_2g.10gb:1
```

The pending job was resized from `3g.20gb` after a multi-day scheduler estimate;
Slurm then scheduled the 10 GB slice. No duplicate job, corrective retry, scheduler bypass,
unrelated job cancellation, or login-node training was performed.
The subsequently published Slurm script defaults to the verified `2g.10gb` slice;
the original immutable `833da0c` script is retained for this run's provenance.

Server results: `runs/sprint1_qlora_26946/sprint1_qlora_26946/trajectory_000/`.
Adapter checkpoints are under its `checkpoints/`, simulated descendants under
`trajectory_cache/`, and the new forecaster under `forecasting/forecaster.pth`.
Model/adapter/forecaster weights, datasets and the virtual environment are excluded from Git.
The existing `.venv` and `requirements.server.lock.txt` were preserved; bitsandbytes was
installed in `.runtime-deps/bitsandbytes-0.46.1/` with `--no-deps`.

CPU preflight job **26945** completed in 3 seconds. Compute-node home usage was
43,415,261,184 bytes then, 43,417,399,296 bytes before the GPU job, and
43,994,603,520 bytes afterward, below the declared 50,000,000,000-byte quota.

[The actual job log](sprint1_pilot_26946.log) preserves all measurements and marks 28
private project-path redactions, two home-path redactions, and one compute-hostname redaction.
Metadata paths are marked `[REDACTED_SERVER_PROJECT]`. Original source hashes and the
redacted log hash are in `metrics.json`; no credentials were found during review.
The PEFT offline save warning about missing repository config remains in the log;
no vocabulary changes were made, and adapter-state and numerical reload checks passed.

The October 6 job 25765 preceded the October 9 synchronization of `6700ed5` and is
**not** attributed to that commit. This new job is explicitly attributed to `833da0c`.
