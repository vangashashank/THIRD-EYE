# Sprint 2 Handoff Integration

## Code Integration Update (2026-10-11)

Fetched origin before integration. The reviewed local commit
`2c20d3666b90ab662c5bf178b680835e193caf58` and Mahitha's latest commits
`38ce3d0fe26dcdaf96b2c5e10240549129ca199b` and
`0948ba4fa1a363b00942a5051b2813eb41b9825e` are retained in merge history.
The only textual conflict was [run_forecasting_eval.py](../scripts/run_forecasting_eval.py).
Both `--semantic-hash` and `--expected-semantic-hash` are retained; conflicting
or empty explicit values fail before records are read. Either spelling enforces
the same exact expected hash. A supplied handoff spec remains authoritative;
no hash is translated, truncated or substituted to make incompatible records pass.
Strict manifest/provenance validation and opt-in D13 evaluation remain intact.

**Code integration is not feature-definition agreement or real-data readiness.**
Mahitha's new helpers/specification are preserved verbatim. The isolated
collector, its configuration gates, raw-module handoff and Sprint 1 are unchanged.
The new upstream commit/spec calls its definitions confirmed/agreed; this merge
does not infer joint approval or populate missing collector agreement references.
The older comparisons below and the quick decision sheet refer to `dcd403a`,
not to the newly fetched implementation. The remaining non-equivalences are:

| Area | New upstream implementation | Preserved isolated collector / handoff |
| --- | --- | --- |
| D08/D09 NLL | Equal-answer mean of token means, also specified for gradients. | Explicit equal-answer or token-weighted choice, still unbound; no default accepted here. |
| D07 duplicates | Exact rational Jaccard >= .90, no tolerance. | Floating-point comparison with explicitly supplied threshold/tolerance; definitions still unbound. |
| D05 layout | Optional relative-depth gradient-energy bins remain in the upstream feature specification. | Accepted raw ordered module A/B joint gradient norms; no conversion to bins. These layouts are separate, not interchangeable. |
| D09 diagnostics | Near-zero cosine flags at norm < 1e-12 and integer zero counts. | Explicit epsilon with <= comparison, exact-zero flags and zero-element fractions; no invented conversion or replacement measurement. |
| Probe train slope / KL | Optimizer losses at 1..15; per-answer equal-weight KL, float64 log-softmax, per-position clipping. | Fixed-batch train loss at 0/5/10/15; pooled answer-position KL, float32 log-softmax, final-mean clipping. No formula chosen by this merge. |
| D15 specification | `SemanticConfig` schema `0.3-d05-d12-agreed`, formula/base-revision specification and 16-character fingerprint. | Schema `0.2-sprint2`, raw layout plus complete sealed binding specification and 64-character digest. One is not a substitute for the other. |

These implementations can coexist without selecting a formula for collection.
Reconciliation before real collection must be explicit, versioned and tested;
missing sources/scorers, independent roots/8-2-2 assignment, calibration,
runtime module/template/MIG evidence and execution authorization remain blockers.
The collector still stops offline before loading a model with its unbound config.
No schema is frozen and no real records, calibration, labels or Gate 1 are claimed.

Verification: **199 offline regression tests passed**, including Sprint 1,
both forecasting implementations and three new synthetic CLI regression tests.
The focused handoff suite passed 28 tests. Fixture success is structural only,
not forecasting validation. No GPU jobs, downloads or ignored artifacts were added.

## Original Integration Review (Historical)

Reviewed 2026-10-11 against fetched `origin/mahitha/forecasting` at
`dcd403af9e0b8dd2e79ad1b5bcae987b8785eddb`, initially in the isolated
`mahitha-handoff-review` worktree. Its two commits descend from local
`da142ec`; the diff adds 15 files and changes no existing file. After inspection
and 59 passing upstream offline tests, `main` was fast-forwarded, without a
merge commit, reset, force-push or stash. All 12 pre-existing dirty Sprint 2
files were verified byte-identical. New compatibility edits remain uncommitted;
nothing was pushed. Sprint 1 runtime, evidence, adapters and outputs are unchanged.

**The combined synthetic handoff passes structural/provenance validation.
Real-data readiness, Gate 1 and forecasting performance are NOT established.**
No university job, calibration execution, real label collection or model download
was launched. Tests include tiny synthetic CPU forecaster fits and the existing
tiny CPU PEFT regression test, not university/Qwen training.

## First Real Batch Readiness (2026-10-11)

**Status: blocked on real sources/root provenance, formula agreement and SSH
authentication. No real batch, calibration, throughput measurement or Gate 1
was executed.** This update supersedes readiness claims in the earlier review;
the 185-test full-suite result below is historical, not rerun in this focused step.

### Bindings Verified Now

The recorded [26946 experiment](experiments/sprint1_qlora_26946/report.md) supports
`Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`,
NF4/double-quant/bf16, LoRA r8/alpha16/dropout.05/q_proj+v_proj and inherited
optimizer settings. Both current configs already pin that model. Accepted
T5/K3/H2, source counts, shared sibling continuation, selected-H1 history,
update/probe settings and provisional ceilings are unchanged. The handoff
destination is now `runs/sprint2_qwen3_4b_handoff_spec.json`; **this is a planned
path, not a sealed spec or measured evidence**. Paths resolve relative to the
config directory. Use actual absolute data/root paths when deploying a snapshot.

Ignored-aware repository inventory found only smoke/synthetic data and local
Qwen3-0.6B adapters. `data/` contains `.gitkeep`; the standard local HF cache has
0.6B, not4B. `configs/forecasting_smoke_sets.json` explicitly says smoke-only and
has two examples/role. It cannot satisfy real32/32/256 populations or independent
roots. Old token-loss labels are not accuracy deltas. **No real task was bound**;
there is no defensible dataset/independence identity to fill from this evidence.
The numerical 8/2/2 plan is preserved; its actual assignment/provenance file is absent.

### Exact Blockers and Owners

| Location / fields | Required evidence or action | Owner |
| --- | --- | --- |
| `configs/sprint2_collection.yaml`: `task`, `corpus_version`, `task_rules` | Actual versioned task/corpus and native deterministic answer/reference rule. Numeric/choice adapters exist, but neither has an actual task binding. Verify suitable task scoring, not generic full-text loss. | Shashank supplies cached sources; joint scorer review with Mahitha. |
| `datasets.target/ood/retention/confidence/anchor_probe/val_probe/retention_gradient/final`; `adaptation_manifests` | Actual IDs, references, source-family lineage and hashes; target32/OOD32/retention256, pinned32 subsets; no adaptation/final leakage. Confidence population/size belongs to D06. All per-group source pools must support sealed32 H1 draws and disjoint32 H2 draw. | Shashank/data owner; Mahitha confirms confidence population. |
| `lineage_plan`, `roots`, identities in that JSON | Complete12 preassigned groups8/2/2, independent adapted roots/families/prefixes, real root files and checksums. The collector currently requires all planned root artifacts and identities even for the first group's export. If absent, separately authorized root preparation is required; seeds/copies of one smoke parent are not independent groups. | Shashank supplies/audits root provenance; collector verifies consistency, not independence truth. |
| `module_manifest`, `chat_template_sha256`; `retention.mig_profile/device_total_memory_bytes` | Actual runtime ordered module/A/B names, shapes/dtypes, template hash and allocated MIG properties. New [metadata audit](../scripts/inspect_sprint2_model_bindings.py) loads only cached model/root; no optimizer/generation/labels and no feature-formula gate. Its root tensor digest is distinct from file-byte checksums. It cannot certify independent lineage. | Shashank grants access/allocation; tooling extracts real metadata. |
| `definition_agreements.D06/D07/D08/D09/D12_measurements`; `definitions.*` | Resolve only the five formula rows in the [quick decision sheet](sprint2_mahitha_quick_decisions.md). NLL reduction, duplicate tolerance, confidence population/parse policy, zero-cosine boundary and probe train trace are explicit, not accepted defaults. Conditional small formula fixes are listed there. | Shashank and Mahitha jointly. |
| `continuation.prompt_pools/recipe` | Copy actual adaptation IDs from the sealed manifests; recipe equals pinned update, greedy/nonthinking generation, reviewed correction version and stop-on-empty rule. Counts/seed sharing/selection are already accepted, not new decisions. | Binding tooling after actual data and D08 correction agreement. |
| `configs/sprint2_generation_calibration.yaml`: `training_group_ids`, `tasks[].task_id/training_manifest/answer_rule` | Actual assigned train-group manifests, >=50 unique prompts/task; no held-out rows. | Shashank/data owner. |
| Collection `calibration_reports`, `generation_caps`, `caps_frozen_before_collection`, `caps_review_reference` | Real uncensored training-only calibration and reviewed fixed task caps. At least30 correct is the existing evidence-screening proposal, not population-p99 reliability. Report truncation on ALL prompts. | Authorized calibration, then Shashank/Mahitha review. |
| Server snapshot/cache/quota/jobs | Current server code and dataset/cache/MIG/headroom inspection plus code-only comparison; only deploy reviewed sources into a new snapshot, preserving server-only Slurm and outputs. | Shashank supplies working SSH access; tooling then inspects. |

### Server Access: Exact Failure and Action

Retry using existing SSH defaults, `BatchMode=yes`, eight-second timeout and
`UpdateHostKeys=no` reached `10.59.121.172`, then exit255:
`se23ucse205@10.59.121.172: Permission denied (publickey,password).`
There is no local `~/.ssh/config` or default private-key file in `~/.ssh`.
No credentials were read/exposed and authentication was not modified. Current
MIG availability, model/data caches, quota, jobs25765/26946 and server code remain
**unverified**; historical A1002g.10gb evidence is not current availability.

Smallest action: in Terminal, run this read-only command and enter the university
password there (never paste it into chat), or make an already-authorized key
available through the existing SSH agent. It sends the stdlib inspection script
via stdin, writes no server file and does not submit/interrupt jobs:

```bash
cd /Users/vangashashankgoud/Documents/ChatGPT/Third-Eye
set -C  # Refuse to overwrite a prior local report.
ssh -o ConnectTimeout=8 -o UpdateHostKeys=no se23ucse205@10.59.121.172 \
  'python3 - --project /dgxa_home/se23ucse205/Third-Eye-4B-Smoke' \
  < scripts/inspect_sprint2_server.py > runs/sprint2_server_readiness_20261011.json
.venv/bin/python -B -m scripts.inspect_sprint2_server \
  --project "$PWD" --compare runs/sprint2_server_readiness_20261011.json
```

Use a fresh report filename if it already exists. On SSH failure the report is
not valid evidence. The inspection records scheduler visibility/own jobs, quota,
filesystem/home usage, known pinned model and data caches, source hashes and data
manifest metadata without prompt contents/logs/credentials. Login-node visibility
does not prove an available MIG allocation; quota failure is not free headroom.

### Prepared Commands: Authorization Still Required

After access, compare/deploy this **uncommitted** code into a unique reviewed
server snapshot without replacing server files. The commands below must run from
that Sprint2 source directory, not the old Sprint1 directory (Python's current
directory otherwise shadows the new modules). Use the existing server environment
and cache; no pip install/download command is prescribed. Review actual package
versions and cached file availability first. `AUDITED_ROOT` and `TRAIN_TRAJECTORY`
must be actual supplied/audited paths/IDs, not invented placeholders. No allocation
or Slurm submission is authorized by these commands.

```bash
export PROJECT_ROOT=/dgxa_home/se23ucse205/Third-Eye-4B-Smoke
export HF_HOME="$PROJECT_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PY="$PROJECT_ROOT/.venv/bin/python"
# In the reviewed Sprint2 source directory; an authorized single-MIG allocation is needed for --execute.
: "${AUDITED_ROOT:?Set an existing actual adapter path; this does NOT certify independent lineage}"
"$PY" -B -m scripts.inspect_sprint2_model_bindings --config configs/sprint2_collection.yaml \
  --root-adapter "$AUDITED_ROOT" --out "$PROJECT_ROOT/runs/sprint2_model_bindings_v1.json" --execute
# Bind real train tasks/manifests first. This draws ~50/task once, starts at512, greedy base model.
"$PY" -B -m scripts.calibrate_generation_caps --config configs/sprint2_generation_calibration.yaml \
  --output "$PROJECT_ROOT/runs/sprint2_caps_512_v1" --execute
# Only if censored: same prompts, explicit increased cap, new output;1024 is an example, NOT a frozen cap.
"$PY" -B -m scripts.calibrate_generation_caps --config configs/sprint2_generation_calibration.yaml \
  --recheck-cap 1024 --output "$PROJECT_ROOT/runs/sprint2_caps_1024_recheck_v1" --execute
```

Review measured caps, populate all bindings/decision references and make sure the
planned spec's parent directory exists. Then run offline before any collector:

```bash
: "${TRAIN_TRAJECTORY:?Set one actual preassigned TRAIN trajectory ID}"
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --prepare-spec
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --preflight
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --dry-run --trajectories "$TRAIN_TRAJECTORY"
"$PY" -B -m scripts.measure_sprint2_throughput --config configs/sprint2_collection.yaml \
  --trajectory "$TRAIN_TRAJECTORY" --out "$PROJECT_ROOT/runs/sprint2_throughput_v1" --execute
# Review timings/memory/storage and authorize fit before running the unchanged T5 pilot.
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --trajectories "$TRAIN_TRAJECTORY" \
  --out "$PROJECT_ROOT/runs/sprint2_first_train_t5_v1" --execute
# Only for interrupted collection: same sources/config/root, retained evidence and cumulative budget.
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --trajectories "$TRAIN_TRAJECTORY" \
  --out "$PROJECT_ROOT/runs/sprint2_first_train_t5_v1" --resume --execute
: "${EXPORT:?Set the actual fresh export directory printed by the successful collector}"
"$PY" -B -m scripts.run_forecasting_eval --records "$EXPORT/meta.jsonl" \
  --split-manifest "$EXPORT/manifest.json" --handoff-spec "$EXPORT/spec.json" --validate-only
```

The separate [throughput check](../scripts/measure_sprint2_throughput.py) uses
one sealed32-source candidate draw and fresh32-step AdamW update, plus **four
generation samples per target/OOD/retention role**. These are benchmark samples,
not reduced collector diagnostics. It finally restores/fingerprints the root,
retains attempts and its temporary endpoint, records actual synchronized timings,
tokens/truncation/bytes and GPU peaks. No H2 labels, promotion, history or candidate
records. Its partial cost extrapolation excludes probes/features/other overhead
and explicitly does not establish budget feasibility.

### Expansion and Resource Boundary

After the complete first15 records validate and cost is reviewed, **30 states =
six distinct assigned TRAIN lineage groups x5 decisions, with3 candidates/state
=90 records**, not30 candidate rows or extra decisions in one group. Preserve
the remaining2 train/2 validation/2 test groups and the full12-group plan; batch
manifest includes only groups actually present. Train-only90 rows do not satisfy
full train/validation/test forecaster evaluation.

Prepare the following expansion dry-run with five other actual train trajectories
from distinct planned groups. Resume skips the already-complete first group:

```bash
: "${TRAIN_2:?}" "${TRAIN_3:?}" "${TRAIN_4:?}" "${TRAIN_5:?}" "${TRAIN_6:?}"
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --dry-run \
  --trajectories "$TRAIN_TRAJECTORY" "$TRAIN_2" "$TRAIN_3" "$TRAIN_4" "$TRAIN_5" "$TRAIN_6"
# Only after actual resource review and separate expansion authorization; SAME cumulative ceilings apply.
"$PY" -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --resume --execute \
  --out "$PROJECT_ROOT/runs/sprint2_first_train_t5_v1" \
  --trajectories "$TRAIN_TRAJECTORY" "$TRAIN_2" "$TRAIN_3" "$TRAIN_4" "$TRAIN_5" "$TRAIN_6"
# Set EXPORT to its NEW printed90-record export, then repeat the exact validate-only command above.
```

The collector does **not** reset the two-hour/5GiB ceiling on resume or allow
automatic expansion of it. If six groups cannot fit, stop and request a reviewed
staged/per-group collection and verified combining workflow or revised execution
budget; that larger-budget workflow is not implemented/approved by this step.
Do not delete the first batch or regenerate it to avoid cumulative accounting.

First group: **1,185 optimizer updates,11,200 endpoint-scoring generations**.
Six groups: **7,110 updates,67,200 generations**, plus corrections/confidence/
feature forwards/probes/restore/init. Historical two-step fits give only a coarse
update-only scenario21.7-31.6min/group, not current throughput; even that scenario
for six groups is130-190min before generation/probe overhead. The historical
2,949,120-parameter float32 scenario gives~.330GiB for30 endpoint adapters and
~1.978GiB for180, excluding roots/base cache, serialization, traces/logs, host
snapshots and interrupted attempts. Actual full corpus time/bytes/CPU snapshot
overhead, allocated MIG memory and free quota remain unmeasured. No silent
diagnostic/cap/step reduction, base duplication or feasibility promise.

### Focused Verification and Preservation

Preflight, prepare-spec and dry-run were invoked with the real preparation config;
all exit2 **before model loading**, with exact missing fields. Calibration offline
validation also exits2 on unbound task manifests. No spec/record was manufactured.
Focused readiness/collection/backend/measurement/protocol tests cover actual CPU
fixture updates, full success/failure rollback, hashes/root identity, source
separation/shared draws, H1 history and resumable journal behavior. These are
synthetic fixtures, not extracted research features or real Gate1 evidence.
The new metadata audit and read-only inspection are prepared but unexecuted on
the university server. **40 focused offline tests pass**, including10 new readiness
fixtures; the new10 also pass with `python -O`. RuntimeWarnings are errors;
the existing non-failing LibreSSL warning remains. Local0.6B adapter metadata is
correctly rejected by the4B audit before model loading. Syntax/whitespace checks
pass. Seven protected pre-existing dirty files remain byte-identical, including
schema, validator/tests and calibration config; tracked Sprint1 runtime/evidence
diffs are empty. No real artifact was overwritten or deleted. Changed this step:
the collection config/blocker diagnostics and the existing integration/decision
docs; new throughput module/CLI, model-metadata audit, read-only server inspection,
quick Mahitha sheet and synthetic readiness tests. No staging/commit/push.

## Collection Tooling Update (2026-10-11)

The latest user instruction approves **configuration/implementation for one
assigned training-group T=5 pilot**, not execution. No new jobs, calibration,
Qwen runs, downloads, commits or pushes occurred. Existing dirty work is preserved.
The branch was not reapplied. Earlier code-review findings below describe the
pre-collector baseline; this section records the subsequent implementation.

### Implemented, Not Measured

| Component | Current implementation and boundary |
| --- | --- |
| [Collection preparation](../configs/sprint2_collection.yaml), [CLI](../scripts/collect_sprint2.py) | Offline preflight/dry-run; explicit allocation-only execution; immutable spec/config/source hashes, complete12-group plan, per-state ledgers/events/checksums and fresh exports. Resume verifies artifacts/config/source; interrupted states are retained/retried from last applied H1. Failed-session time counts toward the ceiling; a hard-killed/unclosed session or stale lock requires manual review, not deletion/budget reset. Initialization is included when the session reaches collection; initialization failure still requires manual accounting. Concurrent runs refused. |
| [State draw planner](../src/third_eye/sprint2_protocol.py) | Each candidate draws 32 unique IDs using a separate seed keyed by `(group,state,candidate)`. Shared procedure/count/pool/greedy generation; sibling overlaps are logged. One H2 draw of 32 excludes the **union** of all three H1 draws, keyed by `(group,state)`, with separate source/generation/correction/training streams and per-prompt seeds. All T=5 state plans are checked before outcomes; insufficient pool capacity stops, never redraws. A128-ID pool suffices for worst-case disjoint draws; actual required size is union size +32, not an approved dataset size. |
| [Runtime](../src/third_eye/sprint2_runtime.py) | Finally restores CPU snapshots of all parameters/buffers, including nonpersistent buffers, modes, requires-grad flags, gradients, explicit optimizer/scheduler/scaler/sampler state, Python/NumPy/CPU/CUDA RNG and explicit torch generators; verifies the full fingerprint. No parameter/module creation or guessed sampler cursors. Probes cannot promote/history-advance. Full siblings restore the common parent after H2; only the preselected H1 is reloaded/applied. |
| [Measured reductions](../src/third_eye/sprint2_measurements.py), [cached backend](../src/third_eye/sprint2_backend.py) | Real forward/scorer/tensor producers for schema fields, all three H1/H2 accuracy heads, counts, parse failures, stopping/truncation and source provenance. Causal answer-token alignment; raw ordered module A/B norms, gradient diagnostics and streamed per-example gradients avoid retaining 32 graphs. Confidence uses emitted tokens in their actual generation context, not a reconstructed correct answer. Candidate confidence/NLL uses its verified corrected batch. Frozen-base embeddings disable adapters. Temporary measurements remain separate from full labels. **Supported proposed formulas are not agreement or real measurements.** |
| Corrections | Generate a reconsidered answer at the common parent (H1 candidates), or each child's H1 endpoint (H2). Check the actual emitted final value against the deterministic reference; train only verified corrections to original failures. All original/correction attempts remain in denominators/evidence. No direct reference substitution. The exact reconsideration prompt is a proposed versioned producer choice awaiting D08 review. Empty verified batches stop with generated attempts retained. |
| D13 [opt-in evaluation](../src/third_eye/sprint2_evaluation.py) | `TrainConfig(protocol="sprint2_d13")` and CLI `--protocol sprint2_d13`: nested equal group/trajectory/state/candidate weights, training-only normalization, native targets, validation LHV regret then native three-H2-channel MSE. One-step H1 vectors are explicitly an H2 proxy for the common secondary criterion, not H2 heads. Full-batch weighted CPU fitting preserves legacy paths. Reports training-derived constant, exact expected uniform/constant selection, highest pre-update NLL, prediction MAE/MSE/RMSE, ties, per-group coverage and paired lineage-group bootstrap intervals. PD/Dynamics remain inactive; real fitting also requires complete bound configuration and measured-count provenance. |
| Calibration | Same ~50 outcome-blind training prompts/task, greedy cached base, initial 512; `--recheck-cap` changes only the explicit training-generation cap, not prompt selection, and refuses old output directories. Any censoring/insufficient evidence yields no percentile. >2% all-prompt truncation requests an increased-cap rerun, followed by fixed-cap review. Reports include actual template hash and effective cap. Collection checks actual measurements against summaries and train-group sources; caps must be explicitly reviewed/fixed before collection, never from held-out results. No schema freeze. |

### Latest Approved Pilot Settings

- K=3, one assigned train lineage, decisions S0..S4: **15 candidate records**.
  Uniform state-seeded selected ID is sealed before features and full outcomes;
  NLL/diagnostics/H2 cannot choose the applied child in this pilot.
- H1/H2: **32 optimizer updates each**. Probe: **15**, fixed population losses
  at **0/5/10/15**; per-update optimizer losses are also logged separately.
  OLS/KL/answer masks remain proposed reductions, not approved by a budget choice.
- Reuse supported lr=1e-4 and LoRA rank8/alpha16/dropout.05/q_proj+v_proj.
  Existing AdamW defaults are now explicit: weight decay .01, betas .9/.999,
  eps1e-8. Batch1, accumulation1, deterministic sealed-prompt-order cycling;
  fresh moments for every probe/H1/H2, no scheduler (the supported trainer has none).
  Existing max_length128 is retained as preparation; missing/truncated answer
  spans fail, rather than treating this smoke bound as proven suitable real data.
- Fixed target32/OOD32 diagnostic populations, retention256, anchor-probe32 and
  retention-gradient32; val-probe reuses fixed target32. Overlaps with diagnostic
  subsets must be declared; adaptation/final families cannot overlap diagnostics.
  Confidence population/parse policy remains unbound. Parent/H1/H2 use identical
  scorer populations/caps; endpoint accuracy counts are revalidated on import.
- Provisional **single-MIG / two-hour / 5 GiB** ceiling for this pilot. Guard
  before generation/updates and artifact writes; reserve 120 seconds for cleanup,
  interrupt generation with a remaining-time bound and roll back the parent.
  Interrupted calls are not completed accuracy measurements. CPU snapshot
  overhead, actual MIG/cache/free quota and CUDA behavior must be measured first.
  Bind inspected `mig_profile` and `device_total_memory_bytes`; execution checks
  the allocated device name/profile and memory property match. If the runtime name
  does not expose the profile, stop/review evidence instead of guessing equivalence.
- Retain roots, all pilot H1/H2 adapters, selected parents, manifests/records,
  calibration/source IDs, ledgers/logs/checksums. No base duplication or pruning;
  temporary in-memory probe states are restored/disposed. An individual operation
  can cross a storage limit before the next guard; stop/preserve, never delete.

### Remaining Real Bindings

Actual research task/dataset/scorer identities are absent locally. `data/` has
only its placeholder; the cached local adapter configs inspected are Qwen3-0.6B,
not the pinned 4B research roots. The supported 4B NF4 revision is
`1cfa9a7208912126459214e8b04321603b3df60c`; this establishes a loader setting,
not actual research source pools, templates, module order or independent roots.
Current read-only SSH attempt reached the university host but failed
`Permission denied (publickey,password)`. Jobs/results/cache/MIG/quota were **not**
re-inspected or synchronized. Historical smoke numbers are not current capacity.

Still required: actual 12-group root/source-family recipes and assignments,
independently adapted cached root artifacts/checksums, explicit A/B names/shapes/
module order, corpus/evaluation identities, all role manifests and parser bindings,
template hash, measured reviewed per-task caps, real continuation pools and the
remaining D06-D09 plus D12 measurement-reduction agreements. No roots are trained
by the collector; it starts from explicitly audited cached root adapters. Root
preparation is a separate real prerequisite, not seed-only copies of one parent.
Identity `root_sha256`/endpoint hashes use `sprint2_runtime.digest_state` over
the actual named trainable A/B tensors, **not** the adapter file-byte checksum;
the root `files` map retains those separate file SHA-256 values. Supplied root
identity must match actual loaded parameters before any state measurement.
Preflight deliberately requires the complete planned root/identity plan, even
when the exported batch contains only one train group. It also gates arbitrary
unsupported task parsers rather than guessing their correctness.

One consolidated question set was sent; the third received the pilot settings
recorded above. Dataset/scorer and feature-semantics questions remain unanswered.
D12 budget/D14 preparation are approved for this pilot; D06-D09 and D12 feature
reductions are not. The schema remains unfrozen, and approval strings/hashes
certify declared consistency only, not independent evidence or signatures.

### Exact Next Commands (Not Executed Real Runs)

From the project directory with its installed environment:

```bash
.venv/bin/python -B -m scripts.calibrate_generation_caps --config configs/sprint2_generation_calibration.yaml
.venv/bin/python -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --preflight
```

Both currently fail **before model loading** on missing real bindings. After task/
train-manifest binding, in a separately authorized existing compute allocation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.calibrate_generation_caps \
  --config configs/sprint2_generation_calibration.yaml --output runs/sprint2_caps_512_v1 --execute
# If censored/>2%: same prompts, increased explicit cap, NEW directory; no held-out tuning.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.calibrate_generation_caps \
  --config configs/sprint2_generation_calibration.yaml --recheck-cap 1024 \
  --output runs/sprint2_caps_1024_recheck_v1 --execute
```

1024 is an explicit recheck example, not a measured/approved final cap. Fix/review
caps from actual evidence; bind all configuration/definitions/data/roots. Then:

```bash
python -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --prepare-spec
python -B -m scripts.collect_sprint2 --config configs/sprint2_collection.yaml --dry-run --trajectories "$TRAIN_TRAJECTORY"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.collect_sprint2 \
  --config configs/sprint2_collection.yaml --trajectories "$TRAIN_TRAJECTORY" \
  --out runs/sprint2_first_train_t5_v1 --execute
# Resume the SAME configuration/source after an interrupted state:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.collect_sprint2 \
  --config configs/sprint2_collection.yaml --trajectories "$TRAIN_TRAJECTORY" \
  --out runs/sprint2_first_train_t5_v1 --resume --execute
python -m scripts.run_forecasting_eval --records "$EXPORT/meta.jsonl" \
  --split-manifest "$EXPORT/manifest.json" --handoff-spec "$EXPORT/spec.json" --validate-only
```

`TRAIN_TRAJECTORY` must be an actual preassigned train-group identity, currently
unavailable; `EXPORT` is the fresh export directory printed by the collector.
No Slurm submission is prepared/issued here. These commands cannot be truthfully
filled with invented dataset/trajectory IDs. After reviewing actual first-group
cost/completeness, 30 states means six assigned train groups x T5 =90 records;
that expansion requires the staged resource review, not automatic continuation.
Full D13 fitting needs separate assigned validation/test groups, not train-only
pilot rows; its strict manifest/spec command supports only completed measurements.

### Cost Scenario and Verification Boundary

One cached-root pilot: **30 full fits x32 =960 updates; 15 probes x15 =225**,
total1185 optimizer updates; 30 retained H1/H2 adapters plus the existing root
(selected parents reference H1 snapshots). Root fitting, if needed, is additional.
At the historical two-step smoke's coarse1.1-1.6s/update, update-only arithmetic
is **21.7-31.6 minutes**, not a measured real runtime. Endpoint scoring alone is
35 evaluations x(32+32+256)=**11,200 generated answers**; at an illustrative512
cap that is at most5,734,400 generated tokens, before confidence, source answers,
corrections, features, loading and restoration. Calibration initially permits
50x512=25,600 generated tokens/task. Generation throughput is unknown; the
two-hour ceiling **cannot yet be certified**, especially on the unknown MIG.

Using Sprint1's recorded2,949,120 adapter parameters and a float32-only scenario:
30 endpoint snapshots are353,894,400 bytes (~.330GiB), plus an11,796,480-byte root.
This excludes source IDs/texts/token traces, scoring/probe logs, other processes,
host snapshots and metadata; it is not measured artifact size or proof of5GiB fit.
The real first group must record wall time, peak allocation and actual bytes
before approving the six-group/90-row batch. Test success below is synthetic
plumbing only; no Gate1 or forecasting claim follows.

Current verification: **185 offline tests pass**, including existing Sprint1/
upstream forecasting regressions and tiny synthetic CPU fits. **35 Sprint2 cases
also pass with `python -O`**. RuntimeWarnings are errors; existing non-failing
LibreSSL/PEFT warnings remain. The exact18-row/6-state validate-only command below
passes `synthetic=True`, not Gate1/real-data validation. Five-state journal/backend
fixtures test CPU reductions, full/failure restoration (including replaced
nonpersistent buffers), interruption/resume, failed-attempt budgets, H1 history,
sealed shared/H1 draws, source/RNG/scorer/hash/leak guards and immutable exports.
A fully bound **fabricated temp-directory** preflight tests positive/negative
configuration paths without loading a model; it is not supplied research evidence.
AST syntax passes for all23 dirty/new Python files; all165 local document links/
line bounds, whitespace and credential-pattern
checks pass (pattern scanning is not proof of secret absence). Seven protected
pre-existing dirty files match their original byte hashes, including schema,
validator/tests, calibration config and prior fixture/tests. Tracked Sprint1
runtime/evidence diffs are empty. Branch `main`/HEAD `dcd403a` is unchanged;
no staging/commit/push occurred.

Verification commands:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -W error::RuntimeWarning \
  -m unittest discover -s tests -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -O -W error::RuntimeWarning \
  -m unittest discover -s tests -p 'test_sprint2_*.py' -q
```

Changed in this collector step (pre-existing uncommitted work preserved):

- New: `configs/sprint2_collection.yaml`, `scripts/collect_sprint2.py`;
  `src/third_eye/sprint2_runtime.py`, `sprint2_measurements.py`, `sprint2_backend.py`,
  `sprint2_collection.py`, `sprint2_evaluation.py`.
- Updated isolated helpers/consumer: `scripts/calibrate_generation_caps.py`,
  `scripts/run_forecasting_eval.py`, `src/third_eye/generation_calibration.py`,
  `sprint2_protocol.py`, `forecasting_handoff.py`, `forecasting/data.py`,
  `forecasting/train.py`.
- New synthetic tests: `tests/test_sprint2_measurements.py` (includes runtime),
  `test_sprint2_backend.py`, `test_sprint2_collection.py`, `test_sprint2_evaluation.py`;
  updated `tests/test_sprint2_protocol.py`.
- Updated existing documents: this integration report, feature audit, decision
  sheet and evaluation protocol. No new planning layer or schema freeze.

## Original Handoff Source Review

The local files, ignored-aware inventory and full fetched branch tree were
inspected, then the branch was fetched again after Shashank's question. The
branch was unchanged. No Mahitha-authored forecasting handoff document was found
in either repository tree at that review: branch `docs/` contained Sprint 1
reports/evidence and the feature audit. Shashank subsequently supplied
`Third_Eye_Mahitha_Modeling_Handoff.docx` from Downloads. Its entire paragraph/
table content was read, including the empty comments/footnotes/endnotes; there
are no tracked insertions/deletions. The original DOCX was not changed or copied
into Git. Source SHA-256:
`a2066e548ab619310a61afba29131cf2f6f85c469607fc2f103a8e0f83ced52b`.

The supplied document states it was prepared **10 Oct 2026 against base main
`da142ec`**, rather than identifying the final forecasting branch commit. Its
recommended defaults and schedule language are reference material, not new user
authorization or recorded joint agreement. The document actually recommends
**300 states per model/task**, not 30. Neither number is a validator minimum.
Actual code and the subsequent explicit user revisions govern the compatibility
findings. No repository `AGENTS.md` was found in either checkout.

Sources: [schema](../configs/feature_schema.yaml), [audit](sprint2_feature_audit.md),
[decisions](sprint2_contract_decisions.md), [evaluation draft](sprint2_evaluation_protocol.md),
[local protocol](../src/third_eye/sprint2_protocol.py), [Sprint 1](sprint1_report.md).
D01/D02/D04/D05/D13/D15 and explicit D03/D10/D11 revisions retain their recorded
status. D06-D09 and D12 measurement formulas are still unresolved; D12 budget and
provisional D14 pilot preparation/retention now follow the latest reply above.
Importing code is not approval
of Mahitha's recommended defaults; the schema remains unfrozen.

## Supplied Document Reconciliation

Section references below refer to the DOCX headings, not guessed page numbers.
This review does not accept its section 3 defaults on anyone's behalf.

| Document evidence | Current contract/code and review disposition |
| --- | --- |
| Sections 1 and 5: harness smoke results are not real forecasting evidence; planted-signal metrics must not appear as research results. | Consistent with Sprint 1 reports and current integration checks. The DOCX's 94-test/Python 3.12/torch 2.14 account is its author's report, not locally reproduced evidence. Our prior full suite passed 161 tests on Python 3.9.6/torch 2.8.0 CPU. Its old audit count 0/10/23 predates the current 0/13/20 producer audit; helper tests are not new measured features. |
| Section 3: six relative-depth energy bins replace the layer manifest; section 5 claims matching widths/order. | Conflicts with subsequently accepted D05 raw joint A/B norms per adapted module. Candidate width 22 matches raw `16+L` only when L happens to be 6; equal width is not equal meaning. Keep the isolated raw-module layout and explicit actual manifest; a transfer-bin transform remains a separate proposal, including its zero-energy rule. |
| Section 3: loss reductions may fit unchanged, although final labels should be accuracy fractions. | Legacy Sprint 1 loss experiments remain separate. Current D04 requires endpoint-minus-parent accuracy fractions and H2-only LHV2; no old smoke losses may be relabeled. All three H1 accuracy deltas are required for the comparable .50/.25/.25 one-step utility, not optional missing heads renormalized to target alone. |
| Section 3: H1 accepted utility, left padding/oldest first; D10 timing is still OPEN with an H2 switch. | Padding and H1 utility align, but the later user instruction settles D10 as applied, observed H1 only. The complete ledger, selected-candidate joins and ranking cutoff remain mandatory. H2 simulations are never history. Telescoping two changes on one actual chain does not supply an unobserved standardized sibling H2 endpoint. |
| Sections 3 and 4: D07 lexical components, D09 all-element signs with zeros logged; missing columns fail, newest history mismatch warns. | These remain proposed definitions. Actual D07 tolerance and D08/D09 token-weighting/epsilon differences are detailed below. The upstream row schema does not require zero diagnostics; helpers do not establish actual aligned gradients. Its any-sibling target-only history warning is weaker than an applied-provenance join. The strict adapter supplies those guards without inventing measurements. |
| Sections 3 and 7: fixed 8/2/2 lineage groups and hashes; example JSON omits group/hash/provenance and some commands omit the manifest. | Actual upstream group/hash fields are optional; hash checking is conditional on an expected hash. A bare example plus validate-only is not sufficient for D02/D15. Use the existing strict spec and exact bare group manifest, which is now read before validate-only returns. Declared metadata hashes do not approve unspecified feature formulas. |
| Sections 2, 3 and 6: meta-validation stopping, within-state Gate 2 Spearman, group-bootstrap recommendation, ranking/relative features off by default; PD skipped and Dynamics conditional. | Supports keeping extras inactive, not approving a new method or announcing a passed gate. Legacy code retains the D13 discrepancies described below; isolated `sprint2_d13` now implements accepted regret/native-MSE, baselines, exact ties and group uncertainty. Gate thresholds, Spearman/gate disposition and larger-study settings need explicit review. |
| Sections 1 and 6: dated freeze/pivot suggestions; sections 4/6 ask for a 300-state slice. | No automatic schema freeze, method pivot, label collection or execution permission follows. The 30-state pilot and accepted 12-group preliminary scope are not silently replaced. Review grouping, held-out allocation, measured compute/storage and power before expansion. |
| No explicit measured-cap protocol, transactional probe restoration, or D11 state-keyed source-draw/separate-RNG specification. | Current D03/D10/D11 revisions still apply. Safe runtime producers are now isolated/offline tested; actual tasks/scorers, cap measurements and semantic/source bindings remain blockers. Their absence from this handoff does not waive them. |

## Actual Formats and Compatibility

Mahitha's [record reader/validator](../src/third_eye/forecasting/data.py)
(`validate_records`, `from_records`, `fixed_split`) consumes **one JSONL row
per candidate**, not one row per state:

```text
model, task, trajectory_id, state_id, candidate_id
state: {every ordered layout state column: number}
candidate: {every ordered layout candidate column: number}
history: {values: {each of four channels: [H numbers]}, mask: [H binary numbers]}
labels: {delta_target_h1, delta_ood_h1, delta_retention_h1,
         delta_target_h2, delta_ood_h2, delta_retention_h2}
group_id, semantic_hash                 # optional upstream, mandatory in our handoff
```

`lhv2` is computed upstream; our export also preserves its validated YAML label.
Extra `provenance` is legal metadata, not model input. The split manifest is a
**bare JSON object** `{group_id: "train" | "val" | "test"}`, with exactly the
groups present in the supplied records. It is not a wrapped manifest or a
trajectory/row map. Local `validation` is explicitly serialized as `val`.
The full preassigned 12-group plan stays in the spec; a partial-batch CLI manifest
is its restriction to present groups, never a new outcome-dependent assignment.

| Boundary | Actual evidence and isolated compatibility action |
| --- | --- |
| Complete identities / independence | Upstream falls back to trajectory as group and accepts K>=2. The new [handoff adapter](../src/third_eye/forecasting_handoff.py), `validate_handoff`, requires pinned corpus/group/trajectory/root/evaluation identities, parent and descendant hashes, K=3, states 0..4, observed ticks and applied joins. It verifies planned source/prefix components and shared actual root hashes cannot cross groups/splits. Supplied identity assertions still require producer/source audits to establish their truth. |
| Fields / order / dimensions / dtype | Upstream `contract_layout` hard-codes the older layout: state 6, candidate 16+B, history 3x4 plus mask; B defaults to 6 depth-energy bins. YAML declares state 5, candidate 16+L and raw module norms; history includes separate int32 index and mask. `raw_module_layout` preserves YAML declaration order and explicit L/module names, moves the index to the final state column and separates the mask. Source arrays validate as float32/int32 before export; JSON has no native dtype. Consumer matrices are float32; integer index semantics are checked before conversion. No inference of L from model depth/rank or substitution of B for L. |
| Definition hashes | `SemanticConfig.fingerprint` is only 16 hex characters over weights/history/scoring/schema string; it does not bind all feature formulas, datasets or caps. Local schema/config fingerprints are different algorithms, not equivalent strings. The opt-in spec binds YAML bytes, ordered schema/config, actual layer order, layout, unresolved status and supplied bindings with full SHA-256. CLI enforces it for every row. This certifies consistency of declared metadata, not agreement or measurement truth. Real per-field extractor/scorer/calibration implementation IDs still need binding. |
| D10 | Upstream masks/counts are correct for `applied_h1`, with the weighted utility of each history slot; its previous-candidate target-label check is only a warning and can match any sibling. Adapter reconstructs the complete applied ledger, checks cutoff/identity/selected candidate joins, and verifies all four channels of the last min(t,3) H1 observations. Left padding only, oldest first, masks 0,1,2,3,3 across decisions. A real zero change remains masked 1. H2 label observations occur after the ranking cutoff and are never history. |
| D11 | No training-side continuation producer exists in her original stack. New isolated producer implements shared group/state draws and fresh per-branch updates; strict import verifies actual source IDs/RNGs, H1 overlaps, H2 exclusions and verified corrections. Real runtime evidence still must be collected. One draw; two unapproved. |
| CLI manifest validation | `--validate-only` validates manifest coverage/names before returning. `--handoff-spec` requires records+manifest and all H1 heads with applied H1 history; real records additionally require recorded scorer/count/source/restoration provenance. Fitting requires explicit `--protocol sprint2_d13`, held-out groups and, for real data, `--collection-config` preflight. Legacy/default paths remain separate. |

### Exact D05, D07 and D09 Findings

In [features.py](../src/third_eye/forecasting/features.py), `module_energy`
correctly combines A/B squared norms, but `block_energies` merges modules into
contiguous blocks and `energy_fractions_from_blocks` pools normalized bin energy.
At zero total energy it returns **uniform 1/B**, not measured module norms or
evidence of nonzero energy. Accepted D05 instead needs one joint A/B norm per
actual adapted module, numeric block then explicit placement order. The raw
adapter avoids this transform; cross-model depth binning needs separate review.

`duplicate_rate` implements the D07 proposal: NFC/whitespace normalization,
case/punctuation preserved, 3-gram set Jaccard >=0.90 with a `1e-12` boundary
tolerance, short-text exact normalized equality, transitive components and
`(N-components)/N`. `[A,A,B]` gives 1/3. It is still a proposal, not a source-family
independence proof; decide the tolerance explicitly before binding its hash.

`grad_l2_norm`, `grad_cosine`, `sign_agreement`, `zero_sign_stats` use flat LoRA
A/B arrays, declared token-mean answer CE, unscaled/pre-clipping gradients;
cosine returns zero when either norm is **<1e-12**. Sign(0)=0, both zeros agree;
the helper returns `zero_in_a`, `zero_in_b`, `zero_in_both`. Local D08/D09 propose
equal per-answer mean NLL, eval-mode same-parent gradients and an exact-zero
fallback, so these are not identical definitions. The helpers alone do not
verify parameter alignment, answer spans or eval mode; cosine/sign helpers also
do not independently reject all empty/nonfinite cases. The adapter requires
explicit zero-norm/zero-sign diagnostic metadata and valid fractions, not guessed
zeros. Actual gradient tensors and extraction/restoration checks remain missing.

Other live disagreements: token-weighted versus equal-answer NLL; multiple-attempt
failure probability versus original failure fraction; missing embedding encoder/
pooling provenance; confidence span/failure handling and short-probe policy.

## Verification and Reproduction

[Fixture builder](../scripts/build_synthetic_handoff.py) creates 18 explicitly
fabricated rows: three groups x two decisions x three candidates; six rows in
each of train/val/test. This is a format fixture, **not a completed T=5 corpus**.
Its spec records unbound, unexecuted calibration; no measured cap is invented.
Artifacts are in ignored `runs/sprint2_handoff_synthetic_20261011_v3/`.
The builder refuses an existing output directory. Reproduce in a new directory:

```bash
.venv/bin/python -B -m scripts.build_synthetic_handoff --out runs/NEW_synthetic_handoff
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -W error::RuntimeWarning \
  -m scripts.run_forecasting_eval \
  --records runs/NEW_synthetic_handoff/records.jsonl \
  --split-manifest runs/NEW_synthetic_handoff/split_manifest.json \
  --handoff-spec runs/NEW_synthetic_handoff/handoff_spec.json --validate-only
```

Actual validate-only result: 18 rows, six states, three trajectories, K=3;
structural/provenance checks pass, synthetic=True. The warning about only three
lineage groups is expected. No Gate 1 result or forecasting metric is claimed.
Full T=5 fixtures additionally exercise all history lengths, true-zero observations,
all three raw-layout model forward passes on CPU, wrong shapes/fields/hashes/
order, future/simulated history, selected joins, descendant identity, split leaks,
shared draws and child-specific batches. Utility's identical weighted sum now
uses elementwise reduction, eliminating reproduced local matmul RuntimeWarnings.
Prior pre-collector verification: **161 offline tests passed** (59 upstream forecasting,
25 new handoff, 67 existing Sprint 2, 10 Sprint 1), with RuntimeWarnings treated
as errors. Non-failing LibreSSL and PEFT missing-config warnings from existing
Sprint 1 tests remain. AST syntax passes for all five changed/new Python files;
Git whitespace and credential-pattern checks pass. Original dirty-file SHA-256
checksums still match. No synthetic fixture metrics are published as research.
On receipt of the DOCX, the 25 isolated handoff tests and the 18-row validate-only
command were rerun; no feature/training code was changed for this document review.

## Real Records: Exact Remaining Work

1. **Dataset/scorer bindings (D03):** pin actual task IDs, deterministic verifier/
   final-answer parser code/version, tokenizer/chat template, disjoint source-family
   adaptation manifests, target diagnostics, OOD proxy, 256 unique retention-anchor
   IDs, confidence/probe/retention-gradient subsets and separate untouched final LM
   evaluation sets. Record correct/total counts, failed parses and truncation for
   every evaluated parent/H1/H2 endpoint; never reinterpret smoke losses as accuracy.
2. **Cap calibration (D03):** bind real LM-training prompts and preassigned forecaster
   train groups in `sprint2_generation_calibration.yaml` (both lists are currently
   empty). Approximately 50 prompts/task, greedy base, initial 512, exact numeric
   or uppercase A-D `Final answer: VALUE` line outside think spans terminated by
   newline/EOS only; task suitability is not established. Record every length,
   correctness, stop reason and truncation. Reject a cap estimate if censored or
   insufficient; flag >2% all-prompt truncation. Review/fix caps before collecting
   labels, never using held-out outcomes. Offline preflight currently exits 2 with
   `Unresolved task IDs/training manifests`, before model loading.
3. **Source/group/model configuration:** preassign the complete 12-group 8/2/2 plan
   outcome-blind; produce actual independent adapted roots and source-family audit.
   Bind cached model revision, quantization, actual numeric-block/placement module
   manifest and A/B parameter shapes/order, dataset/scorer/cap hashes and generation
   policy. Pin the concrete one-draw continuation recipe, source pools, correction
   verifier, empty-failure-batch policy and full-update optimizer/reset settings.
4. **Measured feature producers:** isolated code now implements parent scoring/confidence; token/span
   lengths, NLL, corrections/attempts/difficulty, diversity and duplicate component
   evidence; aligned pre-update candidate/retention gradients and diagnostics;
   probe slope/KL/adapter-change extraction. D06-D09/D12 still need agreement.
   Actual data/formula bindings and allocated-model execution remain unverified.
   Missing measurements stop collection, not zero-fill or label-derived features.
5. **Probe/full-update separation:** transactional candidate probes must restore
   weights/buffers, train/eval modes, gradients, optimizer/scheduler/scaler, Python/
   NumPy/CPU/CUDA RNG and sampler position in `finally`; verify hashes/logits before
   the next sibling. Prefer isolated disposable probe state. No history advance or
   parent promotion from a probe. Full candidate H1 and shared-draw H2 endpoints are
   separate measured labels; only an explicitly selected successfully applied H1
   result may enter the next state's as-of ledger. These are implemented in the
   isolated backend/runtime, not by the metadata validator; GPU verification pending.
6. **Resources/retention (D14):** inspect server snapshot/jobs/results, compare code
   before any new authorized run, preserve server-only Slurm files and outputs.
   Obtain current quota/allocation/storage approval, measure throughput/peak memory
   and bytes, retain small records/manifests/logs/checksums and approved adapter
   artifacts for reproducibility. Latest preparation retains all pilot adapters
   under provisional two-hour/5 GiB ceilings. No automatic branch/output deletion
   or base copies; no execution approval or verified fit follows.

## Gate 1 Versus Full Evaluation

[`gate1`](../src/third_eye/forecasting/selection.py) uses real weighted H1 versus
weighted H2 best-set disagreement >=20%, **or** >=15% of candidate rows having
any negative H2 accuracy delta. It reports H1 harm but does not use it for passing;
ties use atol=1e-12 and margin defaults to zero. There is no minimum of 30 states
in code. It needs all siblings' real H1/H2 outcomes on identical fixed diagnostic
sets with common-parent deltas and standardized continuations, not a trained
forecaster. All three H1 channels are needed for the agreed H1 utility; target-only
would silently change it by renormalization. Gate 1 can inspect outcomes without
all input features, but passing it is not learnability or collector completeness.
Use train-group data for the exploratory gate, not held-out data to revise policy.
Thresholds/denominators/tie tolerance are upstream defaults, not a new agreement.

Full evaluation additionally needs complete measured features, independently
preassigned held-out groups, validated history/probe transactions, calibrated
fixed generation policy and the accepted training-only normalization/baselines.
**Legacy D13 differences, now fixed only in opt-in `sprint2_d13`:**
`train.fit` selects epochs by normalized validation Huber loss, not validation
LHV regret then native MSE. Normalization is row-weighted, not the local nested
group/trajectory/state weighting; equal-size pilot groups only make these coincide.
The CV report's default constant reference is the evaluation-label mean; manifest
mode supplies the training-group mean but omits the complete baseline/Gate 1 policy
table. The default heuristic is verifier pass rate, not accepted highest pre-update
NLL. Reports lack explicit MSE and use 1e-12 near-ties rather than accepted exact
ties; some policy CIs bootstrap states instead of lineage groups.
The isolated path corrects these for future real fitting; do not alter Sprint 1 or accept
held-out/transductive normalization. Two held-out groups support preliminary
results only, with group-level uncertainty, not candidate-row precision.

## State Counts and Proposed Collection Order

**Clarification from the supplied handoff:** section 4 recommends roughly
**300 states per model/task (about 60 five-decision chains)** and section 6 asks
for a first 300-state slice. It does not request 30 states. The smaller 30-state
batch below is our staged proposal, not a quotation, gate minimum or approved
replacement for the document's broader recommendation.

A state is one unique **(trajectory_id, state_id)** shared by its three candidate
rows. Thus 30 states means 90 candidate records and 90 H2 descendants, not 30
rows or 30 independent trajectories. Under agreed T=5, one complete trajectory
per independent group gives **six groups x five decisions = 30 states**. Dependence
within a group remains; the independent sample size is six, not 90. The final
decision's simulated H2 endpoint can extend to S6 without adding an applied core
decision beyond S0..S4 ending at S5.

The document's 300 states would give **900 candidate rows/H2 descendants** with
K=3, still keeping T=5 per chain. Sixty chains do not imply sixty independent
groups: shared roots/prefixes/source families must merge. For illustration only,
five chains in each of the existing twelve groups yield 300 states but still
only twelve independent units; equal-size 8/2/2 assignment gives 200/50/50 states
or 600/150/150 rows, not 300 held-out test states. Sixty genuinely independent
groups would instead require a newly reviewed split/count policy; the current
helpers deliberately preserve the full 12-group 8/2/2 plan. Do not create that
expansion, change the assignment or treat correlated chains as new groups here.

Section 4's approximate top-1 standard errors assume independent state trials.
They are not a lineage-cluster power analysis or a promise of 45% top-1 detection.
An explicit held-out-group target, intragroup dependence and uncertainty method
must be agreed before using those approximations to justify collection scale.

**Proposal, not an approved new budget:** preassign all 12 groups before labels;
use six of the eight assigned train groups for an initial 30-state batch. First
validate one complete train-group T=5 trajectory, then a second and review observed
cost/completeness as the existing draft requests. Only then collect four more train
groups. Finish the remaining two train, two validation and two test groups without
changing assignment: 60 states/180 rows total, 120/30/30 rows per split. Never mix
candidates, descendants, resumed/shared prefixes or correction families across
splits. Keep test labels inaccessible to model selection.

With root construction counted as one fit/group and selected H1 promoted rather
than retrained, 30 states imply **6 root fits + 90 H1 + 90 H2 full candidate fits**,
plus 90 temporary probes. This is operation counting, not an approved step budget,
runtime or storage estimate. Generation/scoring and feature-extraction costs are
unmeasured and additional; use the first complete group to replace draft estimates.

**Smallest complete real collector check before that batch:** one preassigned
train group, one T=5 chain, K=3, one shared H2 continuation/state: 15 complete
candidate records/H2 descendants, all parent/H1/H2 accuracy channels and input
features, history lengths 0/1/2/3/3, rollback, applied joins, reload and validation.
A two-decision/six-row wiring preflight may fail-fast before finishing that same
group, but is not T=5 completion. Run validate-only with its restricted train-only
manifest and the pinned spec, plus CPU label-free model forward checks. Preserve
the other groups unopened. This checks real measurement plumbing, not forecaster
generalization; a real forecaster fit/model-selection evaluation requires separate
train/validation groups and the opt-in D13 path. Calibration and all blockers above must
be resolved before this real check; none was executed in this task.
