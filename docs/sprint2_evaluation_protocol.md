# Sprint 2: Evaluation Protocol Draft

**PARTLY AGREED DRAFT, updated 2026-10-11. Not a schema freeze,
execution authorization, or a claim of research readiness.**

This document proposes an independent-group, multi-state forecaster evaluation.
It does not authorize real label collection, training or changing Sprint 1.
The isolated collector and opt-in D13 evaluation are now offline implemented;
see the [integration report](sprint2_handoff_integration.md#collection-tooling-update-2026-10-11)
for exact commands, verified tests and real-data blockers.
D01/D02/D04/D05/D13/D15 are accepted by the current user instruction, with explicit
D03 calibration, D10 applied-H1-history and D11 shared-draw revisions; see the
[decision sheet](sprint2_contract_decisions.md). T=5, preliminary 8/2/2 grouping,
fraction-delta utility and train-only evaluation rules now follow that instruction.
The latest reply accepts one-group pilot source draws/selection, 15/32 optimizer
updates, 32-example diagnostics, 256 anchors and provisional two-hour/5 GiB
preparation with all pilot adapters retained. Actual datasets/feature semantics
remain unbound; wider-corpus budgets remain proposals. No launch permission.

## Evidence and Scope

Read alongside the [feature audit](sprint2_feature_audit.md),
[Sprint 1 completion report](sprint1_report.md),
[progress record](sprint1_progress.md), and the reports for
[job 25765](experiments/qwen3_4b_smoke_25765/report.md) and
[job 26946](experiments/sprint1_qlora_26946/report.md).
The repository's current schema is [0.2-sprint2](../configs/feature_schema.yaml),
with `frozen_on: null`. Its header is not approval.

Sprint 1 verified tiny-data pipeline infrastructure, not forecasting accuracy.
Job 26946 had one state, three identical candidate input vectors, no held-out
evaluation, and an unused history encoder. Its training MSE approximately matched
the constant-output baseline. Job 25765 saved no forecaster weights; its exact run
revision is unknown and is not attributed to the later `6700ed5` synchronization.
Job 26946 has verified source provenance at `833da0c`, not the later publication
commit. Neither run becomes a test set for this protocol: their outcomes are
already inspected and their feature meanings differ from the proposed contract.

Current [pilot generation and promotion](../scripts/run_sprint1_pilot.py#L152)
collect full outcomes before promoting by t+1 target loss. The
[forecaster fit](../scripts/fit_direct_forecaster.py#L18) assumes one state and
three rows; its constant mean is computed on those same training rows. These
scripts do not implement the split/evaluation protocol below. New opt-in tooling
is separate in `collect_sprint2.py` and `sprint2_d13`; preserve the working v1 path.

## Trajectories and Independent Groups

A **trajectory** is one chronological applied lineage
`S_0 -> S_1 -> ... -> S_T`, starting from a recorded root adapter/model state.
At decision state `S_t`, record all `K` proposals from the same unchanged parent,
the inputs observable at the ranking cutoff, full candidate t+1 outcomes, and the
declared H=2 continuation outcome for each candidate. Only the chosen/applied t+1
successor advances the actual lineage. A simulated H=2 descendant is an offline
supervision branch, not automatically the applied `S_(t+2)`.

An **independent trajectory group** is a connected component of related lineages,
not a job ID, candidate row, state, random seed, or workspace directory. Connect
records if they share an adapted root/ancestor checkpoint, any applied prefix,
candidate/descendant branch, a resumed/copied trajectory, or a group-specific
update/correction source pool or duplicate source family. Seed-only replicates of
the same root/data lineage belong together. Transitive links merge groups: if A
shares a prefix with B and B with C, all three remain one group.

For the proposed initial design, each group has one independently adapted root,
one applied trajectory, disjoint group-specific adaptation/update source families,
and separately declared random streams. Cross-group deduplication must use source
identity and an agreed duplicate rule, not text hashes alone. Independent group
IDs are a design assertion requiring a lineage/data audit, not statistical proof.

Sharing the immutable pretrained base/tokenizer and fixed evaluation sets is
intentional and does not merge every group. The resulting claim is conditional
generalization to new adaptation trajectories of that same base/task/evaluation
population, not independence across base models or novel evaluation examples.
A shared adapted parent does merge groups. Different model/domain regimes need
explicit strata or a separate held-out-regime study, not inflated group counts.

## Assignment Before Label Collection

1. Declare source pools, duplicate families, root recipes, allowed proposal
   variation, applied-lineage collection policy, H=2 continuation policy,
   failure/retry rules and planned group IDs before seeing candidate outcomes.
   Build the dependency components from recipes and existing lineage metadata.
2. Write a versioned split manifest with `group_id`, group membership,
   `trajectory_id`, root recipe/ancestor IDs, data-family manifests, split,
   assignment algorithm/seed, source/config/schema hashes, and creation time.
   Add realized root hashes after execution without changing split membership.
3. Accepted 8/2/2 preliminary assignment: sort groups by SHA-256 of a canonical JSON encoding of
   `[split_seed, group_id]`, with group ID as collision tie-breaker. Use a
   predeclared seed (proposal: `20261010`) and allocate the first eight groups to
   train, next two to validation, last two to test. Do not search for a seed that
   produces favorable labels. Outcome-blind strata, if needed, must be declared
   before assignment and would require a revised count/assignment plan.
4. Every state's candidates, full t+1 records, H=2 descendants, shared prefixes,
   history rows, resumes and retries inherit their group's split. Never shuffle
   candidate rows, slice a trajectory by state/time, or split a shared branch.
   Different Slurm jobs do not create new independent units.
5. Keep feature records and labels separate, joined only on full
   `(corpus_version, group_id, trajectory_id, state_id, candidate_id)` identities,
   with parent/continuation/evaluation hashes and observation provenance checked.
   Test labels may be collected offline but must remain inaccessible to fitting,
   feature construction and model-selection code until final evaluation.
6. Record missing/failed states and attempts; do not replace inconvenient
   outcomes, move groups to rebalance results, or select complete groups by label
   quality. A replacement recipe, if required, must be outcome-blind and retain
   the failed group's split. Report planned versus usable counts by split.
7. Recheck lineage and data-family overlap before fitting. A newly discovered
   cross-split dependency invalidates the split; stop, record prior exposure and
   redesign/version it before further collection. Relabeling already inspected
   test data does not restore an untouched test set.

The actual manifest is not yet supplied. Pure component/assignment helpers in
[sprint2_protocol.py](../src/third_eye/sprint2_protocol.py) derive canonical group
IDs from sorted connected trajectory IDs; hashes/seed provide order, not proof of
independence. The manifest is not a feature-schema freeze. Changes
to definitions, continuation rules or source pools require explicit review and
corpus versioning, not silent reinterpretation of old labels.

## Two Different Dataset Axes

Forecaster train/validation/test partition **trajectory groups and their feature /
label records**. They are not the following language-model datasets:

Language-model root/candidate/continuation fitting inside a forecaster test group
would be label collection for that group, not forecaster training. Its records
must still stay out of forecaster optimization and fitted preprocessing. Likewise,
the language model's target-validation set is not the forecaster's validation
split and does not authorize opening forecaster test labels for model selection.

| Language-model dataset role | Permitted use | Separation rule |
| --- | --- | --- |
| Adaptation/training and candidate-correction pools | Root fitting, candidate updates and declared continuation/probe training. | Group-specific source families stay in one forecaster split; no evaluation examples enter updates. |
| Target-validation set | Agreed parent performance, permitted probe measurements and full outcome labels. | A fixed diagnostic set, not forecaster validation. It is exposed to the update/selection workflow and is not final held-out evaluation. |
| OOD proxy set | Agreed OOD feature/label diagnostics. | Version/hash/domain boundary required; the proxy is exposed, not evidence of final OOD generalization. |
| Retention anchor | Fixed reference population for accuracy/loss and allowed gradient/probe measurements. | Accepted pilot size 256 unique examples; identity/scoring remains unresolved. The anchor is exposed and not a final retention test. |
| Probe population/subsets | Only the agreed bounded pre-selection measurements. | Declare overlap with diagnostic sets explicitly; never use final evaluation examples. Full t+1/t+2 results are not probe inputs. |
| Final language-model evaluation sets | One final report of actual selected models' target/OOD/retention behavior under the locked study recipe. | Separate, previously unexposed source families; no updates, features, labels, normalization, early stopping or ranking choices may use them. |

Deduplicate across these roles by underlying example/source family, including
paraphrases or repeated generated problems, under the agreed rule. Record all
allowed diagnostic overlaps; never treat an undeclared overlap as independence.
The same diagnostic target/OOD/anchor manifests may serve all forecaster splits
for this fixed-task study; disclose that evaluation population reuse. A separate
new-example/domain claim needs separate held-out populations and a new protocol.

Feature and label scoring must match the agreed metrics, units and population.
Sprint 1's full-text token loss cannot be relabeled as answer accuracy. Neither
forecaster test labels on diagnostic sets nor simulated branch scores substitute
for the final language-model evaluation of actual selected models.

## Collection Policy and As-Of Features

Accepted first pilot configuration: collect a fixed corpus with a predeclared,
state-seeded uniform candidate policy, sealed before features/outcomes, to advance
each applied trajectory. This keeps the state
distribution independent of whichever forecaster later wins validation. Record
that collection policy and its seeds. Comparisons on this corpus measure
**one-decision selection on common states**, not closed-loop trajectory returns.
Evaluating policies that reach different future states needs a separately
authorized closed-loop collection and group-preserving evaluation.

At each state, snapshot inputs/provenance at the ranking cutoff, before full
candidate updates or continuation labels. All candidate probes start from the
same immutable parent and restore/dispose of parent, optimizer, gradients,
modes, RNG and data-position state as specified in the
[audit's restoration proposal](sprint2_feature_audit.md#parent-and-training-state-restoration-contract).
The legacy raw-logit `probe()` is not an update-probe feature extractor. New
isolated update probes finally-restore and fingerprint all explicit owned state;
their OLS/KL/span formulas remain pending despite the approved cadence/budget.

D11 uses **one shared continuation draw per `(group_id,state)`**, fixed before
outcomes, with identical recipe, seeds and source prompts across all three
siblings. Each child may generate different failures/corrected training examples
from those shared prompts. Seal a plan hash, not a candidate-keyed source draw.
Use separate source, generation, correction and training RNG streams; generation/
correction seeds are per-prompt and identical for siblings. Variable child
processing must not advance the shared draw RNG. The pure shared builder accepts
no outcomes or candidate ID. Pilot H1 draws are three candidate-keyed sets of32,
same pool/procedure, sibling overlap logged; H2 draws32 once excluding their union.
Fresh AdamW each branch/probe. Empty verified failure batches stop/preserve, never
redraw/no-op. Actual pool/correction semantics still need binding. **Two draws
remain optional and unapproved.** One realization is
a noisy conditional label, not an expectation over future policies.

D10 history is the last `min(t,3)` **applied observed H1 transitions** at decision
t, oldest-first, left zero padding, real-row mask 1. Require complete as-of applied
ledger identity, parent/endpoint chain, matching evaluation identity and cutoff.
Every delta is its own H1 endpoint minus parent accuracy; accepted value uses
0.50/0.25/0.25 on those H1 deltas. Real zero changes are observed rows; missing
expected applied results fail, never pad or silently skip. No forecasts, H2
outcomes or simulated branches enter this history. Thus no bootstrap predictor
is required for accepted value. `lhv2` remains strictly H2 supervision/ranking.
The collator/validator checks supplied ledger consistency; actual recording/join
integrity needs real provenance auditing; isolated recording/joins and integrated
Direct/Dynamics mask-aware encoding are implemented and fixture-tested. Never backfill observations
that were unavailable at the ranking cutoff.

## Fitting and Model Selection

- Fit feature normalization, clipping/imputation rules (if explicitly permitted),
  target transforms and any learned embedding/extractor component on **train
  groups only**. Record fitted parameters, training IDs and layout fingerprint.
  Externally fixed pretrained components must be pinned and disclosed. Apply the
  unchanged transforms to validation/test; never fit them on the full corpus.
- Normalize continuous values using real observations only; padding must remain
  zero after transformation. Do not normalize masks or infer missing observations
  from zeros. Opt-in raw-module layout puts checked index in the state column;
  Direct/Dynamics encode only observed history and ignore masked padding.
  Handle zero-variance training channels explicitly and log held-out out-of-range
  values; no test-dependent clipping thresholds or silent nonfinite substitution.
- Train forecasters and fit all learned baselines using train groups. Accepted
  weighting gives each group equal weight, then each trajectory, state and
  candidate within it equal weight; long trajectories must not dominate.
- Use validation for architecture/hyperparameters, stopping, rank-selection rule
  and any calibration choices. Record every tried configuration and a fixed
  selection criterion. Accepted primary criterion is group-macro validation
  `lhv2` regret, then mean native-unit MSE over three H2 channels as tie-breaker.
  Model search/trial budget is still pending. Only tiny synthetic CPU fits were
  tested here, not real forecasting validation.
- Keep the selected training-only fit and transforms for the first final test;
  no train+validation refit is proposed for this small corpus. Lock a versioned
  evaluation recipe including model hash, metric formulas and tie rules before
  accessing test labels. Test is used once for final reporting, not tuning or
  choosing the best random seed. Later changes require a new untouched test set.

## Baselines, Prediction Error and Regret

Let `y_(g,s,k,j)` be candidate k's recorded outcome channel j at state s of group g.
H1 target delta and the three H2 deltas are separate prediction tasks. Accepted
D04 defines accuracy fractions and endpoint-minus-parent deltas, positive for
improvement. `lhv2 = .50*target_H2 + .25*OOD_H2 + .25*retention_H2`, measured from
the same parent to the standardized H2 descendant (not H2 minus H1). It need not
be an additional Direct output. Below `u_(g,s,k)` is that higher-is-better utility;
apply identical aggregation to predicted and realized H2 deltas. Historical loss
files cannot supply unrecorded accuracies.

| Comparator | Fit / selection rule | What it can report |
| --- | --- | --- |
| Constant-output predictor | For each channel, the group-weighted train-label mean; reuse that vector at every validation/test candidate. No held-out mean recomputation. | Per-channel prediction error; its ranking ties all candidates. Aggregate predicted deltas with the agreed utility rule, not a fabricated LHV formula. |
| Uniform random candidate selection | Uniform over the complete K candidates at each state, independent of inputs/outcomes. No fitted parameters. | Expected selection regret, exactly enumerated from all K labels; no prediction error unless a separate predictor is declared. |
| Accepted simple ranking baseline | **Highest `pre_update_nll` first**, a pre-update hard-example heuristic measured at the unchanged parent. Its direction is predeclared, not chosen from test results. | Selection regret only. NLL reduction/text span/data variation remain unresolved (D08); if absent, report unavailable, never substitute full-update training loss. Any later calibration must use train only. |
| Proposed forecaster | Predict declared channels from as-of inputs; rank by agreed utility of predictions. | Both prediction error and selection regret; no uncertainty/confidence claim without a separate agreed definition. |

The NLL heuristic is not claimed to be effective. Ensure proposal/data variation
can produce distinct measured inputs; Sprint 1's seed-only identical vectors do
not test ranking. Identical baseline scores are legitimate ties, not a reason to
change its direction after seeing test results. Any alternate baseline must be
declared and selected on validation before the final test.

**Prediction error:** report MAE, MSE and RMSE for every channel in original
declared units, after reversing training-only target transforms. Average squared
or absolute errors within state over candidates, then within trajectory, then
equally over trajectories in group and groups in split. Compute RMSE as the square
root of that macro MSE, not the mean of per-group RMSEs. Any scalar multi-channel
average requires declared scaling/weights; do not hide OOD/retention errors behind
target-only error. Report constant baseline errors on the same usable records.

**Regret:** on the complete candidate set at a state,
`u_best = max_k u_k` and `R(selection) = u_best - u_selected`. The oracle uses
recorded outcomes only for scoring, never for feature construction, training-data
selection or trajectory advancement. This is conditional regret over the K
collected candidates and declared continuation realization, not global optimality.
Report native-unit regret, selected target/OOD/retention outcomes, and the same
state/trajectory/group macro aggregation. Do not invent range-normalized regret
when all utilities match or when units are unresolved.

**Explicit ties:** accepted policy is exact ties (`eps_abs = eps_rel = 0`). If near-ties
are desired, agree training-only numerical-repeatability tolerances before
validation selection/test access. For any ranking scores a, use the maximum-
anchored set `T(a) = {k: max(a)-a_k <= eps_abs + eps_rel*abs(max(a))}` with
metric-specific units; do not chain pairwise approximately equal comparisons.
Policies randomize uniformly within `T(a)`, not by row order or candidate ID.
For evaluation report the exact expected tied-policy regret
`u_best - mean_(k in T(a)) u_k`, plus tie-set sizes/tie rate. Uniform random regret
is `u_best - mean_k u_k`, so constant-output ranking and random selection coincide.
For a realized rollout, save its predeclared tie RNG stream; do not pick a lucky
seed. With tied realized best utility, any exact best candidate has zero regret.
Report top-choice success as probability of selecting a member of the declared
realized-best tie set, separately from raw regret; near-best need not have zero
raw regret. A declared secondary tie-breaker would be a different policy.

Use identical complete states for paired comparisons. Nonfinite values, missing
candidates or failed continuations invalidate a state's primary regret score;
do not shrink K silently or fall back to an oracle. Report exclusions/coverage
and failures by split, group and method; partial-state prediction error may be
reported only as a separate diagnostic. Publish per-group metrics, paired
differences versus each baseline, and dispersion. If interval estimates are
used, resample independent groups, not candidate rows; with only two held-out
groups, intervals are fragile and cannot support a significance/readiness claim.

## Initial Corpus and Proposed Budgets

| Quantity | Accepted scope versus proposed binding/budget |
| --- | --- |
| Base/config | One pinned Qwen3-4B NF4/bf16 regime, no multi-model study or hyperparameter sweep. |
| Independent groups / split | Accepted preliminary 12 groups: 8 train, 2 validation, 2 test. One trajectory/root recipe per group is the initial design binding, not observed independence. |
| Decisions per trajectory | Accepted T=5: decisions `S_0..S_4`, five applied updates, ending at `S_5`; four-state stage cannot fulfill the core scope. |
| Candidates / continuation | K=3; accepted one shared draw per group/state, three child-specific H2 descendants. |
| Records | 60 complete decision states, 180 candidate rows/descendants; split rows 120 / 30 / 30. Distinct H1/H2 labels and fixed weighted H2 LHV. |
| Full adaptation budget | Accepted first-pilot H1/H2: 32 optimizer updates, batch1/accumulation1 sealed-order cycling, supported lr1e-4/rank8/q+v, fresh AdamW/no scheduler. Existing max length128 fails on lost answer spans. Root preparation/wider-corpus recipes remain separate prerequisites/proposals. |
| Temporary probe | Accepted pilot 15 optimizer updates; fixed population losses at0/5/10/15, separate optimizer-loss trace. Target32/OOD32, retention256, probe/retention-gradient32. Reduction formulas/actual IDs remain unbound; 15/32 is substantial overhead. |
| Collection order | **One assigned train group T=5 first: 15 records**, validate and measure cost. Only after review expand to six train groups =30 states/90 records. Complete the broader12-group plan separately; no relabeling held-out groups. |

### First-Group Preparation Versus Wider Proposal

Current preparation uses one MIG and provisional **two-hour / 5 GiB** ceilings,
retaining roots/selected parents/all pilot H1/H2 adapters and small evidence.
No deletion/pruning/base duplication. Server authentication currently fails;
current MIG/cache/free quota and fit must be inspected before execution.
Cached-root group cost is **960 full +225 probe =1,185 optimizer updates**;
35 parent/H1/H2 endpoint scorings of320 examples =**11,200 generated answers**,
before source corrections/confidence/features. The illustrative float32 endpoint
parameter storage is~.330 GiB; actual artifact size and generation throughput are
unknown. Two-hour/5 GiB fit is **not certified**. Details and exact calibration/
collection/resume/validate-only commands are in the integration report.
The historical 12-group arithmetic below is a **wider, unapproved scenario**,
not the pilot's current resource caps or adapter-retention policy.

Five decisions expose H1 history lengths **0,1,2,3,3**; H2 labels are never history.
Two groups per held-out split support only preliminary evaluation. Uncertainty
must be assessed by lineage group, not 180 correlated candidate rows; no robust
significance or research-readiness claim follows from this count.

### Compute Estimate: Conditional Planning Arithmetic

Recorded [job 26946 metrics](experiments/sprint1_qlora_26946/metrics.json) show
315 seconds Slurm runtime, 253.655 seconds Python runtime, and seven full fits
(one root, three candidates, three descendants), each with two optimizer steps.
Parent fit took 3.149706 seconds; a recorded candidate fit took 2.148273 seconds.
Thus approximately 1.1-1.6 seconds/step is a coarse rate inferred from these
two-step tiny-data fits, **not measured throughput for the proposed corpus**.
The full pilot also includes repeated loading/evaluation/checkpoint checks and
a 200-step CPU forecaster fit. Do not interpret 315 seconds as training alone.

Planning arithmetic for the proposal:

- Full fits: `12 + 60*3 + 60*3 = 372`; promotion reuses a trained candidate.
  `372*32 = 11,904` full-update optimizer steps, plus `180*15 = 2,700` probe steps.
  Additional descendant branches or probe replicas increase this budget.
- Repeating a whole pilot envelope per decision gives `60*315 = 18,900` seconds
  (**5.25 allocated MIG-device hours**) for the original tiny two-step workload.
  This intentionally includes redundant root/CPU-fit overhead; it is a scenario
  anchor, not an upper bound for new evaluation populations.
- Raising the 372 required fits from 2 to 32 steps adds 11,160 steps. Together
  with 2,700 probe steps, at the above coarse 1.1-1.6 seconds/step this adds
  approximately 4.2-6.2 hours: **about 9.5-11.4 device hours before new feature /
  scoring overhead**. Scaling beyond the observed two-step regime is unverified.
- Accuracy generation, the 256-example anchor, retention gradients, embedding
  extraction, slopes/KL evaluation and final evaluation have no measured cost
  here. Proposal: reserve up to **16 device hours total**, including the staged
  review, and stop/re-estimate if measured costs would exceed it. This is a cap
  to discuss, not a forecast that all required features will fit that budget.

Count MIG-slice allocation hours, not full-A100 equivalent hours or currency.
Queue wait is not included. Proposed single concurrent `2g.10gb` allocation,
four CPUs and 32 GiB requested host memory mirror job 26946; its peak allocated /
reserved device memory was 3.950179 / 5.308594 GiB, not assurance for gradient-
comparison/embedding probes or larger scoring sets. Future preflight must measure
capacity and replan rather than submit automatically. Forecaster model-search
CPU cost is unmeasured and needs a separate bounded trial budget before fitting.

### Storage Estimate and Retention Proposal

Recorded home usage increased from 43,417,399,296 to 43,994,603,520 bytes around
job 26946: **577,204,224 bytes (0.577 GB)**. This is a home-usage delta, not an
isolated artifact-size measurement; it may include other files/activity. Keeping
60 whole-pilot envelopes would provision roughly **34.6 GB**, exceeding the
approximately **6.0 GB historical headroom** under the recorded 50 GB quota.
Current usage is unknown; do not treat these October 9 readings as current.

Parameter-only scenario: the recorded 2,949,120 trainable parameters times four
bytes (explicit float32 assumption) is 11,796,480 bytes per unique adapter.
For 372 unique root/candidate/descendant fits this is **4.39 GB before tokenizer,
metadata, duplicate parent/promoted copies, logs and any optimizer state**. It is
not a measured saved-adapter size or an upper bound; serialized dtype is to verify.

Historical wider-corpus proposal, superseded for the first pilot: **2.0 GB**, contingent on a new approved
compute-node usage check and at least 2.0 GB remaining safety headroom. Use an
approved scratch/archive location if full branch retention is required. Do not
delete Sprint 1 adapters, snapshots, environments or existing outputs to fit it.

Historical wider-corpus discussion only (first-pilot retains **all** H1/H2): retain hashes/source/config/environment/split manifests and full
small metric/feature/label records, plus each root and five applied successors.
Those 72 unique retained adapters have a parameter-only float32 estimate of
**0.849 GB**. Process one group at a time, sharing immutable base/tokenizer assets
where safe. Unselected candidate/descendant weights could be removed only after
verification and **separate retention approval**, never by this document; that
choice sacrifices exact branch replay without recomputation. Without approval or
external storage, the storage budget remains blocked.

Portable JSON/CSV/evidence allowance proposal: **100 MiB**, included in the 2 GB
incremental cap; actual serialized record sizes are unmeasured. Do not save full
logit arrays/gradients/embeddings as routine evidence; agreed scalar/vector
features and sufficient manifests must remain reproducible. Weights, datasets,
raw sensitive text, credentials, caches and environments stay outside Git;
publish only reviewed/redacted small evidence. No deletion, transfer or upload is
authorized or performed by drafting this protocol.

## Training-Only Generation-Cap Calibration: Prepared, Not Executed

[Configuration](../configs/sprint2_generation_calibration.yaml) and
[CLI](../scripts/calibrate_generation_caps.py) prepare approximately 50 actual
LM-training prompts per task from preassigned **forecaster-train groups**. The
pinned cached Qwen3-4B base (no adapters), NF4 regime, non-thinking chat template,
greedy generation and initial 512-new-token cap are recorded explicitly. This is
not training or H2 labeling. Tasks/group IDs are intentionally unbound; no fake
50-prompt research corpus has been created. Confirm generation-template/model
regime suitability when binding the real tasks; no model sweep is implied.

Bind `training_group_ids` to the actual train split and `tasks` to explicit task
IDs, `training_manifest` paths (relative to config or absolute), and answer rules.
Manifest JSON format (identities below are placeholders, not approved settings):

```json
{"task_id":"ACTUAL_TASK_ID","split":"train","dataset_role":"lm_training",
 "prompts":[{"prompt_id":"ACTUAL_PROMPT_ID","group_id":"ACTUAL_TRAIN_GROUP",
 "source_family":"ACTUAL_FAMILY","prompt":"ACTUAL_TRAINING_PROMPT",
 "reference":"TASK_FORMAT_REFERENCE"}]}
```

Require at least 50 actual records per task, unique prompt IDs, explicit family/
group identities, valid references and no held-out or evaluation-role prompts.
Select 50 by a predeclared ID hash/seed before outcomes; manifest membership and
split authenticity still require human/source auditing. The loader cannot prove
that a falsely tagged dataset is training data.

Exact implemented task adapters, pending actual task bindings:

| Rule | Final-answer stop and correctness |
| --- | --- |
| `final_numeric_line_v1` | One `Final answer: VALUE` line outside `<think>`, VALUE an integer/decimal/fraction with nonzero denominator. Newline or actual EOS terminates; exact rational equality to reference. No scientific notation, comma normalization or text guessing. |
| `final_choice_line_v1` | Same complete-line stopping rule, VALUE exactly uppercase A-D; exact reference equality. |

An unsupported task scorer fails preparation; implement/review its exact native
rule rather than silently treating every task as arithmetic. Unparsed outputs
count wrong; cap-ended incomplete lines count truncated/wrong. EOS-ended wrong
answers are recorded as wrong but not truncated. Length includes emitted EOS.

For every prompt, write length, correctness, stopping reason, truncation flag and
completion digest to `measurements.jsonl`; retain config/manifest/model/rule hashes
and all-prompt rates in `report.json`. No raw prompts/completions, adapters or
weights are exported. An empirical nearest-rank p99 of correct **completed** lengths
is only a proposed fixed cap. Flag all-prompt truncation >2%, any censoring by
512 and insufficient correct completions. With censoring/too few prompts or
correct answers, produce **no percentile/cap estimate**. The 30-correct screening
threshold is a pilot proposal, not reliable p99 evidence; even 50 correct samples
do not establish a reliable population 99th percentile. Also report projected
ALL-prompt truncation under a proposed cap, including long incorrect completions.
Review/fix final task caps before collection; never adapt them using held-out data.
For censored/>2% evidence, rerun the **same preselected training prompts** at an
explicit increased cap with `--recheck-cap 1024` and a fresh output directory;
1024 is an example, not an approved final cap. Reports bind the actual chat-template
hash and effective cap; collector preflight rechecks measurement summaries.

Offline preflight (currently fails clearly until bindings are supplied):

```bash
.venv/bin/python -B -m scripts.calibrate_generation_caps \
  --config configs/sprint2_generation_calibration.yaml
```

Exact next execution command, **only after bindings/review, server-code comparison/
synchronization and a separately authorized existing Slurm CUDA allocation**:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m scripts.calibrate_generation_caps \
  --config configs/sprint2_generation_calibration.yaml \
  --output runs/sprint2_generation_calibration --execute
```

Run from the synchronized project root with the compatible installed server
environment/cache. This command does not submit a job or download a model;
execution rejects the login node, missing allocation/CUDA, multiple devices and
existing output paths. No execution or new server synchronization occurred here.
Calibration compute/storage is **unmeasured** and additional to the earlier cost
scenario; include it in D14 review, not an invented device-hour estimate.

## Remaining Decisions With Mahitha

1. D03 actual task/dataset/scorer IDs, target/OOD/256-anchor/final populations,
   answer-rule bindings, calibration sufficiency and measured final caps.
2. D05 actual adapted-module manifest/count/placement; D06-D09 confidence,
   near-duplicates, NLL/length/verifier/difficulty/embedding and gradient semantics.
3. D11 actual source pools/correction prompt version and real provenance. Pilot
   32-ID H1 draws/shared32-ID H2 excluding union, uniform presealed selection,
   fresh optimizer and explicit empty-batch stop are now accepted configuration.
4. D12 measurement reductions/actual subset IDs; 15/32 updates/cadence/counts are
   accepted. Restoration/masked routing/identity joins are offline implemented.
5. D14 current actual MIG profile/memory/cache/quota and wider-study/search budgets.
   Pilot two-hour/5 GiB/retain-all preparation is approved provisionally, not
   measured fit; historical wider16-hour/2 GB scenarios are not current pilot caps.

Isolated collector/runtime/measurement producers and opt-in `sprint2_d13` are now
implemented with fixture-only verification. No real labels, Qwen training,
downloads, GPU jobs, schema freeze or server modifications occurred. Tiny synthetic
CPU fits/regressions are not real forecasting validation. Sprint1 runtime unchanged.
