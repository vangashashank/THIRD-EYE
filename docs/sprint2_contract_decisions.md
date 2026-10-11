# Sprint 2 Contract Decisions

Review sheet for Shashank and Mahitha, updated 2026-10-11. **D01, D02, D04,
D05, D13 and D15 are accepted based on Shashank's current user instruction.**
The explicit D03 calibration, D10 H1-history and D11 shared-continuation revisions
are recorded below. This records the instruction, not fabricated separate
signatures from Mahitha. The latest user reply accepts the **one-group T=5 pilot
configuration** below, including D11 draws/selection, D12 budgets and provisional
D14 preparation. D06-D09 and D12 feature reductions remain awaiting agreement;
actual D03 datasets/scorers/caps and source/root bindings remain unresolved.
No schema freeze, real label collection, GPU training, jobs, commits or pushes.
Worked examples are synthetic acceptance fixtures, not measured research results.

Sources: [schema](../configs/feature_schema.yaml),
[feature audit](sprint2_feature_audit.md),
[isolated validator](../src/third_eye/feature_contract.py),
[evaluation draft](sprint2_evaluation_protocol.md), and
[Sprint 1 report](sprint1_report.md).

For quick joint review, use the [five remaining formula decisions](sprint2_mahitha_quick_decisions.md).
It compares current code with Mahitha's actual helpers; recommendations are not
approvals. Real bindings/access and launch commands are tracked in the
[handoff readiness update](sprint2_handoff_integration.md).

## Existing Boundaries Versus Recommendations

| Evidence / boundary | What it establishes, not a new agreement |
| --- | --- |
| Shashank's current instruction accepts D01: **T=5 decisions**. | `S_0..S_4`, ending at applied `S_5`; now encoded in isolated helpers. Numerical core scope is distinct from freezing feature semantics. |
| YAML declares K=3, H=2 labels, history length 3, 33 fields and `frozen_on: null`. | Declared layout, not proof of agreed confidence, scoring, LHV, layer order or history semantics. |
| Sprint 1 smoke evidence and validator tests exist. | Infrastructure/structural checks, not research accuracy or approved Sprint 2 definitions. The validator does not produce features or verify ledger truth. |
| Existing user constraints preserve Sprint 1 and prohibit execution here. | Offline collector/producers/evaluation and synthetic tests are implemented; pilot configuration approval is not execution authorization. |

## D01: T=5 and the Four-State Draft

**Question:** Does T=5 count five ranking decisions, and how does the four-state
pilot relate to that core requirement?

**Recommendation and rationale:** Interpret T as applied update/decision count:
five decision states `S_0..S_4`, ending at `S_5`. Treat the previous draft's four decisions
`S_0..S_3` as a smaller feasibility stage, never core completion. This counting
convention is accepted by the current instruction; do not silently redefine T.

**Check:** With 12 groups and K=3, four decisions yield 48 states/144 candidate
rows; five yield 60/180, split 120/30/30 rows for 8/2/2 groups. Under D10,
applied H1 history lengths at decisions 0..4 are 0,1,2,3,3 with timely observations.

**Consequences:** Evaluation draft/counts now use five decisions; wider-corpus budgets remain proposals.
The fifth decision's simulated H=2 endpoint can be `S_6` on its branch; that does
not add a sixth applied core decision. The smaller stage cannot replace T=5.
**Status:** accepted by current user instruction (2026-10-11).

## D02: Lineage Independence and Assignment

**Question:** Which records constitute one independent unit, and when are splits assigned?

**Recommendation and rationale:** Use connected lineage/source-family groups:
shared adapted root, prefix, branch, resume or group-specific correction family
merges groups transitively. Distinct seeds alone do not establish independence.
Assign 8/2/2 groups outcome-blind before collection, using the draft's seeded hash
manifest; all states, candidates and H=2 descendants inherit the group split.
Sharing the immutable base and fixed diagnostic sets is an explicitly conditional
same-base/task study, not a reason to count candidates as independent groups.

**Check:** A shares a prefix with B, B with C: all three get one split. A renamed
checkpoint or seed-only replicate cannot cross splits. Future outcome mutation
must not alter assignment; discovering cross-split ancestry invalidates the manifest.

**Consequences:** Add lineage/data-family component checks and full identity joins;
independent root recipes need disjoint group-specific adaptation families. **8/2/2
supports preliminary evaluation only.** Resample/report uncertainty and paired
differences by lineage group, never candidate row; two held-out groups are too few
for a robust generalization/significance claim, even with 180 rows.
**Status:** accepted by current user instruction (2026-10-11). Pure component/
8/2/2 assignment helpers implemented; actual recipe/source-family manifest pending.

## D03: Dataset Roles and Accuracy Scoring

**Question:** Which examples may be trained on, used for features/labels, or reserved
for final evaluation, and what counts as a correct answer?

**Recommendation and rationale:** Version/hash source-family-disjoint LM adaptation,
target diagnostic, OOD proxy, 256-example retention anchor and final-evaluation
manifests. The same diagnostic manifests may serve forecaster splits, disclosed
as fixed-population evaluation. Score fraction-correct with a pinned task-native
deterministic verifier/final-answer parser and greedy base-model generation.
The earlier fixed 64-token proposal is superseded: calibrate approximately **50
training prompts per task**, initially **512 generated tokens**. Unparseable/unfinished answers count wrong; zero valid reference
examples fail. This is auditable accuracy, not full-text token loss or an LLM judge.

**Check:** Three correct of four is 0.75, regardless of answer length. The anchor
must have 256 unique approved IDs. Any final-evaluation family in updates/probes
or features fails the role audit; an unparseable output is not dropped from accuracy.

**Consequences:** Shashank must supply actual dataset/task/verifier IDs and OOD
boundaries; none are invented here. Separate forecaster group splits from these
LM roles: LM updates in a forecaster test group collect its labels, but its records
cannot fit forecasters/transforms. Diagnostic sets are not untouched final tests.
Use only forecaster-train groups and LM-training prompts for cap calibration.
Record every prompt's generated-token length (including emitted EOS), correctness,
stopping reason and truncation, not just correct answers. An empirical nearest-rank
99th percentile of correct completed lengths proposes a fixed task cap; report
ALL-prompt truncation and flag >2%, insufficient correct answers and any initial-cap
censoring. Censoring/insufficient evidence yields **no cap estimate**. The 30-correct
screening threshold is an explicit pilot proposal, not reliable population-p99
evidence; about 50 prompts cannot establish that reliability. Review/fix caps
before collection, never adapt them using held-out results.

Prepared exact adapters: `final_numeric_line_v1` accepts one `Final answer: VALUE`
line (integer/decimal/nonzero-denominator fraction, exact rational reference equality);
`final_choice_line_v1` accepts one such line with uppercase A-D (exact equality).
Stop only outside `<think>` spans, at newline or actual EOS; an incomplete final
line at the cap is truncated/wrong. No substring/full-text answer guessing. Each
actual task must bind its suitable adapter or have a separately reviewed native
scorer implemented; task identities are not inferred from arithmetic smoke text.

**Status:** measured-calibration procedure revised by current user instruction.
Dataset/task/scorer identities, answer-rule bindings and measured final caps are
**proposed, awaiting agreement/evidence**, not approved here. See the evaluation
draft for manifest format and exact execution command.

## D04: Delta Direction and LHV

**Question:** What endpoints/units/sign define the delta labels, and exactly what is `lhv2`?

**Recommendation and rationale:** Accuracy fractions; delta is
`accuracy(endpoint)-accuracy(parent S_t)`, positive for improvement. H1 uses t+1;
H2 uses the declared t+2 continuation, not t+2 minus t+1. Use
`lhv2 = 0.50*delta_target_h2 + 0.25*delta_ood_h2 + 0.25*delta_retention_h2`.
This transparent target-weighted tradeoff has no hidden hard constraint or learned
weights; it is not evidence that these weights are optimal. Report each channel,
including any harm that a positive aggregate masks.

**Check:** Parent accuracies `(0.60,0.50,0.80)` and H2 `(0.64,0.48,0.79)` give
deltas `(0.04,-0.02,-0.01)` and LHV `0.0125`. A 0.04 fraction change is four
percentage points, not a 4% relative change. Swapping endpoints must reverse sign.

**Consequences:** Add a versioned label/utility builder, not a reinterpretation of
Sprint 1 loss files. Compute LHV identically from predicted and measured H2
channels; keep labels out of inputs. Direct needs three H2 outputs, not an extra
LHV head by default. A retention floor would be a separate revised utility policy.
**Status:** accepted by current user instruction (2026-10-11). Pure fraction/delta/
utility checks implemented; no actual labels generated or legacy losses reinterpreted.

## D05: LoRA Counting and Order

**Question:** Does `num_lora_layers` count blocks, adapted modules or A/B matrices,
and which vector order is canonical?

**Recommendation and rationale:** One entry per adapted module; combine that
module's A/B gradient squared norms before taking its norm. Enumerate numeric
transformer block order, then an explicit placement list (q_proj before
v_proj), retaining qualified module names. A/B and rank do not increase L.
This aligns entries with actual adapted operations and avoids lexical 10-before-2.

**Check:** Two blocks, two adapted projections each: L=4, not 2 or 8. Block 2
precedes block 10; swapping same-shape module entries fails manifest validation.

**Consequences:** Build/verify a pinned model/placement manifest and supply count
and order explicitly to the validator. Do not infer the actual Qwen count here.
Unexpected targets/unmapped modules fail rather than being sorted by guesswork.
**Status:** accepted by current user instruction (2026-10-11). Explicit-manifest
ordering helper implemented; actual model revision/modules/count must be supplied.

## D06: Confidence

**Question:** What do `gen_confidence_mean/std` and candidate `mean_confidence` measure?

**Recommendation and rationale:** For each final-answer span, average its emitted
token log-probabilities under the current parent; exclude prompt, reasoning,
padding and special tokens from the reduction. State confidence is the mean and
population std across answers on a pinned probe set. Candidate confidence is the
mean of answer scores for its corrected-answer batch, teacher-forced at the same
parent. Equal answer weighting avoids length dominance; these are log scores,
not probabilities, calibrated correctness or forecaster uncertainty.

**Check:** Answer log scores -1 and -3 yield mean -2 and population std 1;
singleton std is 0. Changing excluded prompt/padding scores in a fixed token
trace cannot change the reduction (changing the actual prompt can change probabilities).
An empty answer span or empty set fails extraction, not a fake finite confidence.

**Consequences:** Add answer-span/token alignment, probe IDs and generation-policy
metadata; log failures instead of dropping them invisibly. No uncertainty head
implements these fields, and no valid-span failure may be filled with label data.
**Status:** proposed, awaiting agreement.

## D07: Near-Duplicates

**Question:** What constitutes a near-duplicate, and which examples count in the numerator?

**Recommendation and rationale:** Preserve task-sensitive punctuation/case;
normalize Unicode NFC and whitespace, then connect examples whose sets of
whitespace-token 3-grams have Jaccard similarity >=0.90. Shorter texts require
exact normalized equality. Use connected components; rate is extra copies
`(N-number_of_components)/N` for N>0. This inexpensive, versioned lexical rule makes the
denominator explicit without claiming semantic duplicate detection.

**Check:** `[A,A,B]` with distinct B gives 1/3, not 2/3; singleton gives 0.
Whitespace-only variants match, order does not affect components, and changing
numeric answers must not be erased by normalization. Test the 0.90 boundary.

**Consequences:** Implement a pairwise fixture-tested detector and persist version /
component evidence. Source-family grouping for D02 remains mandatory even if a
paraphrase escapes this detector; proposal IDs and adapter hashes are not substitutes.
**Status:** proposed, awaiting agreement.

## D08: Other Candidate Data Statistics

**Question:** Which reductions/populations define length, NLL, verifier success,
difficulty and embedding diversity?

**Recommendation and rationale:** Length = post-truncation nonpadding token count
of serialized prompt+answer (including tokenizer-added specials), population std.
Pre-update NLL = total answer-token NLL divided by total answer-token count at the
parent. This new recommendation matches Mahitha's implemented token-weighted
helper and supersedes our earlier equal-answer recommendation, **not an accepted
decision**; the collector supports either only when explicitly bound. D06
confidence remains equal-answer weighting. Verifier rate = passing attempts / all correction attempts; difficulty
= original failures / unique attempted source examples before correction. Diversity =
mean unordered non-self cosine distance of L2-normalized, mean-pooled non-special
last-hidden-state vectors from the pinned frozen base without adapters. Singleton
diversity is 0; recommend rejecting embedding norms <1e-12 to match her helper
(our current producer rejects exact zero; change only after agreement). Explicit units prevent filtered
success rates and padding width from replacing actual measurements.

**Check:** Lengths 2/4 give mean 3/std 1; two passes of four attempts give 0.5.
Original failures remain unchanged by successful corrections. Orthogonal unit
embeddings give distance 1; identical embeddings give 0.

**Consequences:** Add attempt/source records, masks, tokenizer/base revision and
pooling provenance. These definitions change legacy full-text-loss reductions;
use opt-in producers and assess frozen-base embedding compute before approval.
**Status:** proposed, awaiting agreement.

## D09: Gradient Statistics

**Question:** Which parent gradients/parameters define norm, cosine and sign agreement?

**Recommendation and rationale:** At the same parent in eval mode, separately
differentiate D08's mean answer NLL on candidate and a pinned retention subset;
use the full explicit trainable LoRA A/B manifest, unscaled and before clipping /
optimizer steps. Extract float32 gradients and reduce in float64. Norms are L2;
cosine is dot/product of norms (recommend 0 when either norm <1e-12 to match her
helper, with exact-zero provenance diagnostics; our current configurable boundary
is <= and needs agreement before change); signs use sign(0)=0 over all aligned
elements. Reject missing/nonfinite gradients. This fixes parameter alignment and
prevents optimizer/dropout effects from being mistaken for interference.

**Check:** Candidate `(3,4)` and retention `(3,-4)` give norm 5, cosine -0.28,
sign agreement 1/2. Two zero signs count as agreement; log their prevalence.

**Consequences:** Add no-update gradient extraction, module grouping per D05,
zero-norm/zero-sign diagnostics and restoration tests. Dataset/subset IDs and
precision policy must be recorded; model placement must remain quantization-aware.
**Status:** proposed, awaiting agreement.

## D10: History Availability, Window and Masks

**Question:** Is accepted value predicted or realized, which prior rounds are
eligible, and how are delayed values/padding represented?

**Revised definition:** Use the last `min(t,3)` successfully applied, observed
**t+1/H1 transitions**, oldest-first, left zero padding. Each change is that
transition's endpoint accuracy minus its own parent's accuracy. Accepted value
is `0.50*target_H1 + 0.25*OOD_H1 + 0.25*retention_H1`. It is not `lhv2`, a stored
forecast, or an H2 outcome. Mask 1 means a real observed row, including true zeros.

**Check:** Decisions 0..4 have lengths **0,1,2,3,3**. At S_1, update 0 gives
`[0,0,1]`; at S_4 use updates 1/2/3. All-zero real changes still have mask 1.
A missing applied result fails: never substitute zero padding or skip to an
older row. Wrong identity/parent chain, unsuccessful/simulated records or
observations after the ranking cutoff fail, even if numeric shapes are valid.

**Consequences:** The pure collator/validator require the complete as-of applied
ledger 0..t-1, corpus/group/trajectory/root/evaluation identities, parent and
endpoint SHA-256 values, selected candidate, applied record type, successful flag,
first-available state and logical observation tick/cutoff. They verify supplied
consistency, not independent ledger truth. `applied_h1` is valid for accepted
value; `applied_h2`, simulation and forecast sources are rejected. The isolated
collector records/joins those observations and the integrated Direct/Dynamics
encoders honor masks/lengths. Raw-module layout routes int32 `current_gen_index=t`
to a checked scalar state column, not the recurrent mask. No real dataset/ledger
authenticity is established by these synthetic checks.
**Status:** revised and accepted by current user instruction (2026-10-11).
The prior delayed-H2-history recommendation is superseded, not silently reused.

## D11: H=2 Continuation and Collection Policy

**Question:** Which second update defines an H2 label, and how is the applied lineage advanced?

**Revised definition:** Draw **one standardized continuation keyed by
`(group_id,state)`**, never candidate. Seal the recipe, common generation/correction/
training seeds and source-prompt draw before observing any sibling outcomes.
Share these across all three siblings. Different children may fail on different
prompts and therefore produce different corrected, failure-derived training
batches; their actual corrected texts need not match.

**Check:** Three candidates yield three t+1 branches and three H2 descendants,
not nine searched descendants. Children A/B may correct different subsets of the
same five source prompts; both batches bind the same plan hash/training seed.
Mutating arbitrary outcome values or consuming different amounts of ambient RNG
cannot change the prebuilt shared draw. Never insert a simulated branch into history.

**Consequences:** Pure builder seals immutable recipe JSON, prompt IDs, stream
seeds and plan hash without candidate/outcome arguments. Source draw has its own
local RNG; generation and correction use separate per-prompt identity-keyed seeds,
reset equivalently for siblings; training uses a separate shared seed. Variable
child loops cannot advance the source RNG. The isolated collector honors/logs this
plan. Legacy repeated-batch continuation is not automatically this policy.
Single draws remain noisy; **two draws are optional and unapproved**.

**Latest accepted pilot configuration:** Three H1 draws of **32 unique prompts**
from the same pinned adaptation pool/procedure/generation policy, candidate-specific
seeds keyed `(group,state,candidate)`; overlap is allowed and logged. Generate
answers/corrections at the common parent, train only verifier-passing corrections
of failures. H2 draws **32 unique prompts once**, excluding the **union of all
H1 source IDs**, with shared source/generation/correction/training streams, but
child-specific verified subsets. Insufficient pool/empty verified batch stops,
preserving partial attempts: no redraw, substitute example or fabricated update.
Uniform state-seeded candidate selection is sealed **before features/outcomes**.
Fresh AdamW/no-scheduler state for each branch/probe; inherit weights only.
Actual pool IDs and correction/feature semantics still need binding/review.
**Status:** revised and accepted by current user instruction (2026-10-11) for
one shared draw and this pilot configuration; offline implementation/tested,
no real model execution.

## D12: Probe Budget, Measurements and Restoration

**Question:** What may a temporary probe do/measure, and what state must it restore?

**Accepted pilot budget:** **15 optimizer updates**, measurements at **0/5/10/15**;
H1 and H2 each **32 optimizer updates**. Fixed target32/OOD32, retention256;
anchor-probe32/retention-gradient32; val-probe uses target32. Batch1, accumulation1,
sealed-prompt-order cycling are pinned preparation choices. Reuse supported
lr1e-4, weight decay .01, betas .9/.999, eps1e-8, fresh AdamW/no scheduler.
An optimizer update, not a microbatch/epoch, is one step. All endpoints score the
same fixed populations. Small subsets are plumbing checks, not final conclusions.

**Proposed measurement reductions:** At the immutable parent, temporary update
only; never promote/append history. Fixed candidate/anchor/target population losses
at 0/5/10/15, separately log optimizer loss at updates1..15. Slopes are ordinary
least-squares loss versus optimizer-step index. KL is parent||post-probe on fixed
teacher-forced answer contexts, full-vocabulary KL averaged over valid answer
positions; LoRA delta is the L2 change of explicit A/B parameters, not scaled BA.
These choices separate temporary trajectories from full t+1/t+2 labels.

**Check:** Loss `2-0.1*step` has slope -0.1; identical distributions/weights have
KL/delta 0. On success and injected failure, parent weights/buffers, modes,
gradients, optimizer/scheduler/scaler, RNG, sampler cursor and files/history must
match the pre-probe control, including its next intended training step.

**Consequences:** Dispose in `finally`; isolate or restore all caller RNG touched.
If in-place, deep-snapshot the entire training state, not merely adapter paths.
Supply actual subset IDs and agree numerical/formula tolerances. 15/32 is
substantial overhead, not proof of a cheap probe; never reuse full-outcome losses.
The runtime finally-restores all explicit owned state and verifies a full
fingerprint; success/failure tests also compare the next optimizer step with control.
**Status:** budget/cadence/subset sizes accepted for pilot preparation only;
OLS/KL/answer-span/A-B delta reduction choices **proposed, awaiting agreement**.

## D13: Fitting, Baselines, Ties and Error Reporting

**Question:** What may be fitted on which split, and how are predictions and selections compared?

**Recommendation and rationale:** Train-only observed-value mean/std normalization
with unit scale for zero-variance channels; preserve zero padding and masks.
Keep targets in native fraction units. Fit group-weighted training-mean constant
outputs; compare uniform selection and highest pre-update NLL, plus the forecaster.
Use validation LHV regret for selection, then mean native-unit MSE across the
three H2 channels as tie-breaker;
test once with unchanged training-only fits. Exact score ties randomize uniformly;
enumerate expected regret, avoiding favorable candidate-ID ordering.

**Check:** Realized utilities `(0.03,0.01,0)` and all-tied scores give expected
regret `0.03-(0.04/3)=0.0166667`; uniquely choosing the best gives 0. Mutating
validation/test labels cannot alter training means/transforms. Missing candidates
invalidate primary state regret, not silently a smaller K.

**Consequences:** Implement paired complete-state MAE/MSE/RMSE per channel and
`max(u)-u_selected` regret; macro-average states/trajectories/groups. Report each
group, failures, ties and coverage; cluster uncertainty by D02 groups. No test
tuning, train+validation refit or candidate-row bootstrap. Ranking-only baselines
have no prediction error; constant ranking and uniform selection coincide.
**Status:** accepted by current user instruction (2026-10-11). Opt-in
`sprint2_d13` implements nested group weighting, training-only normalization,
validation regret/native-MSE checkpointing and native errors, selected H2 channel
deltas, all baselines and paired lineage-group uncertainty. Legacy paths unchanged;
real fitting remains blocked on measurements/bindings, not a synthetic-test result.

## D14: Compute, Storage and Retention

**Question:** What budget/retention policy remains feasible after reconciling T=5?

**Revised pilot preparation:** Start **one assigned train group, T=5**, single MIG,
provisional **two-hour / 5 GiB incremental artifact** ceilings. Inspect available
profile/free quota/existing cache before execution; don't assume fit or duplicate
base files. Stop/preserve resumable state at limits; revise estimate for review
if infeasible. No old-output deletion or automatic pruning.

**Check:** Cached-root first group: 30 full fits x32 +15 probes x15 =**1,185
optimizer updates**, 15 records. Thirty float32-only endpoints are ~.330 GiB
under Sprint1's parameter-count scenario; 35 endpoint evaluations require **11,200
generated answers**. Generation/restoration/artifact throughput is unmeasured;
two-hour/5 GiB fit is not established. Wider 12-group estimates remain proposals.

**Consequences:** Current quota/free space, saved dtype, embedding/gradient memory,
generation costs must be checked in a later authorized preflight. Read-only SSH
currently fails authentication; historical headroom/profile are not current.
Retain roots and selected parents, **all pilot H1/H2 adapters**, records/manifests,
source IDs/config/spec/calibration/scoring/ledgers/logs/checksums. Restored temporary
probe states may be disposable. No base duplicates/weight uploads; if storage is
insufficient, stop/report, not delete. See integration report for cost scenario.
**Status:** accepted **provisional pilot preparation/retention**, no launch approval;
actual fit/current quota/capacity and wider-corpus budgets await measured review.

## D15: Ownership and Change Control

**Question:** Who records agreement and versions semantic/layout changes without
mistaking validator success for approval?

**Recommendation and rationale:** Shashank and Mahitha jointly record a separate
dated review disposition for each decision, with exact recipe/data/manifest
references. Shashank owns producers/provenance; Mahitha owns consuming layouts
and masking; both review utility and evaluation. Semantic changes version the
contract/corpus even when shapes remain equal, preventing silent reinterpretation.

**Check:** Changing D04 weights or D10 history endpoints must invalidate semantic
compatibility even if all arrays still pass structural validation. Loader metadata
or a document header cannot supply missing agreement.

**Consequences:** Contract now uses `0.2-sprint2` with schema/source/configured-layout
fingerprints; continuation/calibration plans also carry recipe/manifest/config
hashes. Original 0.1 source hash is retained in the audit and Git history. Preserve
v1 checkpoints/Python smoke schema; `frozen_on` remains null. Future corpus records
must bind all relevant semantic recipes/config hashes, not only equal dimensions.
No additional signatures or execution permission are inferred.
**Status:** accepted by current user instruction (2026-10-11).

## Review Order

Next settle actual D03 datasets/scorers/cap bindings, D06-D09 feature definitions,
D12 measurement reductions, actual D05 parameter manifest and D11 source pools/
correction semantics. Inspect D14 profile/cache/quota and revise infeasible estimates
before execution. Pilot budgets/retention/selection above are accepted, not blanket
semantic or execution approval. Offline collector/fixture implementation
does not approve these remaining choices. T=5 counts replace the four-state draft;
8/2/2 is preliminary and uncertainty remains lineage-group based. No Sprint 1
runtime code, labels, training, downloads or GPU jobs changed/executed.
