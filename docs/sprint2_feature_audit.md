# Sprint 2 Task 1: Forecasting Feature Contract Audit

**Audit updated 2026-10-11: isolated loader/validator, applied-H1 history,
collection/runtime/measurement producers, opt-in D13 evaluation and calibration
preparation implemented and synthetic-tested. No real measurements, jobs or
Sprint 1 integration; remaining definitions/bindings unresolved; not frozen.**

Audited on 2026-10-10 against local `main`, commit
`0a1978cddd17c03a6f2af562a2ff86ee6d1a36d6`. The initial audit began with a clean
working tree; the new isolated implementation was developed on that base.
Publication of this implementation is recorded in Git history.
No repository or parent-directory `AGENTS.md` was found.
The [Sprint 1 report](sprint1_report.md) was read for context; findings below come
from the current source, not from assuming its checklist proves a new contract.

Contract source: the user-supplied `/Users/vangashashankgoud/Downloads/feature_schema.yaml`,
SHA-256 `5092f748af6f854bf52a19b57d943259dbc844555c97e0236128373520602c68`.
It was read and safely parsed, not modified or installed into the pipeline.
The original exact copy was published in Git history. The portable
[configs/feature_schema.yaml](../configs/feature_schema.yaml) is now a deliberate
`0.2-sprint2` semantic revision, **not byte-identical to that attachment**; ordered
field names/dtypes/shapes are preserved and `frozen_on` remains null. The original
attachment's "FROZEN" header did not establish approval. Its comments describe a proposed contract;
they are not execution instructions or proof of agreement with Mahitha.
The repository still uses a separate Python smoke schema. All 33 named YAML
fields, its controls, and derived dimensions are compared below.

## Implemented Isolated Step

Updated [feature_contract.py](../src/third_eye/feature_contract.py) and
[fixture-only tests](../tests/test_feature_contract.py). Existing Sprint 1 source,
checkpoint format, training scripts and tests are unchanged. Only isolated
Sprint 2 entry points import the new validator; no Sprint 1 runtime does.
NumPy and PyYAML were already dependencies.

- `load_feature_contract(path, dimensions=..., layer_order=...)` supports the
  revised `0.2-sprint2` format. It uses a restricted `SafeLoader`, rejects duplicate
  keys, YAML merge keys, unsafe tags, missing/unknown fields and unsupported
  dtypes/shapes/version, and preserves declared group/field order. No descriptions
  or `derived` prose are executed. Source, schema and configured-layout hashes
  expose schema/order/configuration changes without silently approving them.
- Explicit bindings for **both** `num_lora_layers` and `history_len`, plus an
  ordered, unique layer manifest, are required for input validation. Layer names
  never infer a count; a count never infers an order. Configured history length
  must match the YAML and agreed value 3. Missing bindings remain reportable, not guessed.
  Dimensions resolve to state=5, candidate=16+L, and flattened history=5H=15 for
  the declared H=3; this is not a GRU input width or a new architecture.
- `validate_inputs` accepts one unbatched candidate's **named, nested NumPy
  arrays** with exact declared dtypes/shapes and finite values, including padding.
  It does not coerce lists, float64, int64 or masked arrays. Labels are forbidden
  at the top level and inside any input group. `validate_labels` is a separate
  API accepting only the `labels` group; nothing combines labels into inputs.
- D10 `build_applied_history` and validation require a complete as-of applied-H1
  ledger: corpus/group/trajectory/root/evaluation identity, transition IDs,
  parent/endpoint chain, success/applied provenance, first-available state and
  logical observation tick/cutoff. Use the last `min(t,3)` transitions, oldest
  first, left zeros and mask 1 for real rows. Lengths at decisions 0..4 are
  **0,1,2,3,3**. Missing expected observations fail, never pad or skip.
- Every history value, **including `prev_accepted_value`**, accepts only
  `applied_h1`; forecasts, applied H2 and simulated branches are rejected. Verify
  H1 endpoint-minus-own-parent accuracy fractions and their 0.50/0.25/0.25 utility
  against supplied observations; true zero changes remain real rows. Separate
  label validation checks H2 `lhv2` consistency and fraction-delta bounds.
- [sprint2_protocol.py](../src/third_eye/sprint2_protocol.py) adds pure lineage
  components/8-2-2 assignment, explicit numeric-block/module order, one immutable
  shared `(group,state)` continuation draw with separate RNG streams, permitted
  child-specific failure batches, train-only group-macro statistics, native
  prediction errors and exact-tie expected regret. No candidate/outcome keys
  affect shared draws. See [synthetic tests](../tests/test_sprint2_protocol.py).
- [Calibration preparation](../scripts/calibrate_generation_caps.py) defaults to
  offline manifest validation. Explicit cached-base greedy generation at 512
  requires a separate execute flag/existing CUDA allocation. ALL-prompt lengths,
  correctness/stopping/truncation are analyzed without inventing a reliable p99;
  actual task/train-manifest bindings are missing. Config, rules, thresholds and
  next command are in the [evaluation draft](sprint2_evaluation_protocol.md).

**Limits:** results are always `structural_only=True` and contain
`unresolved_definitions`. The validator does not extract any research feature,
check independently that observations are authentic or were recorded before
ranking. Latest-window/completeness/identity checks verify supplied ledger
consistency only. The validator itself does not implement producers, GRU masking,
update probes or restoration; the isolated components below now supply these.
No schema approval/freeze is inferred
from the header or a date. The synthetic fixtures' layer identities, scalar zeros
and caller policies are **not extracted research features or agreed settings**.

Read-only inspection without inventing configuration:

```python
from src.third_eye.feature_contract import load_feature_contract

contract = load_feature_contract("configs/feature_schema.yaml")
print(contract.dimension_summary())  # state=5; candidate/history unresolved
print(*contract.unresolved_definitions(), sep="\n")
```

## Current Isolated Producer Coverage

These functions require real bound sources and still-pending formulas; synthetic
success is **not** end-to-end real collection or approval of those definitions.
All33 schema fields have a proposed/accepted offline producer path, as below;
the original Sprint1 comparison table is retained as a **baseline inventory**, not
a claim these newer functions are absent. Zero-fill/old smoke-loss relabeling is
never a path to completing a missing measurement.

| Fields | Current isolated implementation | Remaining real-data boundary |
| --- | --- | --- |
| target_val_accuracy, ood_proxy_accuracy, retention_anchor_accuracy | `HFCollectionBackend.measure_parent/_score`, `score_completions` in [backend](../src/third_eye/sprint2_backend.py)/[reductions](../src/third_eye/sprint2_measurements.py): all-attempt accuracy/count/parse/truncation. | Actual role/scorer IDs, fixed32/32/256 populations, measured caps. |
| gen_confidence_mean/std | Same parent: actual emitted-token causal logprob alignment, per-answer mean/population std. | D06 formula, confidence population and parse-failure policy pending. |
| token_length_mean/std, pre_update_nll, mean_confidence, verifier_pass_rate, difficulty_proxy, embedding_diversity, duplicate_rate | `prepare_candidate`, `candidate_data`, `answer_logprobs`, `answer_nll`, `duplicate_components`: verified failed-answer corrections only; full attempt denominators, frozen-base embeddings, explicit tolerance/reduction. | D06-D08 definitions/token span/correction prompt/NLL weighting/tolerance; real source pool and verifier. |
| grad_norm, grad_cosine_retention, layerwise_grad_norms, sign_agreement | `gradient_measurements` + exact A/B names/shapes, raw joint module norms/zero diagnostics, no missing-gradient imputation, finally-restored eval extraction. | D09 mode/precision/loss/zero epsilon; actual manifest and32 retention-gradient IDs. |
| train_loss_slope, anchor_loss_slope, kl_drift_anchor, lora_delta_norm, val_loss_slope | `probe_measurements`: 15 updates, population traces0/5/10/15; OLS, parent||post full-vocab answer KL/raw A-B displacement; separate optimizer trace. | Budget/cadence approved, **reductions still D12 proposals**; no H1/H2 feature reuse. |
| prev_accepted_value, current_gen_index, prev_target_change, prev_ood_change, prev_retention_change, history_mask | [collector](../src/third_eye/sprint2_collection.py) + `build_applied_history`: complete applied H1 joins/cutoff,0/1/2/3/3 history, own-parent deltas/utility, real-zero mask. | Real ledger/source authenticity; no H2/branch/history leakage. |
| delta_target_h1, delta_target_h2, delta_ood_h2, delta_retention_h2, lhv2 | `run_sibling` plus `export_candidate`: separate full H1/H2 scores, common-parent fraction deltas, .50/.25/.25 H2 utility. Also records OOD/retention H1 for D10/upstream. | Actual scorer populations, cached roots, shared H2 draw and completed allocated-model runs; not yet executed. |

The [pilot configuration](../configs/sprint2_collection.yaml) now pins three32-ID
candidate draws and one shared32-ID H2 draw excluding their union; uniform selection
sealed before features/outcomes, fresh AdamW, empty-batch stop/preserve, 15/32
updates, diagnostic sizes and provisional two-hour/5 GiB retain-all preparation.
Actual task/scorer/source/lineage/root/cap/module/template bindings are empty.
`D12_measurements` remains a separate agreement gate from the accepted budget.
No real GPU fit, schema freeze, download, commit or push was performed.

[sprint2_runtime.py](../src/third_eye/sprint2_runtime.py) verifies finally-restoration
of tensor state including nonpersistent buffers, gradients/modes, explicit owned
optimizer/scheduler/scaler/sampler and RNGs. Source imports additionally verify
sealed IDs, overlaps, H2 exclusions, per-prompt streams and actual verified
corrections. Hash consistency is not independent proof of measurement truth.
Integrated mask-aware encoders and opt-in [D13](../src/third_eye/sprint2_evaluation.py)
use training-only nested weights/native targets, validation regret/native MSE,
exact ties, baselines and paired lineage-group uncertainty; legacy paths preserved.
See [integration report](sprint2_handoff_integration.md) for verification,
first-group cost scenario, server-authentication blocker and exact commands.

## Original Sprint 1 Findings

1. **The current feature contract does not implement the supplied YAML.** It has
   three parent loss features, four proposal hyperparameters, three applied-history channels,
   and three H=2 loss-reduction outputs. The YAML requests accuracies, generation
   confidence, data/gradient/probe statistics, and masked three-round history.
   No YAML field is implemented end to end under its declared contract. There is
   no YAML loader in the Sprint 1 path or general feature-construction module.
   The new isolated loader validates supplied payloads, not their extraction. See
   [Python schema](../src/third_eye/forecaster_checkpoint.py#L8) and
   [inline tensor construction](../scripts/fit_direct_forecaster.py#L35).
2. **History storage is not end-to-end history forecasting.** The store returns
   prior applied transitions, but the pilot fit never passes them to Direct.
   Direct cannot mask padding or mix empty and nonempty actual histories in a
   batch. See [history slicing](../src/third_eye/history_store.py#L88),
   [fit calls](../scripts/fit_direct_forecaster.py#L46), and
   [GRU consumption](../src/third_eye/direct_forecaster.py#L71).
3. **The pilot does not rank candidates with forecasts before full updates.** It
   generates full t+1/t+2 results, promotes the smallest measured t+1 target loss,
   then fits the forecaster. This verifies an offline pipeline, not the intended
   pre-selection workflow. See [promotion and subsequent fit](../scripts/run_sprint1_pilot.py#L185).
4. **Probe restoration is not a training-state transaction.** The current `probe`
   reads raw next-token logits and changes the model to evaluation mode. Adapter
   rollback only returns a path; the pilot explicitly reloads it into a new model.
   Neither API restores optimizer moments, accumulated gradients, RNG, or data
   position. See [probe](../scripts/run_sprint1_pilot.py#L57),
   [rollback](../src/third_eye/checkpoint_manager.py#L71), and
   [fresh optimizer on every training call](../src/training/trainer.py#L26).
5. **Runtime availability/identity enforcement remains absent.** Existing
   isolated workspaces help, but the fit joins by candidate ID and asserts state
   zero; it does not validate a complete run/trajectory/parent/evaluation identity
   or observation cutoff. The original YAML's ambiguous accepted LHV could leak
   the update being ranked if interpreted as realized H2. Revised D10 excludes
   that interpretation: accepted utility is observed applied H1 only, enforced
   in the isolated validator but not yet the runtime. See
   [current join checks](../scripts/fit_direct_forecaster.py#L21).

## Schema Field Comparison

Named-feature statuses below describe the **pre-collector production/Sprint 1 baseline**,
not standalone structural validation. Adding a validator does not create the
requested measurements. Metadata/control implementation notes include the new
loader separately.

Required definitions reflect the revised YAML and accepted D04/D05/D10 choices;
other missing meanings remain explicit. Shapes are per state/candidate,
before batching. All fields are `float32` except `current_gen_index` (`int32`).
`L = num_lora_layers` counts adapted modules; its actual explicit count/manifest is
not supplied. `H = history_len = 3` below denotes the history
window, not the H=2 forecasting horizon. Ordered YAML lists give field order, but
do not specify order inside the length-`L` vector.

**Implemented** means the exact declared field is produced, validated, and wired
into the applicable model/data path. **Partial** means a relevant primitive or
related recorded quantity exists, but semantics, shape, or wiring are incomplete.
**Missing** means the feature/label calculation does not exist; a related loss,
hash, or proposal ID is not a substitute.

Original baseline coverage: **0 implemented, 13 partial, 20 missing** across the33
fields after the isolated H1 utility/mask and H2 utility helpers. Partial does not
mean that any research metric is actually measured or wired into Sprint 1.

### Current State Performance: Per State

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `target_val_accuracy` | Target-task validation accuracy fraction in [0,1]; `[1]`. Actual task/parser/population unresolved (D03). | Parent `target` is full-text token loss [pilot:146](../scripts/run_sprint1_pilot.py#L146), assembled as loss [fit:36](../scripts/fit_direct_forecaster.py#L36), not accuracy. | Missing | Bind scorer/dataset/measured cap and add accuracy extraction. Three correct of four must return 0.75 regardless of lengths; zero valid reference examples fail, unparsed outputs count wrong. |
| `ood_proxy_accuracy` | Accuracy on a cheap OOD proxy subset; `[1]`. Subset and correctness rule unspecified. | Two arithmetic proxy texts [pilot:37](../scripts/run_sprint1_pilot.py#L37), evaluated only for loss [pilot:65](../scripts/run_sprint1_pilot.py#L65). | Missing | Specify versioned OOD subset and accuracy evaluator. Verify fixed membership/hash, known correct counts, and sharing across all candidates from one parent. |
| `retention_anchor_accuracy` | Accuracy on a fixed 256-example retention anchor; `[1]`. | Current retention set has 2 texts [pilot:38](../scripts/run_sprint1_pilot.py#L38) and measures loss, not accuracy. | Missing | Agree and pin 256 anchor examples, scoring and generation settings. Check count/uniqueness/manifest and fixture accuracy; reject mismatched anchors rather than claiming smoke proxies fulfill the requirement. |
| `gen_confidence_mean` | Mean generation confidence over a probe set; `[1]`; mean token log-probability is an example, not a chosen formula. | `probe` returns one prompt's raw last-token logits [pilot:57](../scripts/run_sprint1_pilot.py#L57); no generation-confidence aggregation or feature input. | Missing | Agree token/sequence aggregation, output-token mask, generation policy and probe set. Verify a fixed log-probability fixture and invariance to excluded prompt/padding tokens under that policy. |
| `gen_confidence_std` | Standard deviation of the same generation confidence; `[1]`; reduction unit and sample/population convention unspecified. | Same raw-logit probe [pilot:61](../scripts/run_sprint1_pilot.py#L61); no standard deviation. | Missing | Use exactly the agreed observations for mean/std. Constant observations must give zero; test known unequal values and the one-observation policy without NaN. |

### Candidate Data Statistics: Per Candidate

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `token_length_mean` | `[1]`; name implies mean token length, but YAML has no description defining text span, tokenizer or truncation. | Dataset tokenizes, truncates and pads [dataset:17](../src/data/dataset.py#L17); proposal records `max_length`, not measured lengths [state:85](../src/third_eye/state_manager.py#L85). | Missing | Agree pre/post-truncation and prompt/answer/special-token policy; compute actual lengths, not padding width. Under the agreed length policy, fixture lengths 2 and 4 must average 3. |
| `token_length_std` | `[1]`; token-length standard deviation convention unspecified. | Token tensors only [dataset:25](../src/data/dataset.py#L25), no statistic. | Missing | Agree population/sample std and singleton behavior. Check known lengths, constant lengths, and singleton input; batching/padding must not change the statistic. |
| `pre_update_nll` | Mean current-parent NLL on candidate batch before update; `[1]`; token versus example weighting and answer masking unspecified. | Token-weighted `evaluate_loss` is reusable [evaluator:7](../src/evaluation/consequence_evaluator.py#L7), but candidates go from dataset construction to training without this evaluation [generator:147](../src/third_eye/candidate_generator.py#L147). | Partial | Evaluate each candidate batch from the same unchanged parent, record before any probe/full update, then build the input. Verify hand-computed unequal-length NLL and that post-update losses cannot supply this field. |
| `mean_confidence` | Mean model confidence in its own corrected answers in the batch; `[1]`. Answer source and confidence formula unspecified. | Candidates receive arbitrary text strings [generator:53](../src/third_eye/candidate_generator.py#L53); no original/corrected-answer structure or confidence extraction. | Missing | Define correction records and answer-only confidence. Test known answer token probabilities, empty answers, and that prompt confidence cannot be substituted. |
| `verifier_pass_rate` | Fraction of attempted corrections passing a deterministic verifier; `[1]`. Verifier and attempt population unspecified. | Fixed arithmetic training strings [pilot:34](../scripts/run_sprint1_pilot.py#L34), not correction attempts or verifier results. | Missing | Record all attempts before filtering, with versioned verifier results. Two passing attempts of four must yield 0.5, not 1.0 after filtering; define zero-attempt behavior. |
| `difficulty_proxy` | Proxy difficulty, e.g. original failure rate; `[1]`. Example does not select a formula. | Proposals store text count/data hash [state:86](../src/third_eye/state_manager.py#L86); no original failures or difficulty. | Missing | Agree proxy and persist pre-correction observations. Test known original failures and ensure later corrected-answer success does not retroactively change difficulty. |
| `embedding_diversity` | Mean pairwise cosine distance of example embeddings; `[1]`. Encoder, pooling, pairs, normalization and singleton policy unspecified. | TextDataset returns only tokens/masks/labels [dataset:33](../src/data/dataset.py#L33); no embedding extraction or pairwise reduction. | Missing | Agree pinned embedding source/pooling and unordered non-self pairs. Identical nonzero embeddings must give zero distance; test orthogonal vectors, singleton, zero vectors, and example-order invariance. |
| `duplicate_rate` | Fraction of near-duplicate examples in batch; `[1]`. Similarity rule, threshold, normalization and denominator unspecified. | Proposal guard rejects repeated identity tuples [state:97](../src/third_eye/state_manager.py#L97); whole-list data hash [state:71](../src/third_eye/state_manager.py#L71) does not compute example duplicate rate. | Missing | Agree exact/near-duplicate rule and whether to count extra copies or every member of duplicate groups. Test `[A,A,B]`, whitespace variants, near duplicates, threshold boundaries and order invariance. |

### Gradient / Interference Statistics: Per Candidate

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `grad_norm` | L2 norm of LoRA gradient on candidate batch; `[1]`. Parameter population, loss reduction, clipping/scaling and mode unspecified. | Trainer calls backward and immediately steps [trainer:59](../src/training/trainer.py#L59); no gradient statistic is recorded. | Missing | Extract at the agreed pre-update parent without applying an update; snapshot/restore gradient and RNG state. Fixture gradient `[3,4]` must have norm 5 under the chosen population. |
| `grad_cosine_retention` | Candidate/retention-anchor gradient cosine; `[1]`. Anchor subset and zero-gradient policy unspecified. | No retention-gradient calculation in trainer [trainer:49](../src/training/trainer.py#L49); evaluator is no-grad [evaluator:22](../src/evaluation/consequence_evaluator.py#L22). | Missing | Compute aligned gradients at the same parent and exact parameter manifest; reuse only matching retention-gradient provenance. Identical/opposite/orthogonal fixtures must yield 1/-1/0; define zero norms explicitly. |
| `layerwise_grad_norms` | `[L]`: joint A/B norm per adapted module, numeric block then explicit placement order (D05). | LoRA placement [lora:15](../src/models/lora_model.py#L15); isolated `canonical_lora_manifest` [helper](../src/third_eye/sprint2_protocol.py) orders supplied metadata, but extracts no gradients. | Missing | Supply actual manifest/count; define D09 extraction. Test blocks 2/10, A/B grouping, exact length and same-shape permutation rejection; never infer count from rank. |
| `sign_agreement` | Fraction of LoRA parameters with agreeing candidate/retention gradient signs; `[1]`. Zero/unused-gradient treatment unspecified. | Only candidate training gradients [trainer:59](../src/training/trainer.py#L59); no aligned comparison. | Missing | Agree denominator and zero-sign policy. Check mixed-sign and zero-valued fixtures with explicit parameter alignment, expected fraction, and no optimizer step. |

### Short-Probe Dynamics: Per Candidate

YAML declares `probe_steps: 15`; comments mention a 10-20 step pilot range. This
audit does not select a different budget or authorize a probe run.

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `train_loss_slope` | Linear-fit training-loss slope over probe steps; `[1]`; step unit, measurement phase and loss weighting unspecified. | Full-update `step_losses` recorded [trainer:82](../src/training/trainer.py#L82); no temporary 15-step runner or slope. | Partial | Add transactional short probe and fit against recorded step coordinates. Synthetic losses `a+b*s` must recover `b`; reject insufficient/nonfinite samples and do not use full-update logs as probes. |
| `anchor_loss_slope` | Linear-fit retention-anchor loss slope during probe; `[1]`; evaluation cadence/subset unspecified. | Pilot evaluates full adapters after completed updates [pilot:163](../scripts/run_sprint1_pilot.py#L163), not anchor loss during a probe. | Missing | Agree fixed anchor sample/cadence and use matching token/answer policy at all probe points. Check known linear trace and unchanged anchor manifest; evaluation must not consume training gradients or alter modes/RNG. |
| `kl_drift_anchor` | Pre/post-probe anchor output-distribution KL; `[1]`. KL direction, token alignment/reduction and conditioning unspecified. | Raw one-prompt logits compare numerical reload equality [pilot:80](../scripts/run_sprint1_pilot.py#L80); no anchor-distribution KL. | Missing | Agree KL direction, fixed teacher-forced/generated context, valid-token mask and vocabulary reduction. Same distributions must yield zero; asymmetric toy distributions must distinguish directions; padding must not affect the result. |
| `lora_delta_norm` | L2 norm of LoRA weight change after probe; `[1]`; A/B parameter change versus effective scaled `B@A` change unspecified. | Adapter hash compares identity [checkpoint:12](../src/training/checkpoint.py#L12), not numerical displacement; no temporary probe delta. | Missing | Agree weight representation/population/scaling and compute from immutable pre-probe snapshot. Zero update must yield zero; a known parameter delta `[3,4]` yields 5 only if that parameter representation is chosen. |
| `val_loss_slope` | Linear-fit small target-validation loss slope during probe; `[1]`; subset/cadence/step coordinates unspecified. | Full t+1/t+2 target evaluations [consequences:138](../src/third_eye/consequence_generator.py#L138), not intermediate probe observations. | Missing | Agree probe-only validation manifest and schedule, separate from final held-out evaluation. Check known trace, exact sample identity and that full t+1/t+2 labels cannot populate this field. |

### Recursive History: Per State

Revised D10 requires last min(t,3) applied observed H1 transitions, oldest-first,
left padding and real masks. The legacy store remains separate; the new pure
collator requires explicit applied ledger evidence, not legacy loss reinterpretation.

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `prev_accepted_value` | `[3]`: .50 target + .25 OOD + .25 retention on each applied H1 endpoint-minus-own-parent accuracy delta; never H2/forecast. | `build_applied_history` computes and validates from caller ledger [contract](../src/third_eye/feature_contract.py); no runtime producer/wiring. | Partial | Bind authentic applied records/evaluation identities. Test applied_h1 acceptance, true zeros, missing observations and rejection of future/simulated/H2/forecast values. |
| `current_gen_index` | Zero-indexed current generation `t`; `int32 [1]`. | `state_id` is stored/validated [state:42](../src/third_eye/state_manager.py#L42); not supplied to Direct [fit:36](../scripts/fit_direct_forecaster.py#L36). | Partial | Define correspondence with recursive generation and embedding/normalization route. Test t=0, positive t, range/dtype validation and that candidate ID is not substituted for t. |
| `prev_target_change` | `[3]`: observed applied H1 target accuracy minus that transition's parent accuracy. | Isolated collator/validator [contract](../src/third_eye/feature_contract.py); legacy [history](../src/third_eye/history_store.py#L77) still holds loss reductions, fit ignores history. | Partial | Join real applied records; test last-three boundary, decisions 0..4, sign, true zeros, missing results and branch exclusion. |
| `prev_ood_change` | `[3]`: applied H1 OOD endpoint-minus-own-parent accuracy fraction. | Isolated collator/validator [contract](../src/third_eye/feature_contract.py); legacy [history](../src/third_eye/history_store.py#L6) remains loss-based. | Partial | Bind actual OOD manifest/scorer; distinct fixture channels must not swap; reject future observations or another trajectory. |
| `prev_retention_change` | `[3]`: applied H1 retention endpoint-minus-own-parent accuracy fraction. | Isolated collator/validator [contract](../src/third_eye/feature_contract.py); legacy [history](../src/third_eye/history_store.py#L77) remains loss-based. | Partial | Bind actual anchor/scorer; test real zero versus padding, retained boundary and same endpoint/evaluation identity across channels. |
| `history_mask` | `[3]`: 1 for required observed applied-H1 row, 0 only for left pre-trajectory padding. | Pure collator/validation [contract](../src/third_eye/feature_contract.py); legacy [Direct](../src/third_eye/direct_forecaster.py#L85) still processes all rows. | Partial | Add opt-in length-aware Direct v2; existing fixture checks lengths 0,1,2,3,3 and missing-result errors. Future forward tests need padding invariance and mixed-batch equivalence. |

### Labels: Per Candidate, Offline Supervision Only

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `delta_target_h1` | `[1]`: H1 target accuracy fraction minus current parent accuracy; label only. | Raw losses [pilot:167](../scripts/run_sprint1_pilot.py#L167); separate pure delta/range validation [contract](../src/third_eye/feature_contract.py), no label collection. | Partial | Bind task scorer/population before collection; test distinct H1/H2 endpoints and input separation. Loss cannot recover missing accuracy. |
| `delta_target_h2` | `[1]`: standardized t+2 target accuracy minus current parent accuracy, not t+2 minus t+1. | Legacy loss label [pilot:170](../scripts/run_sprint1_pilot.py#L170); pure shared plan [helper](../src/third_eye/sprint2_protocol.py) and delta validation, no accuracy collection. | Partial | Bind D03/D11 recipe/scorer and isolated collector; test parent-to-H2 sign and outcome-blind plan/input invariance. |
| `delta_ood_h2` | `[1]`: standardized H2 OOD accuracy minus current parent accuracy fraction. | Legacy loss label [pilot:170](../scripts/run_sprint1_pilot.py#L170); isolated delta validation, no accuracy collection. | Partial | Bind fixed OOD manifest; test improvement/worsening and reject mismatched populations/units. |
| `delta_retention_h2` | `[1]`: standardized H2 retention accuracy minus current parent accuracy fraction. | Legacy loss label [pilot:170](../scripts/run_sprint1_pilot.py#L170); isolated delta validation, no accuracy collection. | Partial | Bind fixed anchor manifest/scorer; test endpoint/sign, finiteness and label-only availability. |
| `lhv2` | `[1]`: .50 target_H2 + .25 OOD_H2 + .25 retention_H2 accuracy deltas, higher is better (D04). | Pure `weighted_utility`/separate H2 consistency validation [contract](../src/third_eye/feature_contract.py); no real label producer. | Partial | Bind H2 endpoint/evaluation provenance; example (.04,-.02,-.01) gives .0125. Apply same rule to predicted deltas; never rank from realized labels or substitute H1 accepted utility. |

### Metadata, Controls and Derived Dimensions

| Schema entry | Declared requirement / shape | Current implementation | Status | Required change / validation |
| --- | --- | --- | --- | --- |
| `schema_version` | String `0.2-sprint2`, intentional semantic revision of 0.1. | Isolated loader/version/fingerprints [contract](../src/third_eye/feature_contract.py). Existing checkpoint version 1 remains separate. | Implemented | Compare semantic/layout/recipe hashes at future joins; no legacy migration or shape-only compatibility. |
| `frozen_on` | Null; no freeze authorized. | Loader preserves metadata and lists selected/remaining decisions [contract](../src/third_eye/feature_contract.py). | Implemented | Never infer blanket approval from successful validation or a date. |
| `current_state_performance.scope` | `per_state`, shared across K=3. | Three repeated parent-loss rows [fit:36](../scripts/fit_direct_forecaster.py#L36), not the five declared fields. | Partial | Share one validated parent observation per state; all three candidates must receive identical state features. |
| `candidate_data_stats.scope` | `per_candidate`. | Proposal hyperparameters only [fit:37](../scripts/fit_direct_forecaster.py#L37). | Missing | Extract each candidate's declared batch statistics; require complete candidate identity. |
| `gradient_interference_stats.scope` | `per_candidate`. | No extracted gradient feature inputs [fit:37](../scripts/fit_direct_forecaster.py#L37). | Missing | Bind gradient extraction to exact candidate/parent/parameter manifest; reject stale cache. |
| `short_probe_dynamics.scope` | `per_candidate`. | Full candidate training [generator:155](../src/third_eye/candidate_generator.py#L155), no temporary feature probes. | Missing | Separate probe records from full outcomes; validate candidate and budget. |
| `short_probe_dynamics.probe_steps` | Integer 15; comments mention 10-20 pilot range. | Training is configured in epochs [trainer:12](../src/training/trainer.py#L12), not a bounded 15-step probe. | Missing | Agree what counts as a step and exact stopping/measurement schedule. Fixture trace length and actual optimizer steps must match recorded budget. |
| `recursive_history.scope` | `per_state`, shared across candidates. | One file per trajectory [history:10](../src/third_eye/history_store.py#L10), no fit wiring. | Partial | Produce one aligned history window per state; candidate ordering must not change it. |
| `recursive_history.history_len` | Integer 3. | Pure last-three H1 collator [contract](../src/third_eye/feature_contract.py); legacy [store](../src/third_eye/history_store.py#L98) unchanged. | Partial | Check required lengths and complete provenance; opt-in model integration still absent. |
| `labels.scope` | `per_candidate`, expensive labeling only; unavailable at ranking. | H2 loss-label rows [pilot:167](../scripts/run_sprint1_pilot.py#L167), not complete YAML labels. | Partial | Keep labels in separate objects/loader and validate run/trajectory/state/candidate identity. Future-label mutation must leave inputs unchanged. |
| `derived.static_candidate_feature_dim` | `8 + (3 + L) + 5 = 16 + L`. Prose, not executable. | Standalone `dimension_summary` [contract](../src/third_eye/feature_contract.py); unresolved without explicit L. Sprint 1 width remains 4. | Implemented | Actual L/manifest still needed; no new model wiring or execution of prose. |
| `derived.static_state_feature_dim` | `5`. Does not include separately routed index. | Standalone summary [contract](../src/third_eye/feature_contract.py); Sprint 1 width remains 3. | Implemented | Index routing/integration remain undecided. |
| `derived.history_feature_dim` | Five length-H arrays including mask, excluding index: `5H = 15`. | Standalone summary [contract](../src/third_eye/feature_contract.py); existing GRU step width remains 3. | Implemented | Serialized width is not recurrent width. Direct v2 mask routing remains absent. |

The five label fields do **not** require a five-output Direct head: the YAML is
shared by H1, Direct and Dynamics; H1 target delta is a separate baseline target,
and `lhv2` is described as a derived ranking/reporting aggregate. Direct's three
H2 outputs match the number of H2 delta channels, but their current loss semantics
do not establish the proposed score contract. Any different training-target plan
needs agreement, not a silent architecture change.

### Current Python Smoke Contract: Separate Legacy Inventory

These rows describe the existing implementation, **not agreed Sprint 2
requirements**. Status refers only to the stated legacy definition. `B` is batch
size and `T` is the number of actual prior applied transitions. The fit constructs
float32 state/candidate/target tensors. History storage returns Python lists; it
does not define a tensor dtype or collator.

| Current field | Legacy definition and shape | Current implementation / reference | Status | Required change or preservation; meaningful validation |
| --- | --- | --- | --- | --- |
| `state.parent_target_loss` | Scalar per row; column 0 of `[B,3]`; parent target mean valid-token loss. | Parent evaluated before candidates [pilot:146](../scripts/run_sprint1_pilot.py#L146); selected by metric order [fit:36](../scripts/fit_direct_forecaster.py#L36). | Implemented | Preserve this legacy meaning; confirm YAML's metric/unit/population. Reject mismatched parent identity. A fixture with distinct losses must select target, not OOD. |
| `state.parent_ood_loss` | Scalar per row; column 1 of `[B,3]`; parent OOD mean valid-token loss. | Same construction [fit:36](../scripts/fit_direct_forecaster.py#L36); current proxy examples [pilot:37](../scripts/run_sprint1_pilot.py#L37). | Implemented | Define research OOD set separately from smoke proxy; require evaluation manifest agreement. Perturb only OOD and verify only its input column changes. |
| `state.parent_retention_loss` | Scalar per row; column 2 of `[B,3]`; parent retention mean valid-token loss. | Same construction [fit:36](../scripts/fit_direct_forecaster.py#L36); current proxy examples [pilot:38](../scripts/run_sprint1_pilot.py#L38). | Implemented | Confirm retention population, normalization, and tokenizer/truncation policy. Verify token-weighted arithmetic with unequal valid-token counts. |
| `candidate.learning_rate_div_1e-4` | Scalar per row; column 0 of `[B,4]`; proposed learning rate divided by `1e-4`. | Proposal recorded before training [generator:96](../src/third_eye/candidate_generator.py#L96); scale [fit:38](../scripts/fit_direct_forecaster.py#L38). | Implemented | Preserve legacy scale; record Sprint 2 scale in the contract. `2e-4` must map to `2`; reject zero, negative, NaN/Inf. |
| `candidate.epochs` | Scalar per row; column 1 of `[B,4]`; positive integer proposal cast to float32. | Validated [state manager:50](../src/third_eye/state_manager.py#L50); assembled [fit:38](../scripts/fit_direct_forecaster.py#L38). | Implemented | Confirm whether new contract wants epochs or bounded step/token budget. Reject fractional/zero epochs rather than silently rounding. |
| `candidate.batch_size` | Scalar per row; column 2 of `[B,4]`; positive integer proposal cast to float32. | Validated [state manager:50](../src/third_eye/state_manager.py#L50); assembled [fit:38](../scripts/fit_direct_forecaster.py#L38). | Implemented | Define microbatch versus effective batch if accumulation is introduced. Distinct batches must affect only the designated column; reject booleans as integers. |
| `candidate.max_length_div_128` | Scalar per row; column 3 of `[B,4]`; proposed maximum token length divided by 128. | Validated [state manager:50](../src/third_eye/state_manager.py#L50); scale [fit:38](../scripts/fit_direct_forecaster.py#L38). | Implemented | Preserve legacy scale; agree actual length versus truncation limit. A proposal of 256 must map to 2; tokenizer/evaluation manifests must match their declared policy. |
| `history.applied_target_loss_reduction` | Scalar per transition; column 0 of `[T,3]` list; parent loss minus actually applied successor loss. | Stored [history:77](../src/third_eye/history_store.py#L77), sliced [history:98](../src/third_eye/history_store.py#L98); not consumed by current fit. | Partial | Add opt-in collator/length handling and validated trajectory reads. Parent 5, applied loss 3 must yield +2; unselected/simulated outcomes must not enter history. |
| `history.applied_ood_loss_reduction` | Scalar per transition; column 1 of `[T,3]`, same applied-transition convention. | Same storage [history:77](../src/third_eye/history_store.py#L77); metric order [history:6](../src/third_eye/history_store.py#L6). | Partial | Bind records to run, trajectory, parent, evaluation, and observation cutoff. A future OOD observation must be excluded even if already present in an offline file. |
| `history.applied_retention_loss_reduction` | Scalar per transition; column 2 of `[T,3]`, same applied-transition convention. | Same storage and slicing [history:88](../src/third_eye/history_store.py#L88). | Partial | Validate persisted order/values, not only append calls. Zero improvement must remain a valid observation and differ from absent history. |
| `labels.target` | Scalar per candidate; column 0 of `[B,3]`; parent target loss minus full simulated t+2 target loss. | Label arithmetic [pilot:170](../scripts/run_sprint1_pilot.py#L170), target tensor [fit:41](../scripts/fit_direct_forecaster.py#L41). | Implemented | Keep supervision separate from inputs. Numeric fixture must distinguish parent-to-t+2 from t+1-to-t+2 and check improvement sign. |
| `labels.ood` | Scalar per candidate; column 1 of `[B,3]`; parent OOD loss minus full simulated t+2 OOD loss. | Same arithmetic [pilot:170](../scripts/run_sprint1_pilot.py#L170), order [schema:13](../src/third_eye/forecaster_checkpoint.py#L13). | Implemented | Confirm whether proposed label is loss reduction or another score. Changing offline OOD labels must not change pre-selection input construction. |
| `labels.retention` | Scalar per candidate; column 2 of `[B,3]`; parent retention loss minus full simulated t+2 retention loss. | Same arithmetic [pilot:170](../scripts/run_sprint1_pilot.py#L170). | Implemented | Agree retention target/direction; reject label/evaluation manifest mismatches. Check worsening gives a negative reduction. |
| `horizon` | Contract metadata integer `2`, not an input feature; pilot records full t+1 and t+2. | [Schema:14](../src/third_eye/forecaster_checkpoint.py#L14), [fit:33](../scripts/fit_direct_forecaster.py#L33). | Implemented | Define an update unit and continuation policy before generalizing. Reject other horizons under v1 rather than relabeling them. |

For these metrics, [evaluate_loss:7](../src/evaluation/consequence_evaluator.py#L7)
computes token-weighted loss, excluding padded and first causal labels, not answer
accuracy. [TextDataset:28](../src/data/dataset.py#L28) masks token padding; **that is
not a forecaster history mask**. No research accuracy or calibrated confidence is
established by these features.

### Semantic Mismatches That Shape Validation Cannot Resolve

- LoRA rank `r` is not `num_lora_layers`. With two blocks and two adapted
  projections per block, the possible counts are 2 blocks, 4 modules, or 8 A/B
  matrices. This is a synthetic counting example, not a Qwen layer-count claim.
  D05 now accepts module counting, joint A/B norms and numeric block/explicit
  placement order. Actual architecture/placement/qualified names still need a
  supplied manifest. Lexical parameter hashing is unrelated.
- YAML confidence is **base-model generation/corrected-answer confidence**, not
  forecaster uncertainty. The three scalar confidence fields still need formulas:
  log-probability versus probability, token/answer weighting, prompt/output masks,
  generation settings, and the standard-deviation convention. A new forecaster
  confidence head would not implement these fields.
- `duplicate_rate` concerns near-duplicate **examples**, not proposal IDs or
  adapter equality. Even the denominator is ambiguous: for `[A,A,B]`, counting
  extra copies gives 1/3 whereas counting all group members gives 2/3. Neither
  choice is asserted here as agreed. Define normalization and similarity threshold.
- Parent-to-t+2 loss reduction is `loss(parent)-loss(t+2)`; scalar continuation
  reduction is `loss(t+1)-loss(t+2)`; regret is `loss(t+2)-best_loss(t+2)`.
  These are different existing quantities. If the new score is accuracy, an
  D04 now accepts `accuracy(endpoint)-accuracy(parent)` in fraction units for
  H1/H2, positive for improvement; H2 remains parent-to-H2, not H1-to-H2.
  Existing loss files cannot reconstruct unrecorded answer accuracy.
- Original YAML lacked an LHV formula; revised D04 accepts the fixed
  0.50/0.25/0.25 weighted three H2 fraction deltas, with no hidden retention floor.
  Task scoring/populations still need binding. Rank/regret/loss reduction are not LHV.
- Revised D10 resolves history padding/order/source: latest min(t,3) applied-H1
  observations, left zeros/oldest-first. Missing applied results fail. It does not
  implement masked recurrent encoding or authenticate a caller-provided ledger.

## History and Input Validation Gaps

- `features_before(s)` slices the first `s` records. Append-time checks require
  consecutive transitions and matching before/after losses, but `_read` trusts
  existing JSON without validating trajectory identity, ordering, finiteness, or
  evidence that a transition was really applied. See
  [history read](../src/third_eye/history_store.py#L16).
- Direct validates dimensions, not finiteness, units, or allowed missingness.
  Fit has finite checks, but these and identity checks are `assert`s and disappear
  under `python -O`. See [Direct:54](../src/third_eye/direct_forecaster.py#L54) and
  [fit:24](../scripts/fit_direct_forecaster.py#L24). New contract validation should
  raise explicit errors and run before forward/training.
- A CPU, forward-only synthetic diagnostic with seed 42 and `ThirdEyeDirect(3,4,3)`
  compared state `[[2,3,4]]`, candidate `[[1,1,1,1]]`, and history
  `[[[0.2,0.1,0.3]]]` with one trailing zero row. Maximum absolute prediction change
  was `0.004708603024482727`. `None` versus one all-zero history row differed by
  `0.007074028253555298`. A NaN state was accepted and produced NaN output.
  These are **synthetic counterexamples, not forecasting measurements**; no
  backward call, optimizer step, training, or model download was used. They show
  why zero padding and missing-value substitution are not safe defaults.
- Proposal metadata includes seed and planned optimizer steps, but neither is in
  the four-column input. Different stochastic candidates can therefore have equal
  input features. Decide which proposal/probe attributes genuinely distinguish
  candidates; do not add arbitrary candidate IDs as explanatory measurements.
- Checkpoint v1 embeds the Python name lists and checks equality, but save/load
  dimensions are fixed to `(3,4,3)` and the output width to 3. See
  [checkpoint:25](../src/third_eye/forecaster_checkpoint.py#L25) and
  [Direct:38](../src/third_eye/direct_forecaster.py#L38). Expanded inputs require an
  opt-in versioned layout with units/order/masks, not silent reuse of old weights.

## Is `prev_accepted_value` Available at Ranking Time?

Let the cutoff be the logical decision time immediately before ranking the current
candidates, after only the explicitly permitted pre-selection probes. Availability
means the value was actually observable by that cutoff in the deployment protocol,
not merely that a hindsight JSON file contains it.

| Interpretation | Availability under revised D10 | Required guard |
| --- | --- | --- |
| Forecast produced at previous acceptance | May have been available, but **not permitted as D10 history**. | Reject forecast source rather than relabel it as an applied observation. |
| Previous accepted update's observed applied H1 accuracy utility | Yes after completion/evaluation at the ranking cutoff; this is the revised definition. | Complete ledger, parent/endpoint/evaluation identities and cutoff; missing results fail, real zero is mask 1. |
| Previous candidate's realized H2 value while ranking the next update | Generally future; **never D10 history**, even if later available. | H2 stays separate supervision; offline simulation does not become an applied transition. |
| Value of the candidate that will be accepted at the current decision | No; selection has not happened. | Do not fill this using the current winner's full outcome. |

In state notation, while ranking `U_t` at `S_t`, the previous accepted update
`U_(t-1)` has reached its t+1 endpoint `S_t`; its t+2 endpoint `S_(t+1)` generally
depends on the not-yet-selected `U_t`. If the original YAML's accepted "LHV" meant ground-truth
`lhv2`, reading that value from the most recent round therefore leaks the current
decision's future. An offline simulated branch label does not make it observable
under a cheap pre-selection deployment protocol.

The revised definition explicitly chooses **H1 accuracy utility**, not accepted
H2 LHV. All four history value channels are available from the same applied H1
transition, so no forecast bootstrap or delayed-H2 row policy is needed. Existing
names/shapes are preserved but semantic version changes. At decision 0 there is
no previous candidate; at decisions 1..4 the expected rows are 1,2,3,3. A future
observation, missing result or simulated sibling cannot stand in as a real row.
The builder takes no label file; separate label validation cannot backfill history.
Caller-supplied identity/observation evidence still needs a real trusted producer.

## Temporary Probes Versus Full Labels

| Data product | Availability / role | Current implementation |
| --- | --- | --- |
| Parent evaluation and static proposal | Before a candidate update; legitimate inputs under the legacy contract. | Parent [pilot:149](../scripts/run_sprint1_pilot.py#L149); proposal [generator:96](../src/third_eye/candidate_generator.py#L96). |
| Existing `probe()` | Raw one-prompt next-token logits used for save/reload equivalence, not update-probe features or confidence. | [pilot:57](../scripts/run_sprint1_pilot.py#L57); no temporary training step occurs inside it. |
| Isolated bounded pre-selection update probe | Latest pilot approves15 optimizer updates and0/5/10/15 cadence; reductions remain pending. | `probe_measurements`/`restored_training_state`, separately marked temporary, no parent promotion/history advance. Real bound-model execution pending. |
| Full candidate t+1 and simulated continuation t+2 | Offline supervision for predicting consequences of the full update, not inputs available under a cheap-probe ranking protocol. | Full candidate training [generator:155](../src/third_eye/candidate_generator.py#L155); continuation [consequences:158](../src/third_eye/consequence_generator.py#L158); labels [pilot:167](../scripts/run_sprint1_pilot.py#L167). |
| Actual selected/applied transition | Eligible history for a later decision only after observed. | [pilot:191](../scripts/run_sprint1_pilot.py#L191); simulated t+2 is excluded. |

A probe feature must not be populated by renaming a full candidate training loss,
full t+1 score, full t+2 score, rank, regret, or target label. A gradient-only probe
and a temporary update probe are different budgets and require separate definitions.
Record parent digest, run/trajectory/state/candidate IDs, seed, steps/tokens,
data/evaluation manifests, extractor version, and observation cutoff. Use only an
agreed probe population; reserve independent held-out data for research evaluation.
Gradient/interference features must be measured at the agreed parent state,
with a pinned parameter manifest and loss reduction. Agree dropout/mode, RNG,
pre/post-clipping and AMP unscaling, retention subset and zero-gradient handling.
For slopes, agree optimizer-step coordinates, measurement phase/cadence and
whether an initial step-zero evaluation is included. For KL, agree direction,
conditioning and token/vocabulary reduction; for adapter displacement, agree A/B
parameter deltas versus the effective scaled adapter matrix. Neither 15 temporary
steps nor their trajectories can silently replace full-update t+1/t+2 labels.

### Parent and Training-State Restoration Contract

**Restoration requirements; isolated implementation now tested:** prefer each probe
in an isolated working model/process loaded from the same immutable parent. Keep
the main model and training state untouched; clone optimizer state into the probe
only if the agreed update semantics require continuation. Dispose of the probe in
`finally`. If it runs in the same process, isolate or save/restore every RNG it
touches. Merely reseeding changes the caller's state rather than restoring it.

If in-place probing is necessary, capture independent snapshots before any probe:

- Trainable parameters and any mutable buffers, active adapter/scaling/config,
  `requires_grad` flags, per-module train/eval modes, and cache/checkpointing flags.
- Optimizer parameter groups and moments/master weights; scheduler and AMP scaler
  if used; accumulated gradients and step/accumulation counters. Restore optimizer
  state onto the same parameter identities.
- Python and Torch CPU/device RNG states, plus NumPy or other generators if used;
  sampler/DataLoader generator, shuffle position, and continuation data cursor.
- Relevant bookkeeping such as training metrics and pending history/output writes.

Restore in `try/finally` on success, exceptions, and supported cancellation. Probes
must not write parent checkpoints, promote adapters, or record applied transitions.
Adapter `save_pretrained` is not a snapshot of this entire state. Current
`train_lora` owns a fresh local AdamW and returns only the model, so continuation
versus reset must be agreed before a new probe runner is designed. `evaluate_loss`
restores the top-level mode in `finally` [evaluator:45](../src/evaluation/consequence_evaluator.py#L45),
but that does not restore optimizer/RNG state or arbitrary mixed submodule modes.
The current raw-logit `probe` does not even restore its prior top-level mode.

Validation for a later, separately authorized implementation: compare parent
parameters/buffers, modes, gradients, optimizer state, RNG, and history/file state
before/after; verify deterministic logits and that the next intended training step
matches a no-probe control. Inject a failure mid-probe and reverse candidate probe
order. Use agreed numerical tolerances and quantization-aware loading; do not
silently apply ordinary `.to()` to NF4 models. **No such update probe was run for
this audit.**

## Ordered Implementation Plan

1. **Completed offline:** partial semantic revision 0.2, strict loader/validator,
   supplied-ledger H1 collator/utility checks and fixture coverage. Sprint 1 is untouched.
2. **Completed offline:** pure lineage/split/manifest-order, shared continuation
   plan, train-only statistics and native error/exact-tie regret helpers. No collection.
3. **Prepared, blocked on bindings:** generation-cap calibration. Supply actual
   tasks, training groups/manifests and suitable exact answer rules; inspect cached
   server code/environment before separately authorized execution. Review/fix caps
   from training-only measurements before collection, not from held-out outcomes.
4. **Implemented offline, real bindings pending:** isolated feature/label producers,
   collection journals/resume/provenance and candidate/H2 source planning. Agree
   D06-D09/D12 reductions; bind actual D03/D05/D11 sources/roots/configurations.
5. **Implemented offline:** mask/length/raw-module index routing and hashed
   checkpoints/opt-in D13 fit/metrics; mixed-length padding tested. Preserve v1.
6. **Implemented offline:** transactional probes with success/failure/next-step
   restoration tests. Probe features never reuse full H1/H2 labels. Inspect
   D14 actual MIG/cache/quota; provisional limits are not verified fit or launch permission.
7. Only after remaining agreement/preflight and separate authorization, collect
   the grouped T=5 corpus and evaluate unchanged training-only fits. Neither
   synthetic tests nor single-state smoke MSE establish forecasting validity.

## Decisions Requiring Agreement With Mahitha

D01/D02/D04/D05/D13/D15 are accepted by user instruction, alongside the explicit
D03 calibration/D10 H1/D11 shared-draw revisions. No blanket approval or separate
Mahitha signature is inferred; `frozen_on` remains null. Still needed:

1. D03 actual datasets/tasks/scorers, target/OOD/256-anchor/final populations,
   answer-rule bindings and measured cap review; no 50-prompt research manifests exist here.
2. D05 actual adapted-module manifest/count/placement and pinned model revision.
3. D06-D09 confidence/std, near-duplicates, length/NLL/token masks, verifier/
   difficulty/embeddings, gradients/zero cases/scaling/reduction.
4. D11 actual continuation/source-pool/correction bindings and real provenance.
   One32-ID H2 draw excluding H1 union, uniform presealed selection, fresh optimizer
   and stop-on-empty are accepted pilot settings, not unresolved choices.
5. D12 slope/KL/adapter/answer-span reductions and actual subset IDs;
   15/32 updates/cadence/sizes are approved. Masks/restoration are offline tested.
6. D14 current capacity/cache/quota and wider-study/CPU-search budgets. First-pilot
   two-hour/5 GiB/retain-all preparation is provisional approved, not measured fit.

## Prior Helper Verification

Read and safely parsed the complete user-supplied YAML, recorded its SHA-256, and
inventoried 33 named fields in six groups (32 float32, one int32). Read current
schema/checkpoint, Direct, fit, pilot, proposal/candidate/consequence
construction, label preparation, history, LoRA, training, evaluation, and existing
tests; inspected the Sprint 1 report and repository state. The initial audit's
forward-only synthetic GRU diagnostic is retained above, not a research result.
Field table coverage/order and links are checked. Pure H1 history/utility builders
use supplied synthetic observations only, not extracted research features.

Prior helper-step checks (current complete-suite results are in integration report):

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_feature_contract.py -v
.venv/bin/python -B -O -m unittest discover -s tests -p test_feature_contract.py
.venv/bin/python -B -m unittest discover -s tests -p test_sprint2_protocol.py
.venv/bin/python -B -m unittest discover -s tests -p test_generation_calibration.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=tests .venv/bin/python -B -O -m unittest \
  test_feature_contract test_sprint2_protocol test_generation_calibration
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=tests .venv/bin/python -B -m unittest \
  test_sprint1.SprintOneTests.test_forecaster_roundtrip_and_schema \
  test_sprint1.SprintOneTests.test_history_and_workspace_isolation \
  test_sprint1.SprintOneTests.test_checkpoint_copies_and_no_overwrite \
  test_sprint1.SprintOneTests.test_quantized_placement_and_default_lora -v
```

**67 synthetic tests pass normally and with assertions disabled**: 42 contract,
11 protocol and 14 calibration tests. Four selected existing **non-training**
Sprint 1 compatibility tests pass, with the existing non-failing urllib3/LibreSSL
warning. The complete Sprint 1 suite is deliberately not run because it includes training.
All new fixtures are synthetic; no learning or model download occurs. Original
schema hash is historical, not the checksum of the revised 0.2 contract.
The real calibration config's offline preflight exits 2 with an explicit unresolved
task/manifests error before model import. No measurements/caps are claimed.
AST syntax checks pass for seven changed/new Python files; whitespace checks pass
for all 12 changed files, including untracked additions. All 33 table rows match
schema order; 139 local document links/line-anchor bounds and decision statuses
were verified. Diff review and credential-pattern scan found no accidental Sprint 1
edits or credential matches (pattern scanning is not a guarantee of secret absence).

Changed files: revised schema/calibration config; feature contract, protocol and
calibration helpers/CLI; three fixture test files; this audit, decision sheet and
evaluation draft. No tracked Sprint 1 runtime code changed. No training, labels,
downloads, SSH/jobs, checkpoint/environment changes or freezing occurred.
These are historical helper-stage counts/boundaries. Current collector/runtime/
opt-in-evaluation verification is in the integration report. Remaining blockers
are actual data/root/configuration/semantic bindings and allocated-model verification,
not permission to start jobs from a passing validator.
