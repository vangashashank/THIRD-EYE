# Sprint 2 Task 1: Forecasting Feature Contract Audit

**Audit complete; isolated loader/validator implemented and tested. Contract
definitions remain proposed/unresolved, not frozen; no training integration.**

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
An exact, portable copy now lives at [configs/feature_schema.yaml](../configs/feature_schema.yaml),
with the same SHA-256 and unchanged `frozen_on: null`.
Its header calls it "FROZEN", but `frozen_on` is null and this task explicitly
prohibits freezing. The attachment's comments describe a proposed contract;
they are not execution instructions or proof of agreement with Mahitha.
The repository still uses a separate Python smoke schema. All 33 named YAML
fields, its controls, and derived dimensions are compared below.

## Implemented Isolated Step

Added [feature_contract.py](../src/third_eye/feature_contract.py) and
[fixture-only tests](../tests/test_feature_contract.py). Existing Sprint 1 source,
checkpoint format, training scripts and tests are unchanged. No production module
imports the new validator. NumPy and PyYAML were already dependencies.

- `load_feature_contract(path, dimensions=..., layer_order=...)` supports the
  supplied `0.1-pilot` format. It uses a restricted `SafeLoader`, rejects duplicate
  keys, YAML merge keys, unsafe tags, missing/unknown fields and unsupported
  dtypes/shapes/version, and preserves declared group/field order. No descriptions
  or `derived` prose are executed. Source, schema and configured-layout hashes
  expose schema/order/configuration changes without silently approving them.
- Explicit bindings for **both** `num_lora_layers` and `history_len`, plus an
  ordered, unique layer manifest, are required for input validation. Layer names
  never infer a count; a count never infers an order. Configured history length
  must match the YAML. Missing bindings remain readable/reportable, not guessed.
  Dimensions resolve to state=5, candidate=16+L, and flattened history=5H=15 for
  the declared H=3; this is not a GRU input width or a new architecture.
- `validate_inputs` accepts one unbatched candidate's **named, nested NumPy
  arrays** with exact declared dtypes/shapes and finite values, including padding.
  It does not coerce lists, float64, int64 or masked arrays. Labels are forbidden
  at the top level and inside any input group. `validate_labels` is a separate
  API accepting only the `labels` group; nothing combines labels into inputs.
- History requires an explicit `HistoryContext`: ranking state, padding side,
  chronological direction and slot-aligned provenance. The mask must be float32,
  binary and contiguous under that policy; all masked values must be zero and
  their provenance absent. Real zero changes remain valid observations. Applied
  rounds must precede the ranking state and be unique/in the declared order.
- Each real history value needs an `ObservationAvailability` source and first
  available state. Under the caller's supplied indexing convention, update r
  reaches H1 at state r+1 and H2 at r+2. `applied_h2` is rejected before r+2 or
  before its observation becomes available; `simulated_h2` is always rejected.
  A stored `forecast` is permitted only for `prev_accepted_value`, not silently
  relabeled as an actual score change. These source tags describe availability;
  they do not choose an LHV formula or approve predicted-versus-realized history.

**Limits:** results are always `structural_only=True` and contain
`unresolved_definitions`. The validator does not extract any research feature,
check its truth, apply metric normalization, select the latest history window,
verify completeness/run/parent identity against a ledger, or independently prove
that provenance was recorded before ranking. Those need agreed definitions and a
future producer/ledger integration. It does not implement GRU masking, an update
probe or parent/training-state restoration. No schema approval/freeze is inferred
from the header or a date. The synthetic fixtures' layer identities, scalar zeros
and caller policies are **not extracted research features or agreed settings**.

Read-only inspection without inventing configuration:

```python
from src.third_eye.feature_contract import load_feature_contract

contract = load_feature_contract("configs/feature_schema.yaml")
print(contract.dimension_summary())  # state=5; candidate/history unresolved
print(*contract.unresolved_definitions(), sep="\n")
```

## Findings

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
5. **Availability and identity must become enforceable contracts.** Existing
   isolated workspaces help, but the fit joins by candidate ID and asserts state
   zero; it does not validate a complete run/trajectory/parent/evaluation identity
   or observation cutoff. YAML calls `prev_accepted_value` the previous accepted
   LHV but does not define predicted versus realized value. A realized H=2 value
   can require the very update being ranked, leaking future information. See
   [current join checks](../scripts/fit_direct_forecaster.py#L21).

## Schema Field Comparison

Named-feature statuses below describe **feature production and Sprint 1 wiring**,
not standalone structural validation. Adding a validator does not create the
requested measurements. Metadata/control implementation notes include the new
loader separately.

Required definitions here mean **what the supplied YAML declares**, not agreed
research semantics. Missing details are explicit. Shapes are per state/candidate,
before batching. All fields are `float32` except `current_gen_index` (`int32`).
`L = num_lora_layers` is unresolved; `H = history_len = 3` below denotes the history
window, not the H=2 forecasting horizon. Ordered YAML lists give field order, but
do not specify order inside the length-`L` vector.

**Implemented** means the exact declared field is produced, validated, and wired
into the applicable model/data path. **Partial** means a relevant primitive or
related recorded quantity exists, but semantics, shape, or wiring are incomplete.
**Missing** means the feature/label calculation does not exist; a related loss,
hash, or proposal ID is not a substitute.

Named-field coverage: **0 implemented, 10 partial, 23 missing** across the 33
fields. Partial does not mean that the YAML metric is already measured.

### Current State Performance: Per State

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `target_val_accuracy` | Target-task validation accuracy; `[1]`. Answer parsing, correctness rule, split, and scale are unspecified. | Parent `target` is full-text token loss [pilot:146](../scripts/run_sprint1_pilot.py#L146), assembled as loss [fit:36](../scripts/fit_direct_forecaster.py#L36), not accuracy. | Missing | Agree verifier/generation/evaluation policy and add accuracy extraction. A fixture with 3 correct of 4 must return the agreed fraction/percentage regardless of token lengths; zero valid examples must fail explicitly. |
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
| `layerwise_grad_norms` | Per-layer LoRA gradient L2 norms; `[L]`. Counting unit and vector order unresolved. | LoRA placement/rank configured [lora:15](../src/models/lora_model.py#L15); no layer manifest or norm vector. Digest's lexical name sort [checkpoint:14](../src/training/checkpoint.py#L14) is not this ordering. | Missing | Agree blocks versus modules versus A/B matrices; enumerate stable numeric block/module order. Test blocks 2 and 10, exact length, A/B grouping and same-shape permutation rejection. |
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

YAML requires the last up-to-3 rounds with zeros plus a mask when fewer exist.
Padding side, within-window order, observed endpoint and value availability still
need agreement. The current store keeps all chronological applied transitions.

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `prev_accepted_value` | Accepted update's LHV for each prior round; `[H] = [3]`. Predicted versus realized LHV/endpoint unspecified. | No LHV field; applied t+1 loss reductions only [history:71](../src/third_eye/history_store.py#L71). Full simulated t+2 labels [pilot:167](../scripts/run_sprint1_pilot.py#L167) are not applied history. | Missing | Resolve availability before implementation. Fixture current ranking must reject a previous update's not-yet-observed H=2 LHV, even if a hindsight label file contains it; test first state and delayed outcomes. |
| `current_gen_index` | Zero-indexed current generation `t`; `int32 [1]`. | `state_id` is stored/validated [state:42](../src/third_eye/state_manager.py#L42); not supplied to Direct [fit:36](../scripts/fit_direct_forecaster.py#L36). | Partial | Define correspondence with recursive generation and embedding/normalization route. Test t=0, positive t, range/dtype validation and that candidate ID is not substituted for t. |
| `prev_target_change` | Prior target changes; `[H] = [3]`; YAML gives no formula, sign, units or endpoint. | Target `before_loss-after_loss` channel [history:77](../src/third_eye/history_store.py#L77), unpadded full-history list [history:98](../src/third_eye/history_store.py#L98); fit ignores it. | Partial | Agree actual applied t+1 score changes versus another definition; select last three eligible rounds and mask/pad. Test five rounds, first state, improvement/worsening sign and exclusion of unselected/simulated candidates. |
| `prev_ood_change` | Prior OOD changes; `[H] = [3]`; formula/endpoint unspecified. | OOD loss-reduction channel in metric order [history:6](../src/third_eye/history_store.py#L6), not YAML score history. | Partial | Use agreed OOD metric/manifest and identical window/mask alignment. Distinct fixture channels must not be swapped; reject a future observation or different trajectory. |
| `prev_retention_change` | Prior retention changes; `[H] = [3]`; formula/endpoint unspecified. | Retention loss-reduction channel [history:77](../src/third_eye/history_store.py#L77), no three-round collator. | Partial | Use agreed anchor scoring and window. Test real zero change versus padding, oldest retained round boundary, and consistent endpoint across all history channels. |
| `history_mask` | 1 for real entry, 0 for padding; `[H] = [3]`. Not a mask for separately delayed values. | No mask in store return [history:98](../src/third_eye/history_store.py#L98) or Direct signature [Direct:41](../src/third_eye/direct_forecaster.py#L41); GRU processes all rows [Direct:85](../src/third_eye/direct_forecaster.py#L85). | Missing | Add aligned binary mask and length-aware encoding in a new path. Test histories of lengths 0/1/2/3, masked-value perturbation invariance, batched-versus-individual predictions, and delayed LHV policy. |

### Labels: Per Candidate, Offline Supervision Only

| Schema field | Required definition and shape | Current implementation / code reference | Status | Required change and meaningful validation |
| --- | --- | --- | --- | --- |
| `delta_target_h1` | Ground-truth target-score change at t+1 for H1 baseline; `[1]`; score, sign and reference endpoint unspecified. | Raw parent/t+1 losses recorded [pilot:167](../scripts/run_sprint1_pilot.py#L167); no named H1 delta label or H1 model path. | Partial | Agree score definition and derive named parent-to-t+1 target delta during full labeling. Verify endpoint/sign on distinct parent/t+1/t+2 fixture and keep it out of ranking inputs. Loss cannot recover missing accuracy. |
| `delta_target_h2` | H2 target delta by name; `[1]`; no description specifying score/reference/sign. | Legacy `labels[0]` is parent target loss minus t+2 loss [pilot:170](../scripts/run_sprint1_pilot.py#L170); Direct returns three H2 outputs [Direct:7](../src/third_eye/direct_forecaster.py#L7). | Partial | Agree H2 update/continuation policy and label meaning. Preserve legacy loss format separately; verify parent-to-t+2 versus t+1-to-t+2 and that changing labels never changes input features. |
| `delta_ood_h2` | H2 OOD delta by name; `[1]`; score/reference/sign unspecified. | Legacy `labels[1]` is parent minus t+2 OOD loss [pilot:170](../scripts/run_sprint1_pilot.py#L170). | Partial | Derive named agreed OOD delta under a fixed evaluation manifest. Test known improvement/worsening and reject mismatched proxy populations or metric units. |
| `delta_retention_h2` | H2 retention delta by name; `[1]`; score/reference/sign unspecified. | Legacy `labels[2]` is parent minus t+2 retention loss [pilot:170](../scripts/run_sprint1_pilot.py#L170). | Partial | Derive named agreed retention delta under the fixed anchor manifest. Test endpoint/order/sign, finite values and label-only availability. |
| `lhv2` | Pre-declared scalar aggregate of three H2 deltas for ranking/reporting; `[1]`. Formula, weights, units/normalization and direction absent. | No aggregate; scalar script computes continuation reduction, rank and regret [scalar labels:89](../scripts/prepare_forecasting_labels.py#L89), none of which is specified as LHV. | Missing | Obtain formula and worked example before code/data generation. Check known three-delta fixture and use the same declared rule on predicted deltas for ranking versus realized deltas for offline reporting, with distinct record types. Never rank using the ground-truth label. |

### Metadata, Controls and Derived Dimensions

| Schema entry | Declared requirement / shape | Current implementation | Status | Required change / validation |
| --- | --- | --- | --- | --- |
| `schema_version` | String `0.1-pilot`. | Isolated loader validates version and hashes ordered schema/configuration [loader:341](../src/third_eye/feature_contract.py#L341). Existing checkpoint version 1 remains separate. | Implemented | Future producer/checkpoint integration must compare fingerprints explicitly; standalone loader does not migrate legacy checkpoints. |
| `frozen_on` | Null in attachment; header's frozen wording does not supply approval/date. | Loader preserves metadata and always reports approval unverified [decisions:50](../src/third_eye/feature_contract.py#L50). | Implemented | Source remains null and unchanged; never infer agreement from a header/date. |
| `current_state_performance.scope` | `per_state`, shared across K=3. | Three repeated parent-loss rows [fit:36](../scripts/fit_direct_forecaster.py#L36), not the five declared fields. | Partial | Share one validated parent observation per state; all three candidates must receive identical state features. |
| `candidate_data_stats.scope` | `per_candidate`. | Proposal hyperparameters only [fit:37](../scripts/fit_direct_forecaster.py#L37). | Missing | Extract each candidate's declared batch statistics; require complete candidate identity. |
| `gradient_interference_stats.scope` | `per_candidate`. | No extracted gradient feature inputs [fit:37](../scripts/fit_direct_forecaster.py#L37). | Missing | Bind gradient extraction to exact candidate/parent/parameter manifest; reject stale cache. |
| `short_probe_dynamics.scope` | `per_candidate`. | Full candidate training [generator:155](../src/third_eye/candidate_generator.py#L155), no temporary feature probes. | Missing | Separate probe records from full outcomes; validate candidate and budget. |
| `short_probe_dynamics.probe_steps` | Integer 15; comments mention 10-20 pilot range. | Training is configured in epochs [trainer:12](../src/training/trainer.py#L12), not a bounded 15-step probe. | Missing | Agree what counts as a step and exact stopping/measurement schedule. Fixture trace length and actual optimizer steps must match recorded budget. |
| `recursive_history.scope` | `per_state`, shared across candidates. | One file per trajectory [history:10](../src/third_eye/history_store.py#L10), no fit wiring. | Partial | Produce one aligned history window per state; candidate ordering must not change it. |
| `recursive_history.history_len` | Integer 3. | Store returns all preceding rows [history:98](../src/third_eye/history_store.py#L98). | Missing | Select last three with agreed order/padding. Five eligible rounds must not produce five history slots. |
| `labels.scope` | `per_candidate`, expensive labeling only; unavailable at ranking. | H2 loss-label rows [pilot:167](../scripts/run_sprint1_pilot.py#L167), not complete YAML labels. | Partial | Keep labels in separate objects/loader and validate run/trajectory/state/candidate identity. Future-label mutation must leave inputs unchanged. |
| `derived.static_candidate_feature_dim` | `8 + (3 + L) + 5 = 16 + L`. YAML value is explanatory text, not executable code. | Standalone shape-product summary [dimensions:207](../src/third_eye/feature_contract.py#L207); returns unresolved without explicit L. Sprint 1 width remains 4. | Implemented | Research L/manifest still needs agreement; no new model wiring or evaluation of derived prose. |
| `derived.static_state_feature_dim` | `5`. Does not include separately routed generation index. | Standalone summary returns 5 from declarations [dimensions:207](../src/third_eye/feature_contract.py#L207); Sprint 1 width remains 3. | Implemented | Generation-index embedding/normalization and integration remain undecided. |
| `derived.history_feature_dim` | Five length-H arrays, including mask, excluding index: `5H = 15`. | Standalone summary requires explicit H matching declaration [dimensions:207](../src/third_eye/feature_contract.py#L207); existing GRU step width remains 3. | Implemented | Distinguish serialized/flattened width from recurrent step width. Four value channels plus a separate mask is a proposed mapping, not implemented GRU masking. |

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
  Pin architecture, adapter placement and qualified module identities, then agree
  numeric block and within-block order. Lexical parameter hashing is unrelated.
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
  improvement-oriented delta would instead be `accuracy(after)-accuracy(parent)`.
  Agree units (fraction versus percentage), endpoint, direction and metric type.
  Existing loss files cannot reconstruct unrecorded answer accuracy.
- `lhv2` is explicitly a scalar aggregate, but YAML contains **no pre-declared
  formula** despite that wording: weights, signs, scaling, constraints and metric
  inputs must be supplied. Do not assume a sum, target-only score, rank or regret.
- YAML fixes `history_len=3` and declares zero padding plus a float32 mask. It does not define
  padding side, oldest/newest order within the last-three window, or a way to
  represent an unavailable LHV for an otherwise real transition. These are
  semantic and batching decisions, not permission to guess values.

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

| Possible interpretation, not an agreed definition | Available? | Required guard |
| --- | --- | --- |
| Forecast produced when the previous candidate was accepted | Yes, if stored before the current cutoff. | Record decision/forecaster revision and distinguish predicted from realized values; do not recompute with a future-trained forecaster. |
| Previous accepted update's realized t+1 loss or improvement | Yes only once that actual applied update has completed and been evaluated before the current ranking. | Bind observation to the applied parent/successor and consistent evaluation manifest; missing/delayed observations need an explicit policy. |
| Previous candidate's realized H=2 value while ranking the next update | Generally not yet available if the current update is needed to reach that endpoint. | Keep it out of current inputs until genuinely observed; offline simulated continuation is not an actual realized transition. |
| Value of the candidate that will be accepted at the current decision | No; selection has not happened. | Do not fill this using the current winner's full outcome. |

In state notation, while ranking `U_t` at `S_t`, the previous accepted update
`U_(t-1)` has reached its t+1 endpoint `S_t`; its t+2 endpoint `S_(t+1)` generally
depends on the not-yet-selected `U_t`. If YAML's accepted "LHV" means ground-truth
`lhv2`, reading that value from the most recent round therefore leaks the current
decision's future. An offline simulated branch label does not make it observable
under a cheap pre-selection deployment protocol.

The single `history_mask` marks **real rounds**, not per-field outcome maturity.
A real prior transition may have observed t+1 changes while its H=2 LHV remains
unavailable. Setting mask=1 and filling LHV with a future label or a fake observed
zero is unsafe. Options for Mahitha are a previously stored predicted LHV, delayed
eligible history, or an explicitly revised availability-mask contract; none is
selected or added here.

Suggested distinct names such as `prev_accepted_forecast` and
`prev_realized_t1_score` would reduce ambiguity, but are proposals for agreement,
not additions to an approved schema. At the first state there is no previous
accepted candidate. A fake zero must not stand in for both absence and true zero
improvement. Record logical transition and observation provenance, not timestamps
alone. A no-leakage test should mutate/remove all later labels and require current
features to remain identical; another should reject future-dated history entries.

## Temporary Probes Versus Full Labels

| Data product | Availability / role | Current implementation |
| --- | --- | --- |
| Parent evaluation and static proposal | Before a candidate update; legitimate inputs under the legacy contract. | Parent [pilot:149](../scripts/run_sprint1_pilot.py#L149); proposal [generator:96](../src/third_eye/candidate_generator.py#L96). |
| Existing `probe()` | Raw one-prompt next-token logits used for save/reload equivalence, not update-probe features or confidence. | [pilot:57](../scripts/run_sprint1_pilot.py#L57); no temporary training step occurs inside it. |
| Proposed bounded pre-selection update probe | YAML proposes 15 steps (comments allow pilot discussion of 10-20); usable only under an agreed budget, optimizer, population, extraction and cutoff policy. | Missing. Must use a separate namespace/record type and must not promote or append history. |
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

**Recommended implementation direction, not a present capability:** run each probe
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

1. Review this completed inventory of supplied YAML version `0.1-pilot` with
   Mahitha. Settle the listed meanings/availability and a stable adapter manifest;
   do not treat the attachment's frozen header as approval. A loader prototype can
   report unresolved symbols before agreement, but runtime feature construction
   must wait for the required definitions.
2. **Completed isolated step:** contract loader/validator and synthetic tests,
   described above. It validates shapes/dtypes/finite values, masks and declared
   as-of availability; it reports missing configuration and all semantic decisions.
   No runtime feature construction or training-path imports were added. Unit/metric
   semantics, verified identity/ledger joins and feature producers remain future
   work rather than silently filled definitions.
3. Add a pure feature builder taking only validated parent measurements, proposals,
   allowed probe records, and as-of applied history. Join on complete identities,
   parent/model/evaluation manifests, and explicit record types. Keep label loading
   separate. Test cross-run collisions and later-label mutations without training.
4. Implement the agreed stable LoRA manifest and cheap deterministic features,
   then a separate transactional probe runner if needed. Add exception/restoration
   tests before GPU use. Confidence and `lhv2` extraction wait for definitions.
5. Add a history collator and opt-in Direct v2 length/mask handling, including
   mixed zero/nonzero lengths. Version full input layout and checkpoint contract;
   preserve v1 loading and numerical behavior. Test padding invariance and layer
   order mismatch rejection using synthetic forward passes, not training.
6. Implement the agreed previous-accepted-value policy and metric-specific labels
   with explicit endpoints/directions. Add no-future-information and sign tests.
7. Only after contract agreement and separate execution authorization, design
   trajectory-level held-out evaluation and bounded data collection. Compare
   constant-output/ranking baselines and any confidence calibration. Do not treat
   this audit or the single-state smoke fit as forecasting validation.

## Decisions Requiring Agreement With Mahitha

1. Contract ownership/version and approval process; the copied header is not proof
   of freezing. `frozen_on` remains null.
2. LoRA counting unit and canonical layer/module/A-B manifest, including cross-model
   layouts; no research count or order has been chosen.
3. Accuracy/scoring/generation, target/OOD sets and fixed 256-example anchor;
   score units/direction/endpoints and a worked `lhv2` formula.
4. Generation/corrected-answer confidence formula, reduction and std convention;
   these are not forecaster-uncertainty outputs.
5. Duplicate-example rule/threshold/denominator, tokenizer/length/NLL policies,
   embeddings, verifier attempt population and difficulty observations.
6. Accepted LHV source/maturity, last-three selection/order/padding, delayed-value
   masking, generation-to-applied-round indexing and index routing; ledger provenance.
7. Probe budget/data and optimizer/RNG/restoration policy; gradient population,
   scaling/zero cases, slope coordinates/cadence, KL and adapter-delta representation.

## Verification and Scope

Read and safely parsed the complete user-supplied YAML, recorded its SHA-256, and
inventoried 33 named fields in six groups (32 float32, one int32). Read current
schema/checkpoint, Direct, fit, pilot, proposal/candidate/consequence
construction, label preparation, history, LoRA, training, evaluation, and existing
tests; inspected the Sprint 1 report and repository state. The initial audit's
forward-only synthetic GRU diagnostic is retained above, not a research result.
Field table coverage/order and links were checked; no field producer was added.

Implemented-step checks:

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_feature_contract.py -v
.venv/bin/python -B -O -m unittest discover -s tests -p test_feature_contract.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=tests .venv/bin/python -B -m unittest \
  test_sprint1.SprintOneTests.test_forecaster_roundtrip_and_schema \
  test_sprint1.SprintOneTests.test_history_and_workspace_isolation \
  test_sprint1.SprintOneTests.test_checkpoint_copies_and_no_overwrite \
  test_sprint1.SprintOneTests.test_quantized_placement_and_default_lora -v
```

36 new synthetic fixture tests pass normally and with assertions disabled; 4
selected existing **non-training** compatibility tests pass. The latter emitted
the existing non-failing urllib3/LibreSSL warning. The complete Sprint 1 suite was
not run because it includes training. Schema source and repository copy have
identical SHA-256. The new tests are clearly marked synthetic, not extracted
research features; no learning or model download occurs.
AST syntax checks pass for both new Python files, all four changed files have no
trailing whitespace, and 114 local document links/line-anchor bounds were checked.

Changed files: this audit, `src/third_eye/feature_contract.py`,
`tests/test_feature_contract.py`, and the exact `configs/feature_schema.yaml` copy.
No existing tracked Sprint 1 code was changed. No training, model downloads, SSH
jobs, persistent checkpoint/environment changes, or schema freezing were
performed. Remaining blockers are semantic agreement, a verified
research manifest/provenance source and future extraction/integration, not local
test execution or server access.
