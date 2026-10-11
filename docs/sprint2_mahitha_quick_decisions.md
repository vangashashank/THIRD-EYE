# Sprint 2: Five Remaining Formula Decisions

2026-10-11. **All entries below are proposed, awaiting agreement.** No defaults
are accepted by being implemented. This sheet does not reopen accepted decisions,
change the 8/2/2 lineage plan or T=5 scope, or freeze the feature schema.
"Ours" means the current isolated collector, not the old Sprint 1 smoke pipeline.
"Hers" means actual integrated code at `dcd403a`, not inferred document intent.

## Exact Comparison and Recommendation

| Decision / exact question | Our current code | Mahitha's implementation | Exact difference; recommended resolution |
| --- | --- | --- | --- |
| **D06:** What answer span, weighting, state population and parse-failure policy define confidence? | [Backend](../src/third_eye/sprint2_backend.py) `measure_parent`: actual emitted VALUE tokens in original generation context; equal-answer mean/population std. Candidate `mean_confidence`: teacher-forced verified corrected VALUE spans at common parent. Boundary whitespace included when merged into a VALUE token; exclude header/reasoning/padding/specials. State parse/truncation failure stops. | [Features](../src/third_eye/forecasting/features.py) `sequence_confidence`: same mean-per-answer logprob, then mean/population std; caller must supply answer tokens and probe population. No generation/span/failure producer here. | Reduction agrees; populations, boundary mask and failed-parse behavior are not bound by her helper. **Recommend** these log scores and explicit `final_value_boundary_whitespace_v1`; pin a source-disjoint state-confidence manifest before outcomes, retain failures and stop on undefined VALUE spans for this pilot. Never drop failures silently or substitute correct reference confidence. This is not calibrated correctness/uncertainty. |
| **D07:** Which text, threshold, tolerance and denominator define duplicates? | [Measurements](../src/third_eye/sprint2_measurements.py) `duplicate_components`: complete corrected training serialization, NFC/whitespace with case/punctuation kept; word-trigram set Jaccard >= threshold minus explicit tolerance; short text exact; connected-component extra-copy rate `(N-components)/N`. Threshold/tolerance are unbound. | `duplicate_rate`: same algorithm/denominator, default threshold **.90**, tolerance **1e-12**; takes caller texts without defining the serialization population. | No accepted tolerance/text population yet. **Recommend** corrected serialization, .90 and 1e-12 to match her boundary rule, recorded explicitly in config/spec. Lexical overlap is not source independence. `[A,A,B]` gives 1/3. |
| **D08:** How are NLL, length, difficulty, verifier rate, corrections and embeddings measured? | `candidate_data`, backend `_failure_batch`/`_prepare`: NLL supports **equal-answer OR token-weighted**, no default; decision sheet previously recommends equal-answer. Length = all post-truncation nonpad serialized prompt+answer tokens including added specials. Difficulty = failures among ALL 32 original source attempts, not filtered corrected examples. Verifier = successes/ALL correction attempts. Reconsider failed answer at parent, verify, then serialize corrected VALUE in original context. Diversity = frozen-base, adapter-disabled last-hidden-state mean over nonspecial nonpad tokens, unordered pair cosine distance. | `token_weighted_nll`: token-weighted answer NLL. `difficulty_proxy`: mean per-prompt failure rate, >=4 sampled attempts recommended. `token_length_stats`: supplied prompt+answer counts, no tokenizer/special-token producer. `verifier_pass_rate` and `embedding_diversity` have the same reductions but supplied arrays only; zero embedding norm <1e-12 rejected. No correction/embedding producer. | **Recommend token-weighted answer NLL** to match her helper; this explicitly revises our earlier equal-answer recommendation, not an acceptance. Confidence stays equal-answer. Single greedy attempt on ALL drawn prompts makes our difficulty exactly her helper with one attempt/prompt, without adding sampling to the agreed pilot. Pin the current correction prompt/version, length/special handling and frozen-base pooling; reject zero/near-zero embedding norms <1e-12 consistently. Our current producer rejects exact zero only, so the near-zero embedding guard needs a small fix **after agreement**. |
| **D09:** Which gradient loss/mode/precision and zero-cosine rule apply? | `gradient_measurements`: eval-mode unscaled/unclipped aligned LoRA A/B autograd on explicit D08 loss; parameter gradients converted to float32 then reduced in float64. Cosine =0 for either norm **<= configured epsilon**; config epsilon unbound. Sign equality over all elements, sign(0)=0; exact-zero norm and zero-in-candidate/retention/both diagnostics. Finally restores state. | `grad_l2_norm` says token-mean answer CE, unscaled/before clipping. `grad_cosine`: float64 reduction, zero if either norm **<1e-12**. `sign_agreement` and `zero_sign_stats` match our signed/zero-element reductions. Caller supplies gradients; no eval-mode/AMP/alignment producer. | **Recommend** token-weighted D08 loss, eval-mode model compute with float32 extracted gradients/float64 reduction, exact manifest, and cosine **norm <1e-12** (not <=) to match her helper. Tiny values still get exact-zero diagnostics, not falsified zeros. Boundary operator requires a small isolated fix after agreement. Retention-gradient population remains fixed 32 approved anchors. |
| **Probe reductions only:** Which loss trace, KL and parameter delta? | `probe_measurements`: fixed-batch train/anchor/val OLS at **actual steps0/5/10/15**; optimizer losses1..15 logged separately. Teacher-forced, full-vocab KL(parent to post), valid anchor VALUE positions, float32 log-softmax/float64 token mean, **clamp final mean >=0**. Delta = raw A/B L2. | `linear_slope` defaults x=0..n-1; doc recommends optimizer train losses1..15, anchor/val0/5/10/15. `mean_token_kl`: same direction/mask, float64 log-softmax, **clip each token >=0 before mean**. `param_delta_norm` agrees. No execution/restoration producer. | **Recommend fixed-batch train slope at0/5/10/15**, avoiding changing-example confounding; pass actual cadence. Keep float32 log-softmax/float64 mean, but match her per-token clipping; document numerical tolerance, not bitwise equivalence. Train trace, precision and rounding/clipping remain unresolved. Step budgets/subsets unchanged. |

## Agreement Versus Implementation

- **Missing agreement:** all five rows above, including correction/answer-span
  policy within D08/D06. Set `definition_agreements.D06/D07/D08/D09/D12_measurements`
  only with an actual decision reference, then populate `definitions.*`.
- **Already implemented:** both NLL reductions, confidence reductions, explicit
  duplicate threshold/tolerance, source-attempt difficulty, verifier denominator,
  corrected-batch producers, aligned gradients, probe traces and restoration.
  Her helpers consume these arrays; their existence does not supply real evidence.
- **Conditional implementation fixes:** if recommendations are agreed, match
  D09's strict `<` boundary, D08's near-zero embedding guard and per-token KL
  clipping (currently our producer clips only the final mean; version the change).
  If another
  confidence/correction/pooling/trace definition is chosen, version and test that
  producer before collection. Do not edit formulas to guess agreement.
- **Missing bindings, NOT formula decisions:** actual task/source/diagnostic/
  confidence manifests, scorers, independent roots and identities, model module/
  tokenizer manifests, measured caps and allocated MIG/quota. Shashank supplies
  sources/access; jointly review task-native scoring. See the exact blocker table
  in [handoff integration](sprint2_handoff_integration.md).

## Acceptance Checks (Synthetic, Not Measurements)

1. Confidence answer scores -1/-3 -> mean -2, population std1. No valid VALUE
   span -> explicit retained failure, no replacement feature.
2. NLL examples: one token with NLL1, three tokens each NLL3 -> token-weighted2.5
   versus equal-answer2.0. Store chosen reduction in every spec/hash.
3. Difficulty: 8 original failures of32 ->.25 even if all8 corrections succeed;
   verifier8/8 ->1. Source failures must not be recomputed on the filtered batch.
4. Gradient norm exactly1e-12: proposed strict `<` rule does not zero its cosine;
   exact zero does. Signs `(0,1)` vs `(0,-1)` ->.5, with zero-in-both=.5 logged.
5. Fixed-population losses4/3/2/1 at steps0/5/10/15 -> slope-.2 per optimizer
   update, not-1 per array index; no temporary probe may advance history.

## Draft Message (Not Sent)

Mahitha, can we resolve the five rows above so the first real Qwen3-4B T=5 batch
can be sealed? I recommend equal-answer log confidence on declared VALUE spans
(retain/stop on undefined spans); duplicate .90 with1e-12 tolerance on corrected
serialization; token-weighted answer NLL, difficulty over all32 original greedy
attempts, and the versioned correction/frozen-base pooling rules; eval-mode
aligned gradients with cosine norm <1e-12 and zero diagnostics; and fixed-batch
probe slopes at0/5/10/15 with parent||post answer-token KL. Please confirm or give
exact replacements, especially the state-confidence population/parse policy and
probe train trace. Accepted scope/history/continuation settings are unchanged.
Data/root bindings, calibration and real validation are still pending; synthetic
checks are not Gate1. This message is a draft only and has not been sent.
