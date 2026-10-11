"""D15: the exact semantic specification, and a hash of it.

Everything that changes what a number MEANS lives here, including the
formula text and the pinned frozen-base revision, so editing a definition (not
just a version string) changes the hash.  Records and checkpoints carry the
hash; mixing data or models with different hashes fails loudly.

The base-model revision is supplied by the collector (e.g. the exact Hugging
Face commit of Qwen3-4B-Base used for diversity embeddings).  Until it is
given the spec is UNPINNED and must not be used for real data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

HISTORY_RULES = ("applied_h1", "applied_h2")
UNPINNED = "UNPINNED"

FEATURE_DEFINITIONS: Dict[str, str] = {
    "D06_confidence": "per answer: mean token log-prob over the final-answer span only (no prompt, "
                      "reasoning, padding, special tokens); mean and POPULATION std across answers, "
                      "equal answer weight; empty set raises",
    "D07_duplicates": "NFC + whitespace normalisation, case/punctuation kept; whitespace-token "
                      "3-gram Jaccard >= 0.90 EXACT (rational comparison, no tolerance); texts "
                      "< 3 tokens need exact normalised equality; transitive connected components; "
                      "rate = (N - components) / N",
    "D08_length": "post-truncation non-padding tokens of serialised prompt+answer incl. tokenizer "
                  "specials; mean and population std",
    "D08_nll": "mean over answers of each answer's mean-token NLL (answer tokens only)",
    "D08_verifier_rate": "passing attempts / ALL correction attempts",
    "D08_difficulty": "original parent failures / unique attempted source examples, measured on the "
                      "unfiltered source pool BEFORE correction/filtering; no redraw to force a mix; "
                      "empty correction batch stops the pilot",
    "D08_diversity": "mean unordered pairwise cosine distance of L2-normalised mean-pooled "
                     "non-special last-layer hidden vectors, frozen base without adapters; "
                     "singleton 0; zero vector raises",
    "D09_gradients": "gradient of the D08 NLL at the same parent, eval mode, full trainable LoRA A/B "
                     "manifest, unscaled, before clipping; L2 norm; cosine = dot/(norm*norm), 0 with "
                     "flag if a norm is zero; sign agreement over ALL elements with sign(0)=0; "
                     "empty or non-finite raises; log zero-norm flags and zero counts (each, both)",
    "D05_layer_depth_bins": "per-module |gA|^2+|gB|^2 summed per transformer block (numeric order), "
                            "pooled by relative depth into B bins, as fractions summing to 1",
    "D12_probe": "15-step temporary update; train loss at steps 1..15, anchor/target loss at steps "
                 "0/5/10/15 on FIXED subsets, all with the D08 mean-of-means NLL; slopes = OLS vs "
                 "optimizer step; KL(parent||post) full vocab averaged over valid positions within "
                 "each answer then equally across answers; lora_delta = L2 change of A/B params",
    "D04_lhv2": "weighted mean of H=2 (target, ood, retention) accuracy-fraction deltas vs the parent",
    "D10_history": "last 3 applied one-step (t+1) results, oldest first, zero left-padding, mask = "
                   "real observed row; prev_accepted_value = lhv2 utility of that slot's deltas",
    "D11_continuation": "H=2 continuation recipe and RNG keyed by (group, state), shared by all "
                        "siblings, fixed before outcomes are seen",
}


@dataclass(frozen=True)
class SemanticConfig:
    lhv_weights: Tuple[float, float, float] = (0.50, 0.25, 0.25)   # D04
    history_rule: str = "applied_h1"                                # D10 (agreed)
    scoring: str = "accuracy_fraction_delta_vs_parent"
    schema_version: str = "0.3-d05-d12-agreed"
    base_revision: str = UNPINNED          # exact frozen-base commit/hash for D08 diversity
    pooling: str = "mean_non_special_last_hidden_l2norm"
    duplicate_threshold: str = "0.90"
    probe_train_steps: Tuple[int, ...] = tuple(range(1, 16))
    probe_eval_steps: Tuple[int, ...] = (0, 5, 10, 15)
    definitions: Dict[str, str] = field(default_factory=lambda: dict(FEATURE_DEFINITIONS))

    def __post_init__(self):
        if self.history_rule not in HISTORY_RULES:
            raise ValueError("history_rule must be one of %s" % (HISTORY_RULES,))

    @property
    def pinned(self) -> bool:
        return self.base_revision not in ("", UNPINNED)

    def to_dict(self) -> Dict:
        return asdict(self)

    def fingerprint(self, require_pinned: bool = False) -> str:
        if require_pinned and not self.pinned:
            raise ValueError("base_revision is not pinned; supply the exact frozen-base revision")
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
