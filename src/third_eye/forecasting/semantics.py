"""D15: a hash of everything that changes what a number MEANS.

Two datasets (or a dataset and a checkpoint) with different hashes must not be
mixed silently: e.g. LHV weights, history rule, or scoring definition changed.
Bump ``schema_version`` whenever a feature formula in ``features.py`` changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Tuple

HISTORY_RULES = ("applied_h1", "applied_h2")


@dataclass(frozen=True)
class SemanticConfig:
    lhv_weights: Tuple[float, float, float] = (0.50, 0.25, 0.25)   # D04
    history_rule: str = "applied_h1"       # D10 (pending): applied_h1 | applied_h2
    scoring: str = "accuracy_fraction_delta_vs_parent"             # D04/D06
    schema_version: str = "0.2-pilot+d05-d09"

    def __post_init__(self):
        if self.history_rule not in HISTORY_RULES:
            raise ValueError("history_rule must be one of %s" % (HISTORY_RULES,))

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
