"""Pinned definitions of the feature-schema quantities (numpy only).

Sprint 2's feature audit lists 23 schema fields as "missing" mainly because
their formulas were unspecified.  This module fixes one definition for each,
with the convention written in the docstring and checked by unit tests.  The
functions take plain arrays, so the training harness can call them on
``p.grad.flatten().float().cpu().numpy()``, logits, token counts, etc.

These are *recommended defaults*.  Changing one changes the meaning of a
feature column, so bump the layout name / schema version when you do.
"""
from __future__ import annotations

import unicodedata
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# Depth handling (cross-model transfer)
# --------------------------------------------------------------------------
def depth_bin_pool(values: Sequence[float], num_bins: int, reduce: str = "mean") -> np.ndarray:
    """Pool a per-block vector of any length L into ``num_bins`` relative-depth bins.

    Block i covers [i/L, (i+1)/L) of relative depth, bin b covers
    [b/B, (b+1)/B).  Each block contributes to a bin in proportion to overlap,
    so the result is defined for any L >= 1 (also L < B) and is mass
    conserving for ``reduce="sum"``.  This is how ``layerwise_grad_norms``
    becomes a fixed-width feature that is comparable between Qwen3-4B,
    Llama-3.2-3B and Gemma-3-4B.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D vector")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if reduce not in ("mean", "sum"):
        raise ValueError("reduce must be 'mean' or 'sum'")
    L, B = v.size, num_bins
    edges_l = np.arange(L + 1) / L
    edges_b = np.arange(B + 1) / B
    overlap = np.zeros((B, L))
    for b in range(B):
        lo = np.maximum(edges_b[b], edges_l[:-1])
        hi = np.minimum(edges_b[b + 1], edges_l[1:])
        overlap[b] = np.clip(hi - lo, 0.0, None)
    # overlap is in units of relative depth; convert to "fraction of a block"
    frac_of_block = overlap * L
    if reduce == "sum":
        return frac_of_block @ v
    weights = frac_of_block.sum(axis=1)
    return (frac_of_block @ v) / weights


def layerwise_energy_fractions(block_grad_norms: Sequence[float], num_bins: int) -> np.ndarray:
    """``layer_depth_bin_*`` columns: share of squared gradient norm per depth bin.

    ``block_grad_norms[i]`` is the L2 norm of all LoRA gradients (A and B, all
    adapted modules) in transformer block i, blocks in numeric order.  Squaring
    gives gradient "energy"; binning with ``reduce="sum"`` and dividing by the
    total removes both the model's scale and its depth.  Sums to 1.
    """
    norms = np.asarray(block_grad_norms, dtype=np.float64)
    if np.any(norms < 0) or not np.all(np.isfinite(norms)):
        raise ValueError("block norms must be finite and non-negative")
    return energy_fractions_from_blocks(norms ** 2, num_bins)


# --------------------------------------------------------------------------
# Candidate data statistics
# --------------------------------------------------------------------------
def token_length_stats(lengths: Sequence[int]) -> List[float]:
    """(mean, population std) of per-example token counts.

    Counts are after truncation, excluding padding, over prompt+answer tokens
    that enter the training loss.  Singleton batch -> std 0.
    """
    x = np.asarray(lengths, dtype=np.float64)
    if x.size == 0 or np.any(x < 0):
        raise ValueError("need at least one non-negative length")
    return [float(x.mean()), float(x.std())]


def mean_answer_nll(per_answer_token_nlls: Sequence[Sequence[float]]) -> float:
    """D08 ``pre_update_nll``: mean over answers of each answer's mean-token NLL.

    Answer tokens only (as in D06); every answer weighs the same, so long
    answers do not dominate.  This is also the loss D09 differentiates.  It is
    NOT the token-weighted batch mean (that is what legacy full-text loss does).
    """
    if len(per_answer_token_nlls) == 0:
        raise ValueError("empty batch")
    means = []
    for nll in per_answer_token_nlls:
        nll = np.asarray(nll, dtype=np.float64)
        if nll.size == 0 or not np.all(np.isfinite(nll)):
            raise ValueError("an answer has no (finite) answer tokens")
        means.append(nll.mean())
    return float(np.mean(means))


def sequence_confidence(per_example_token_logprobs: Sequence[Sequence[float]]) -> List[float]:
    """D06: (mean, population std) over answers of each answer's mean token log-prob.

    Final-answer span only (no prompt, reasoning, padding or special tokens);
    equal weight per answer; log scores, not probabilities.  Answers with log
    scores -1 and -3 give (-2, 1).  Used for ``mean_confidence`` (the batch's own corrected answers) and for
    ``gen_confidence_mean/std`` (a probe set).  Values are <= 0; higher means
    more confident.  Prompt tokens must already be excluded.
    """
    if len(per_example_token_logprobs) == 0:
        raise ValueError("empty set: confidence is undefined, not a fake finite value")
    per = []
    for lp in per_example_token_logprobs:
        lp = np.asarray(lp, dtype=np.float64)
        if lp.size == 0:
            raise ValueError("an example has no answer tokens")
        per.append(lp.mean())
    per = np.asarray(per)
    return [float(per.mean()), float(per.std())]


def verifier_pass_rate(passed: Sequence[bool]) -> float:
    """passed / attempted over ALL correction attempts, before any filtering."""
    p = np.asarray(passed, dtype=bool)
    if p.size == 0:
        raise ValueError("zero attempts: pass rate is undefined")
    return float(p.mean())


def difficulty_proxy(original_failed: Sequence[bool]) -> float:
    """D08: original failures / unique attempted source examples, before correction.

    ``original_failed[i]`` says whether the PARENT failed unique source example
    i on its first attempt; later successful corrections do not change it.
    Informative only if the attempted pool contains parent successes too; if the
    pool is just failures it is constant 1.0.
    """
    f = np.asarray(original_failed, dtype=bool)
    if f.size == 0:
        raise ValueError("no attempted source examples")
    return float(f.mean())


def mean_pool_hidden(hidden: np.ndarray, is_special: Sequence[bool]) -> np.ndarray:
    """D08 diversity input for ONE example: mean of last-layer hidden states over
    non-special tokens ([tokens, d] -> [d]), from the frozen base with no adapters.
    Feed the stacked vectors to ``embedding_diversity`` (which L2-normalises)."""
    h = np.asarray(hidden, dtype=np.float64)
    keep = ~np.asarray(is_special, dtype=bool)
    if h.ndim != 2 or keep.shape != (h.shape[0],) or not keep.any():
        raise ValueError("need [tokens, d] hidden states and at least one non-special token")
    return h[keep].mean(axis=0)


def _unit_rows(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("embeddings must be [n, d] with n >= 1")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms < _EPS):
        raise ValueError("zero-norm embedding")
    return emb / norms


def embedding_diversity(emb: np.ndarray) -> float:
    """Mean cosine *distance* (1 - cos) over unordered pairs i<j.  Singleton -> 0."""
    u = _unit_rows(emb)
    n = u.shape[0]
    if n == 1:
        return 0.0
    sim = u @ u.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def _normalise_text(text: str) -> List[str]:
    """NFC-normalise and split on whitespace.  Case and punctuation are kept
    (they can change the meaning of a math/code example)."""
    return unicodedata.normalize("NFC", text).split()


def _trigrams(tokens: List[str]) -> set:
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def duplicate_rate(texts: Sequence[str], threshold: float = 0.90) -> float:
    """D07: (N - number of near-duplicate components) / N, N > 0.

    Examples are linked when the Jaccard similarity of their sets of
    whitespace-token 3-grams is >= ``threshold``.  Texts shorter than 3 tokens
    have no 3-grams, so they must match exactly after normalisation.  Components
    are connected components, so the result does not depend on example order.
    [A, A, B] -> 1/3 (extra copies), singleton -> 0.  A lexical rule, not a
    semantic duplicate detector: source-family grouping (D02) is still needed.
    """
    n = len(texts)
    if n == 0:
        raise ValueError("empty batch")
    limit = Fraction(str(threshold))             # exact: no floating-point tolerance
    toks = [_normalise_text(t) for t in texts]
    grams = [_trigrams(t) for t in toks]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if len(toks[i]) < 3 or len(toks[j]) < 3:
                same = toks[i] == toks[j]
            else:
                union = len(grams[i] | grams[j])
                same = union > 0 and Fraction(len(grams[i] & grams[j]), union) >= limit
            if same:
                parent[find(i)] = find(j)
    return float((n - len({find(i) for i in range(n)})) / n)


# --------------------------------------------------------------------------
# Gradient / interference
# --------------------------------------------------------------------------
def _checked(g, name="gradient") -> np.ndarray:
    """D09: reject empty or non-finite gradients."""
    g = np.asarray(g, dtype=np.float64).ravel()
    if g.size == 0:
        raise ValueError("empty %s" % name)
    if not np.all(np.isfinite(g)):
        raise ValueError("non-finite %s" % name)
    return g


def grad_l2_norm(flat_grad: np.ndarray) -> float:
    """D09: L2 norm of the flattened gradient over the FULL trainable LoRA A/B
    manifest, of the D08 mean answer NLL (mean of per-answer means), at the same
    parent in eval mode, unscaled and before clipping or optimizer steps."""
    return float(np.linalg.norm(_checked(flat_grad)))


def grad_cosine(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """Cosine = dot / (norm_a * norm_b) over the same parameter manifest.  0.0 if
    either norm is zero (use ``gradient_stats`` to see the flag)."""
    a, b = _checked(g_a), _checked(g_b)
    if a.shape != b.shape:
        raise ValueError("gradients must have the same parameter manifest")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_with_diagnostic(g_a: np.ndarray, g_b: np.ndarray) -> Tuple[float, Dict[str, bool]]:
    """D09 cosine = dot / (norm_a * norm_b); 0.0 if either norm is zero, with a
    provenance flag so a zero is never mistaken for 'orthogonal'."""
    a, b = _checked(g_a), _checked(g_b)
    zero_a, zero_b = bool(np.linalg.norm(a) < _EPS), bool(np.linalg.norm(b) < _EPS)
    return grad_cosine(a, b), {"zero_norm_a": zero_a, "zero_norm_b": zero_b}


def sign_agreement(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """D09: fraction of ALL aligned elements with equal sign, sign(0) = 0.

    Two zeros therefore agree.  Always log ``zero_sign_stats`` next to it: zero
    gradients (e.g. LoRA A-matrices while B is still 0) inflate agreement.
    """
    a, b = _checked(g_a), _checked(g_b)
    if a.shape != b.shape:
        raise ValueError("gradients must share one parameter manifest")
    return float(np.mean(np.sign(a) == np.sign(b)))


def zero_sign_stats(g_a: np.ndarray, g_b: np.ndarray) -> Dict[str, float]:
    """Zero COUNTS in each vector and in their intersection (D09 provenance)."""
    a, b = _checked(g_a), _checked(g_b)
    return {"n_elements": int(a.size), "zero_count_a": int(np.sum(a == 0)),
            "zero_count_b": int(np.sum(b == 0)), "zero_count_both": int(np.sum((a == 0) & (b == 0)))}


def gradient_stats(g_candidate: np.ndarray, g_retention: np.ndarray) -> Dict:
    """All three D09 features plus the provenance to log with them.

    ``g_candidate`` / ``g_retention`` are the flat gradients of the D08 mean
    answer NLL on the candidate batch and on the pinned retention subset, at the
    same parent.  Empty or non-finite input raises.
    """
    cos, flags = cosine_with_diagnostic(g_candidate, g_retention)
    return {"grad_norm": grad_l2_norm(g_candidate), "grad_cosine_retention": cos,
            "sign_agreement": sign_agreement(g_candidate, g_retention),
            "diagnostics": {**flags, **zero_sign_stats(g_candidate, g_retention),
                            "retention_grad_norm": grad_l2_norm(g_retention)}}


def module_energy(grad_a: np.ndarray, grad_b: np.ndarray) -> float:
    """D05: squared gradient norm of ONE adapted module = |gA|^2 + |gB|^2."""
    ga = np.asarray(grad_a, dtype=np.float64).ravel()
    gb = np.asarray(grad_b, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))):
        raise ValueError("non-finite gradient")
    return float(ga @ ga + gb @ gb)


def block_energies(module_energies: Dict[Tuple[int, str], float]) -> np.ndarray:
    """Sum module energies (keys ``(block_index, module_name)``) into one value
    per transformer block, blocks in NUMERIC order (2 before 10).  Blocks must
    be contiguous from 0 so a missing block fails instead of shifting depth."""
    if not module_energies:
        raise ValueError("no modules")
    blocks = sorted({b for b, _ in module_energies})
    if blocks != list(range(len(blocks))):
        raise ValueError("blocks must be contiguous from 0, got %s" % blocks[:10])
    out = np.zeros(len(blocks))
    for (blk, _), e in module_energies.items():
        if e < 0 or not np.isfinite(e):
            raise ValueError("module energy must be finite and non-negative")
        out[blk] += e
    return out


def energy_fractions_from_blocks(block_energy: Sequence[float], num_bins: int) -> np.ndarray:
    """Per-block energies -> ``layer_depth_bin_*`` (sums to 1)."""
    e = np.asarray(block_energy, dtype=np.float64)
    total = e.sum()
    if total <= 0:
        return np.full(num_bins, 1.0 / num_bins)
    return depth_bin_pool(e, num_bins, reduce="sum") / total


# --------------------------------------------------------------------------
# Short-probe dynamics
# --------------------------------------------------------------------------
def linear_slope(y: Sequence[float], x: Optional[Sequence[float]] = None) -> float:
    """OLS slope of y against optimizer-step index x (default 0..n-1).

    Recommended cadence: training loss at every probe step (1..15);
    anchor/validation loss at steps 0, 5, 10, 15 (step 0 included).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if y.size < 2 or x.shape != y.shape:
        raise ValueError("need >= 2 matching samples")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("non-finite samples")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        raise ValueError("x has no spread")
    return float(np.dot(xc, y - y.mean()) / denom)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl_drift_anchor(logits_pre: np.ndarray, logits_post: np.ndarray, answer_ids: Sequence[int],
                    valid_mask: Optional[Sequence[bool]] = None) -> float:
    """D12 ``kl_drift_anchor``: KL(p_parent || p_post-probe), full vocabulary, natural log.

    Teacher-forced on fixed anchor contexts.  Per-position KL is averaged over
    the VALID answer positions WITHIN each answer, then those per-answer means
    are averaged with equal weight (a long answer does not dominate).
    ``logits_*`` are [positions, vocab]; ``answer_ids[i]`` names the answer of
    position i; ``valid_mask`` is 1 for answer positions (excludes prompt,
    padding, special tokens).  An answer with no valid position, or no answer at
    all, raises.
    """
    a = np.asarray(logits_pre, dtype=np.float64)
    b = np.asarray(logits_post, dtype=np.float64)
    ids = np.asarray(answer_ids)
    if a.shape != b.shape or a.ndim != 2 or ids.shape != (a.shape[0],):
        raise ValueError("need matching [positions, vocab] logits and one answer id per position")
    valid = np.ones(a.shape[0], bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != ids.shape or not valid.any():
        raise ValueError("valid_mask must select at least one position")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite logits")
    lp, lq = _log_softmax(a), _log_softmax(b)
    kl = np.clip((np.exp(lp) * (lp - lq)).sum(axis=-1), 0.0, None)
    per_answer = []
    for aid in np.unique(ids):
        m = (ids == aid) & valid
        if not m.any():
            raise ValueError("answer %r has no valid position" % (aid,))
        per_answer.append(kl[m].mean())
    return float(np.mean(per_answer))


def probe_loss(per_answer_token_nlls: Sequence[Sequence[float]]) -> float:
    """Every probe loss (train steps 1..15; anchor and target at 0/5/10/15) uses the
    D08 mean-of-per-answer-means NLL, so slopes are comparable.  The anchor and
    target subsets must be the SAME fixed examples at every checkpoint."""
    return mean_answer_nll(per_answer_token_nlls)


def param_delta_norm(before: np.ndarray, after: np.ndarray) -> float:
    """``lora_delta_norm``: L2 norm of (after - before) over the flattened A and
    B parameters (parameter space, not the effective scaled B@A)."""
    a = np.asarray(before, dtype=np.float64).ravel()
    b = np.asarray(after, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("snapshots must have the same parameter manifest")
    return float(np.linalg.norm(b - a))
