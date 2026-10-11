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


def token_weighted_nll(nll_sums: Sequence[float], token_counts: Sequence[int]) -> float:
    """``pre_update_nll``: sum of token NLLs / sum of tokens, over answer tokens.

    Token-weighted across the batch (not a mean of per-example means).
    """
    s = np.asarray(nll_sums, dtype=np.float64)
    n = np.asarray(token_counts, dtype=np.float64)
    if s.shape != n.shape or n.sum() <= 0:
        raise ValueError("need matching, non-empty arrays with >0 tokens")
    return float(s.sum() / n.sum())


def sequence_confidence(per_example_token_logprobs: Sequence[Sequence[float]]) -> List[float]:
    """(mean, population std) over examples of the mean answer-token log-prob.

    Used for ``mean_confidence`` (the batch's own corrected answers) and for
    ``gen_confidence_mean/std`` (a probe set).  Values are <= 0; higher means
    more confident.  Prompt tokens must already be excluded.
    """
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


def difficulty_proxy(per_prompt_attempt_passes: Sequence[Sequence[bool]]) -> float:
    """Mean over prompts of (1 - pass rate of the *current* model M_t).

    Needs several sampled attempts per training prompt (>= 4 recommended).
    With a single greedy attempt on prompts that were selected *because* they
    failed, every value is 1.0 and the feature is constant.
    """
    rates = []
    for attempts in per_prompt_attempt_passes:
        a = np.asarray(attempts, dtype=bool)
        if a.size == 0:
            raise ValueError("a prompt has no attempts")
        rates.append(1.0 - a.mean())
    return float(np.mean(rates))


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
                same = union > 0 and len(grams[i] & grams[j]) / union >= threshold - 1e-12
            if same:
                parent[find(i)] = find(j)
    return float((n - len({find(i) for i in range(n)})) / n)


# --------------------------------------------------------------------------
# Gradient / interference
# --------------------------------------------------------------------------
def grad_l2_norm(flat_grad: np.ndarray) -> float:
    """L2 norm of the flattened LoRA gradient (A and B), token-mean CE loss,
    after AMP unscaling, before clipping."""
    g = np.asarray(flat_grad, dtype=np.float64).ravel()
    if not np.all(np.isfinite(g)):
        raise ValueError("non-finite gradient")
    return float(np.linalg.norm(g))


def grad_cosine(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """Cosine of two flat gradients over the same parameter manifest.  A zero
    vector gives 0.0 (no evidence of alignment)."""
    a = np.asarray(g_a, dtype=np.float64).ravel()
    b = np.asarray(g_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("gradients must have the same parameter manifest")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sign_agreement(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """D09: fraction of ALL aligned elements with equal sign, sign(0) = 0.

    Two zeros therefore agree.  Always log ``zero_sign_stats`` next to it: zero
    gradients (e.g. LoRA A-matrices while B is still 0) inflate agreement.
    """
    a = np.asarray(g_a, dtype=np.float64).ravel()
    b = np.asarray(g_b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("gradients must share one non-empty parameter manifest")
    return float(np.mean(np.sign(a) == np.sign(b)))


def zero_sign_stats(g_a: np.ndarray, g_b: np.ndarray) -> Dict[str, float]:
    """Provenance diagnostic for ``sign_agreement``: share of zero elements."""
    a = np.asarray(g_a, dtype=np.float64).ravel()
    b = np.asarray(g_b, dtype=np.float64).ravel()
    return {"zero_in_a": float(np.mean(a == 0)), "zero_in_b": float(np.mean(b == 0)),
            "zero_in_both": float(np.mean((a == 0) & (b == 0)))}


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


def mean_token_kl(logits_pre: np.ndarray, logits_post: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """``kl_drift_anchor``: mean over valid tokens of KL(p_pre || p_post).

    Teacher-forced on fixed anchor sequences, full vocabulary, natural log.
    ``logits_*`` are [tokens, vocab] (flatten batch x time first); ``mask`` is
    [tokens] with 1 for answer tokens, so padding never contributes.
    """
    a = np.asarray(logits_pre, dtype=np.float64)
    b = np.asarray(logits_post, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("logits must be matching [tokens, vocab] arrays")
    m = np.ones(a.shape[0], dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    if m.shape != (a.shape[0],) or not m.any():
        raise ValueError("mask must select at least one token")
    lp, lq = _log_softmax(a[m]), _log_softmax(b[m])
    kl = (np.exp(lp) * (lp - lq)).sum(axis=-1)
    return float(np.clip(kl, 0.0, None).mean())


def param_delta_norm(before: np.ndarray, after: np.ndarray) -> float:
    """``lora_delta_norm``: L2 norm of (after - before) over the flattened A and
    B parameters (parameter space, not the effective scaled B@A)."""
    a = np.asarray(before, dtype=np.float64).ravel()
    b = np.asarray(after, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("snapshots must have the same parameter manifest")
    return float(np.linalg.norm(b - a))
