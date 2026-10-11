"""Actual tensor/attempt reductions, opt-in proposed definitions (not approvals).

No loss is converted to accuracy, no empty span is filled, no future label is
accepted as a feature. Callers pin span masks, data identities and definition IDs.
"""
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F

from .feature_contract import ContractError
from .forecasting.features import linear_slope
from .generation_calibration import final_answer, verify_answer
from .sprint2_runtime import TrainingSnapshot, restored_training_state, seeded_stream


def finite(values, *, ndim=None):
    x = np.asarray(values, np.float64)
    if not x.size or not np.isfinite(x).all() or (ndim is not None and x.ndim != ndim):
        raise ContractError("Nonempty finite measurements required")
    return x


def score_completions(rows, rule):
    """All attempts in denominator; missing parses and truncated answers are wrong."""
    if not rows or len({r["prompt_id"] for r in rows}) != len(rows):
        raise ContractError("Unique nonempty evaluation population required")
    results = []
    for row in rows:
        if type(row["truncated"]) is not bool or type(row["at_eos"]) is not bool:
            raise ContractError("Explicit completion stopping metadata required")
        count, cap = row["generated_tokens"], row["cap"]
        if type(count) is not int or type(cap) is not int or not 0 <= count <= cap or cap < 1:
            raise ContractError("Invalid measured generation length/cap")
        parsed = final_answer(row["text"], rule, at_eos=row["at_eos"])
        correct = verify_answer(parsed, row["reference"], rule) and not row["truncated"]
        if row["truncated"] and (row["at_eos"] or count != cap or parsed is not None):
            raise ContractError("Inconsistent truncation metadata")
        results.append({**row, "parsed": parsed, "correct": bool(correct), "parse_failed": parsed is None})
    return {"accuracy": sum(r["correct"] for r in results)/len(results),
            "correct": sum(r["correct"] for r in results), "total": len(results),
            "parse_failures": sum(r["parse_failed"] for r in results),
            "truncated": sum(r["truncated"] for r in results),
            "truncation_rate": sum(r["truncated"] for r in results)/len(results),
            "scorer": rule, "attempts": results}


def answer_logprobs(logits, input_ids, answer_mask, attention_mask):
    """Causal alignment: logit[t-1] predicts selected answer token[t]."""
    if (logits.ndim != 3 or input_ids.shape != logits.shape[:2]
            or answer_mask.dtype != torch.bool or answer_mask.shape != input_ids.shape
            or attention_mask.shape != input_ids.shape
            or bool(answer_mask[:, 0].any())
            or bool((answer_mask & ~attention_mask.bool()).any())
            or not bool(torch.isfinite(logits).all())):
        raise ContractError("Aligned finite logits, causal answer masks and attention required")
    selected = answer_mask[:, 1:]
    if not bool(selected.any(1).all()):
        raise ContractError("Each example needs an observed answer span")
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    target = logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return [target[i][selected[i]] for i in range(len(input_ids))]


def answer_nll(logprobs, *, reduction):
    if not logprobs or any(not p.numel() or not bool(torch.isfinite(p).all()) for p in logprobs):
        raise ContractError("Nonempty finite answer log probabilities required")
    if reduction == "equal_answer_v1":
        return -torch.stack([p.mean() for p in logprobs]).mean()
    if reduction == "token_weighted_v1":
        return -torch.cat(logprobs).mean()
    raise ContractError("Bind equal-answer or token-weighted NLL explicitly")


def duplicate_components(texts, *, threshold, tolerance):
    """D07 proposal; exact threshold/tolerance must be explicitly bound."""
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ContractError("Nonempty texts required")
    if not 0 < threshold <= 1 or not 0 <= tolerance < threshold:
        raise ContractError("Explicit duplicate threshold/tolerance required")
    tokens = [unicodedata.normalize("NFC", t).split() for t in texts]
    grams = [{tuple(t[i:i+3]) for i in range(len(t)-2)} for t in tokens]
    parent = list(range(len(texts)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    edges = []
    for i in range(len(texts)):
        for j in range(i):
            same = (tokens[i] == tokens[j] if min(len(tokens[i]), len(tokens[j])) < 3 else
                    len(grams[i] & grams[j])/len(grams[i] | grams[j]) >= threshold-tolerance)
            if same:
                edges.append([j, i])
                parent[root(i)] = root(j)
    components = [[i for i in range(len(texts)) if root(i) == r]
                  for r in sorted({root(i) for i in range(len(texts))})]
    return {"rate": (len(texts)-len(components))/len(texts), "components": components, "edges": edges}


def candidate_data(*, logprobs, lengths, correction_passes, original_passes,
                   embeddings, texts, definitions):
    nll = float(answer_nll(logprobs, reduction=definitions["nll_reduction"]).detach())
    means = finite([float(p.detach().mean()) for p in logprobs], ndim=1)
    length = finite(lengths, ndim=1)
    if len(length) != len(logprobs) or len(texts) != len(length) or np.any(length <= 0):
        raise ContractError("Candidate length/span/text populations must align")
    for attempts in (correction_passes, original_passes):
        if not attempts or any(type(v) is not bool for v in attempts):
            raise ContractError("All original/correction attempts required, not only successes")
    emb = finite(embeddings, ndim=2)
    norm = np.linalg.norm(emb, axis=1)
    if emb.shape[0] != len(texts) or np.any(norm == 0):
        raise ContractError("Aligned nonzero frozen-base embeddings required")
    units = emb/norm[:, None]
    pairs = [1-float(np.sum(units[i]*units[j])) for i in range(len(units)) for j in range(i)]
    duplicate = duplicate_components(texts, threshold=definitions["duplicate_threshold"],
                                     tolerance=definitions["duplicate_tolerance"])
    return {"token_length_mean": float(length.mean()), "token_length_std": float(length.std()),
            "pre_update_nll": nll, "mean_confidence": float(means.mean()),
            "verifier_pass_rate": sum(correction_passes)/len(correction_passes),
            "difficulty_proxy": 1-sum(original_passes)/len(original_passes),
            "embedding_diversity": float(np.mean(pairs)) if pairs else 0.0,
            "duplicate_rate": duplicate["rate"]}, {"duplicates": duplicate,
            "correction_passes": correction_passes, "original_passes": original_passes,
            "token_lengths": length.tolist(), "answer_token_counts": [len(p) for p in logprobs]}


def lora_parameters(model, manifest):
    """Explicit qualified parameter names/shapes; no model-depth/name inference."""
    params = dict(model.named_parameters())
    names, ordered = [], []
    for module in manifest:
        for side in ("a", "b"):
            name, shape = module[side + "_name"], tuple(module[side + "_shape"])
            if name in names or name not in params or tuple(params[name].shape) != shape or not params[name].requires_grad:
                raise ContractError("Actual A/B parameter name/shape/order differs from explicit manifest")
            names.append(name)
            ordered.append(params[name])
    if not ordered or set(names) != {n for n, p in params.items() if p.requires_grad}:
        raise ContractError("Manifest must cover exactly all trainable LoRA A/B tensors")
    return ordered


def gradient_measurements(model, manifest, candidate_loss, retention_loss, *, zero_epsilon):
    """autograd.grad in eval mode; unscaled/unclipped, exact ordered A/B pairs."""
    if not np.isfinite(zero_epsilon) or zero_epsilon < 0:
        raise ContractError("Explicit cosine zero threshold required")
    params = lora_parameters(model, manifest)
    snapshot = TrainingSnapshot(model)
    try:
        model.eval()
        def gradients(fn):
            result = fn()
            terms = [(result, 1.0)] if isinstance(result, torch.Tensor) else result
            total, mass = None, 0.0
            for loss, weight in terms:
                if loss.ndim or not bool(torch.isfinite(loss)) or not np.isfinite(weight) or weight <= 0:
                    raise ContractError("Finite scalar loss and positive declared reduction weight required")
                values = torch.autograd.grad(loss, params, allow_unused=True)
                if any(g is None for g in values):
                    raise ContractError("Missing aligned LoRA gradient; never fill it with zeros")
                arrays = [finite(g.detach().float().cpu().numpy().reshape(-1), ndim=1)*weight for g in values]
                total = arrays if total is None else [a+b for a, b in zip(total, arrays)]
                mass += weight
            if total is None or not np.isclose(mass, 1.0, rtol=0, atol=1e-12):
                raise ContractError("Empty/incomplete gradient population or invalid reduction weights")
            return total
        a, b = gradients(candidate_loss), gradients(retention_loss)
    finally:
        snapshot.restore()
    ga, gb = np.concatenate(a), np.concatenate(b)
    na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
    norms = [float(np.sqrt(np.sum(a[i]**2)+np.sum(a[i+1]**2))) for i in range(0, len(a), 2)]
    stats = {"grad_norm": float(na), "layerwise_grad_norms": norms,
             "grad_cosine_retention": 0.0 if na <= zero_epsilon or nb <= zero_epsilon else float(np.sum(ga*gb)/(na*nb)),
             "sign_agreement": float(np.mean(np.sign(ga) == np.sign(gb)))}
    diagnostics = {"candidate_zero_norm": bool(na == 0), "retention_zero_norm": bool(nb == 0),
                   "zero_in_a": float(np.mean(ga == 0)), "zero_in_b": float(np.mean(gb == 0)),
                   "zero_in_both": float(np.mean((ga == 0) & (gb == 0)))}
    finite([stats[k] for k in stats if k != "layerwise_grad_norms"])
    return stats, diagnostics


def probe_measurements(model, manifest, *, train_loss, anchor_loss, val_loss, anchor_logits,
                       steps, cadence, optimizer_config, seed, stateful=(), generators=(), train_eval_loss=None):
    """Measured temporary trajectory, finally restores all explicitly owned state.

    Callbacks consume already pinned answer-masked populations. Full-vocabulary
    anchor_logits returns only valid causal answer positions, shape [tokens,vocab].
    """
    if (type(steps) is not int or steps < 2 or not cadence or cadence[0] != 0
            or cadence[-1] != steps or list(cadence) != sorted(set(cadence))
            or any(type(i) is not int or not 0 <= i <= steps for i in cadence)):
        raise ContractError("Explicit probe steps and endpoint-inclusive cadence required")
    params = lora_parameters(model, manifest)
    traces = {"train": [], "anchor": [], "val": [], "optimizer_step_loss": []}
    train_eval_loss = train_eval_loss or (lambda: train_loss(0))
    with restored_training_state(model, stateful=stateful, generators=generators) as snapshot:
        with seeded_stream(seed):
            before = torch.cat([p.detach().float().cpu().reshape(-1) for p in params])
            model.eval()
            with torch.no_grad():
                lp = F.log_softmax(anchor_logits().float(), -1).cpu()
                traces["anchor"].append(float(anchor_loss()))
                traces["val"].append(float(val_loss()))
                traces["train"].append(float(train_eval_loss()))
            if lp.ndim != 2 or not lp.numel() or not bool(torch.isfinite(lp).all()):
                raise ContractError("Finite nonempty anchor answer logits required")
            optimizer = torch.optim.AdamW(params, **optimizer_config)
            for step in range(1, steps+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = train_loss(step)
                if loss.ndim or not bool(torch.isfinite(loss)):
                    raise ContractError("Nonfinite probe loss")
                traces["optimizer_step_loss"].append(float(loss.detach()))
                loss.backward()
                optimizer.step()
                if step in cadence:
                    model.eval()
                    with torch.no_grad():
                        traces["anchor"].append(float(anchor_loss()))
                        traces["val"].append(float(val_loss()))
                        traces["train"].append(float(train_eval_loss()))
            model.eval()
            with torch.no_grad():
                lq = F.log_softmax(anchor_logits().float(), -1).cpu()
            if lq.shape != lp.shape or not bool(torch.isfinite(lq).all()):
                raise ContractError("Probe anchor positions/vocabulary changed")
            kl = float((lp.exp()*(lp-lq)).sum(-1).double().mean())
            after = torch.cat([p.detach().float().cpu().reshape(-1) for p in params])
            values = {"train_loss_slope": linear_slope(traces["train"], cadence),
                      "anchor_loss_slope": linear_slope(traces["anchor"], cadence),
                      "val_loss_slope": linear_slope(traces["val"], cadence),
                      "kl_drift_anchor": max(0.0, kl),
                      "lora_delta_norm": float(torch.linalg.vector_norm(after-before))}
            finite(list(values.values()))
    return values, {"traces": traces, "cadence": list(cadence), "restoration_sha256": snapshot.before,
                    "restoration_verified": True, "temporary_probe_not_h1_or_h2": True}
