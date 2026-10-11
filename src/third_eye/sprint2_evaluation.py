"""Opt-in accepted D13; no legacy metric or normalization definitions changed."""
import numpy as np

from .feature_contract import ContractError
from .sprint2_protocol import CandidateKey, group_macro_weights, prediction_errors
from .forecasting.data import Standardizer


def keys(ds):
    return [CandidateKey(str(g), str(t), int(s), int(c)) for g, t, s, c in
            zip(ds.group, ds.trajectory, ds.state_id, ds.candidate_id)]


def weights(ds):
    return group_macro_weights(keys(ds))


def weighted_stats(values, row_weights, observed=None):
    x = np.asarray(values, np.float64)
    mask = np.ones(x.shape, bool) if observed is None else np.asarray(observed, bool)
    rw = np.asarray(row_weights, np.float64)
    if (x.ndim != 2 or mask.shape != x.shape or not np.isfinite(x).all()
            or rw.shape != (len(x),) or not np.isfinite(rw).all() or np.any(rw < 0) or rw.sum() <= 0):
        raise ContractError("Finite aligned normalization values required")
    effective = rw[:, None] * mask
    denom = effective.sum(0)
    origin = np.array([x[np.flatnonzero(mask[:, i])[0], i] if mask[:, i].any() else 0
                       for i in range(x.shape[1])])
    mean = origin + np.divide((effective*(x-origin)).sum(0), denom,
                              out=np.zeros(x.shape[1]), where=denom > 0)
    var = np.divide((effective*(x-mean)**2).sum(0), denom, out=np.zeros(x.shape[1]), where=denom > 0)
    scale = np.sqrt(var)
    scale[scale == 0] = 1
    return mean, scale


class Sprint2Standardizer(Standardizer):
    """Train-only nested weights; each row shares its weight across observed slots.

    Targets stay native. No held-out/transductive or per-backbone statistic fit.
    An entirely unobserved history channel has mean 0/scale 1, not a measurement.
    """
    def fit(self, ds):
        if self.per_model or not ds.semantic_hash:
            raise ContractError("D13 needs a definition hash and no per-model/transductive normalization")
        w = weights(ds)
        st = {"state": weighted_stats(ds.state, w), "cand": weighted_stats(ds.cand, w)}
        if ds.hist.shape[2]:
            mask = ds.hist_mask.astype(bool)
            if np.any(ds.hist[~mask] != 0):
                raise ContractError("History padding must be zero")
            hw = w[:, None] / np.maximum(mask.sum(1)[:, None], 1)
            hw = np.broadcast_to(hw, mask.shape).ravel()
            observed = np.broadcast_to(mask[..., None], ds.hist.shape).reshape(-1, ds.hist.shape[2])
            st["hist"] = weighted_stats(ds.hist.reshape(-1, ds.hist.shape[2]), hw, observed)
        self.stats = {"global": st,
                      "y2": (np.zeros(3), np.ones(3)),
                      "y1": (np.zeros(ds.y_h1.shape[1]), np.ones(ds.y_h1.shape[1]))}
        return self

    def add_model_stats(self, ds):
        raise ContractError("Held-out normalization is forbidden in D13")

    def to_dict(self):
        return {**super().to_dict(), "protocol": "sprint2_d13"}


def constant_heads(train):
    if not np.isfinite(train.y_h2).all() or not np.isfinite(train.y_h1).all():
        raise ContractError("Complete native targets required")
    w = weights(train)
    return (np.sum(w[:, None]*train.y_h2, axis=0),
            np.sum(w[:, None]*train.y_h1, axis=0))


def policy_report(ds, scores, *, bootstrap=500, seed=0):
    scores = np.asarray(scores, np.float64)
    if scores.shape != (len(ds),) or not np.isfinite(scores).all():
        raise ContractError("Finite ranking scores required")
    w, utilities = weights(ds), ds.lhv2().astype(np.float64)
    group_rows = {str(g): [] for g in np.unique(ds.group)}
    for state in np.unique(ds.state_index):
        idx = np.flatnonzero(ds.state_index == state)
        if len(idx) != 3:
            raise ContractError("Policy report requires complete K=3 states")
        tied = scores[idx] == scores[idx].max()
        best = utilities[idx] == utilities[idx].max()
        row = {"regret": float(utilities[idx].max()-utilities[idx][tied].mean()),
               "selected_utility": float(utilities[idx][tied].mean()),
               "top1": float(np.mean(best[tied])), "tie_size": float(tied.sum()),
               "tie_rate": float(tied.sum() > 1)}
        row.update({"selected_%s_h2_delta" % name: float(ds.y_h2[idx][tied, channel].mean())
                    for channel, name in enumerate(("target", "ood", "retention"))})
        group_rows[str(ds.group[idx[0]])].append((w[idx].sum(), row))
    per_group = {}
    for group, rows in group_rows.items():
        total = sum(weight for weight, _ in rows)
        per_group[group] = {k: sum(weight*row[k] for weight, row in rows)/total
                            for k in rows[0][1]}
    names = list(next(iter(per_group.values())))
    matrix = np.array([[row[k] for k in names] for row in per_group.values()])
    report = {k: float(matrix[:, i].mean()) for i, k in enumerate(names)}
    report.update(n_groups=len(matrix), n_states=ds.n_states, n_records=len(ds),
                  per_group=per_group, tie_rule="exact uniform, enumerated expectation",
                  uncertainty_unit="lineage_group", preliminary=True)
    if bootstrap < 0:
        raise ContractError("Nonnegative bootstrap count required")
    if bootstrap and len(matrix) > 1:
        rng = np.random.default_rng(seed)
        draws = matrix[rng.integers(0, len(matrix), size=(bootstrap, len(matrix)))].mean(1)
        report["ci95"] = {k: np.quantile(draws[:, i], [.025, .975]).tolist()
                          for i, k in enumerate(names)}
    else:
        report["ci95"] = None
    report["uncertainty_warning"] = "Few lineage groups: intervals are preliminary, not candidate-row precision."
    return report


def checkpoint_score(ds, prediction):
    # One-step scores rank H2 utility; its H1 vectors are explicitly an H2 proxy
    # for this common three-channel native-MSE secondary criterion, not H2 heads.
    native = prediction.h2 if prediction.h2 is not None else prediction.h1
    regret = policy_report(ds, prediction.score(), bootstrap=0)["regret"]
    mse = float(prediction_errors(native, ds.y_h2, keys(ds))["mse"].mean())
    return regret, mse


def baseline_tables(train, test, predictions, *, bootstrap=500):
    base2, base1 = constant_heads(train)
    candidates = test.layout.columns("candidate")
    nll = test.cand[:, candidates.index("pre_update_nll")]
    scores = {"constant": np.zeros(len(test)), "uniform_random": np.zeros(len(test)),
              "highest_pre_update_nll": nll,
              **{name: pred.score() for name, pred in predictions.items()}}
    policies = {name: policy_report(test, value, bootstrap=bootstrap) for name, value in scores.items()}
    errors = {}
    for name, vector in {"constant": np.tile(base2, (len(test), 1)),
                         **{name: pred.h2 for name, pred in predictions.items() if pred.h2 is not None}}.items():
        result = prediction_errors(vector, test.y_h2, keys(test))
        errors[name] = {head: {metric: float(values[i]) for metric, values in result.items()}
                        for i, head in enumerate(("target", "ood", "retention"))}
        errors[name]["per_group"] = {
            str(g): {m: v.tolist() for m, v in prediction_errors(
                vector[test.group == g], test.y_h2[test.group == g],
                keys(test.subset(np.flatnonzero(test.group == g)))).items()}
            for g in np.unique(test.group)}
        group_mse = np.array([r["mse"] for r in errors[name]["per_group"].values()])
        group_mae = np.array([r["mae"] for r in errors[name]["per_group"].values()])
        if bootstrap and len(group_mse) > 1:
            draw = np.random.default_rng(0).integers(0, len(group_mse), size=(bootstrap, len(group_mse)))
            draws = {"mse": group_mse[draw].mean(1), "mae": group_mae[draw].mean(1)}
            draws["rmse"] = np.sqrt(draws["mse"])
            errors[name]["ci95_by_lineage_group"] = {m: np.quantile(x, [.025, .975], axis=0).tolist()
                                                     for m, x in draws.items()}
        else:
            errors[name]["ci95_by_lineage_group"] = None
    h1_errors = {name: {m: x.tolist() for m, x in prediction_errors(
        pred.h1, test.y_h1, keys(test)).items()} for name, pred in predictions.items() if pred.h1 is not None}
    paired = {}
    for name, policy in policies.items():
        if name == "constant":
            continue
        difference = np.array([policy["per_group"][g]["regret"]-policies["constant"]["per_group"][g]["regret"]
                               for g in policies["constant"]["per_group"]])
        ci = None
        if bootstrap and len(difference) > 1:
            draw = np.random.default_rng(0).integers(0, len(difference), size=(bootstrap, len(difference)))
            ci = np.quantile(difference[draw].mean(1), [.025, .975]).tolist()
        paired[name] = {"regret_minus_constant": float(difference.mean()), "paired_group_ci95": ci}
    against_baselines = {}
    for baseline in ("constant", "uniform_random", "highest_pre_update_nll"):
        for name, policy in policies.items():
            if name == baseline:
                continue
            difference = np.array([policy["per_group"][g]["regret"]-policies[baseline]["per_group"][g]["regret"]
                                   for g in policies[baseline]["per_group"]])
            ci = None
            if bootstrap and len(difference) > 1:
                draw = np.random.default_rng(0).integers(0, len(difference), size=(bootstrap, len(difference)))
                ci = np.quantile(difference[draw].mean(1), [.025, .975]).tolist()
            against_baselines[name+"_minus_"+baseline] = {"regret_difference": float(difference.mean()), "paired_group_ci95": ci}
    return {"policy_table": policies, "prediction_errors_native": errors,
            "prediction_errors_h1_native": h1_errors, "paired_comparisons": paired,
            "paired_against_each_baseline": against_baselines,
            "constant_h2_train_mean": base2.tolist(), "constant_h1_train_mean": base1.tolist(),
            "ranking_only_baselines_have_no_prediction_error": ["uniform_random", "highest_pre_update_nll"]}
