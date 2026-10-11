"""Evaluate H=1 / Direct / Dynamics forecasters on a meta-dataset (E2, E3, E8, E9, E12, E13).

Usage
-----
Validate Shashank's records first (no training, instant):

    python -m scripts.run_forecasting_eval --records meta.jsonl --validate-only

Full evaluation (held-out TRAJECTORIES; writes metrics.json + policy_table.csv):

    python -m scripts.run_forecasting_eval --records meta.jsonl --out outputs/forecasting_eval/run1

Dry run on planted-signal synthetic data (checks the pipeline only; the numbers
mean nothing about the real study and the output is stamped synthetic):

    python -m scripts.run_forecasting_eval --synthetic --out /tmp/synthetic_run

Add --ablate to run feature-group ablations (E8).  Training is tiny (<1M params)
and runs on CPU; do not submit it to a GPU queue.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.third_eye.forecasting import metrics as M
from src.third_eye.forecasting import selection as S
from src.third_eye.forecasting.data import fixed_split, from_records, read_jsonl, validate_records
from src.third_eye.forecasting.layout import contract_layout, legacy_layout
from src.third_eye.forecasting.synthetic import make_synthetic_records
from src.third_eye.forecasting.train import TrainConfig, cross_validate, fit, predict


def _layout(name, bins, h1):
    heads = ("target", "ood", "retention") if h1 == "all" else ("target",)
    return legacy_layout() if name == "legacy" else contract_layout(bins, h1_heads=heads)


def _report(ds, pred, boot, cluster=None):
    y = ds.lhv2()
    s = pred.score()
    rep = M.ranking_report(y, s, ds.state_index)
    ci = {"rho": M.bootstrap_ci(M.m_within_state_spearman, y, s, ds.state_index, n_boot=boot,
                                cluster_index=cluster),
          "top1": M.bootstrap_ci(M.m_top1, y, s, ds.state_index, n_boot=boot,
                                 cluster_index=cluster)}
    out = {"ranking_vs_lhv2": rep, "ci": ci}
    if cluster is not None:
        out["per_group"] = M.per_group_report(y, s, ds.state_index, cluster)
    if pred.h2 is not None:
        out["per_head"] = M.regression_report(ds.y_h2, pred.h2, ["target", "ood", "retention"],
                                              ds.state_index)
    return out


def _run_manifest(args, ds, cluster, report):
    """D02: fixed lineage-group split (e.g. 8/2/2).  One train->test evaluation per
    model, always labelled preliminary; the constant baseline is the group-weighted
    training mean (D13)."""
    manifest = json.loads(Path(args.split_manifest).read_text())
    idx = fixed_split(ds, manifest)
    train, test = ds.subset(idx["train"]), ds.subset(idx["test"])
    val = ds.subset(idx["val"]) if "val" in idx else None
    base = M.group_weighted_mean(train.y_h2, train.group)
    results = {"preliminary": True, "synthetic": bool(args.synthetic), "validation": report.summary(),
               "split_sizes_rows": {k: int(len(v)) for k, v in idx.items()},
               "test_states": test.n_states, "test_groups": int(len(np.unique(test.group))),
               "history_rule": args.history_rule, "semantic_hash": ds.semantic_hash, "models": {}}
    for kind in args.models.split(","):
        res = fit(train, val, TrainConfig(kind=kind, rank_weight=args.rank_weight))
        pred = predict(res, test)
        y, sc = test.lhv2(), pred.score()
        results["models"][kind] = {
            "ranking_vs_lhv2": M.ranking_report(y, sc, test.state_index),
            "ci": {"rho": M.bootstrap_ci(M.m_within_state_spearman, y, sc, test.state_index,
                                         n_boot=args.boot, cluster_index=test.group),
                   "top1": M.bootstrap_ci(M.m_top1, y, sc, test.state_index,
                                          n_boot=args.boot, cluster_index=test.group)},
            "per_group": M.per_group_report(y, sc, test.state_index, test.group)}
        if pred.h2 is not None:
            results["models"][kind]["per_head_vs_constant"] = M.regression_report(
                test.y_h2, pred.h2, ["target", "ood", "retention"], test.state_index, baseline=base)
        print(kind, json.dumps(results["models"][kind]["ranking_vs_lhv2"]))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics_manifest_split.json").write_text(json.dumps(results, indent=2, default=float))
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--records")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layout", default="contract", choices=["contract", "legacy"])
    ap.add_argument("--depth-bins", type=int, default=6)
    ap.add_argument("--h1-heads", default="all", choices=["all", "target"])
    ap.add_argument("--models", default="one_step,direct,dynamics")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--relative", action="store_true", help="add sibling-relative candidate features")
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--semantic-hash", default=None,
                    help="D15: reject records whose semantic_hash differs (get it from print_semantic_spec)")
    ap.add_argument("--history-rule", default="applied_h1", choices=["applied_h1", "applied_h2"],
                    help="D10: which past results count as history (pending team decision)")
    ap.add_argument("--split-manifest", default=None,
                    help="JSON {lineage_group: train|val|test} (D02); evaluates train->test once, "
                         "instead of K-fold CV. Results are labelled preliminary.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layout = _layout(args.layout, args.depth_bins, args.h1_heads)
    records = (make_synthetic_records(layout, n_trajectories=60, states_per_trajectory=5,
                                      n_groups=12, history_rule=args.history_rule)
               if args.synthetic else read_jsonl(args.records))
    report = validate_records(records, layout, history_rule=args.history_rule,
                              expected_semantic_hash=args.semantic_hash)
    print("validation:", report.summary())
    for w in report.warnings:
        print("  WARNING:", w)
    if args.validate_only:
        return
    ds = from_records(records, layout, validate=False)
    cluster = ds.group if len(np.unique(ds.group)) > 1 else None
    if args.relative:
        ds = ds.with_relative_candidates()

    if args.split_manifest:
        return _run_manifest(args, ds, cluster, report)

    t0 = time.time()
    seeds = [int(x) for x in args.seeds.split(",")]
    results = {"synthetic": bool(args.synthetic), "history_rule": args.history_rule,
               "semantic_hash": ds.semantic_hash, "layout": ds.layout.name,
               "layout_fingerprint": ds.layout.fingerprint(), "validation": report.summary(),
               "n_rows": len(ds), "n_states": ds.n_states, "gate1": S.gate1(ds), "models": {}}
    preds = {}
    for kind in args.models.split(","):
        cfg = TrainConfig(kind=kind, rank_weight=args.rank_weight)
        cv = cross_validate(ds, cfg, n_splits=args.folds, seeds=seeds,
                            by="group" if cluster is not None else "trajectory")
        preds[kind] = cv.pred
        results["models"][kind] = _report(ds, cv.pred, args.boot, cluster)
        results["models"][kind]["n_params"] = cv.fits[0].n_params
        print(kind, json.dumps(results["models"][kind]["ranking_vs_lhv2"]))

    if "direct" in preds and "one_step" in preds:
        y = ds.lhv2()
        results["E3_direct_minus_one_step_within_state_spearman"] = M.paired_bootstrap_diff(
            M.m_within_state_spearman, y, preds["direct"].score(), preds["one_step"].score(),
            ds.state_index, n_boot=args.boot, cluster_index=cluster)
    if "direct" in preds:
        r = results["models"]["direct"]
        results["gate2"] = S.gate2(r["ranking_vs_lhv2"], r["ci"])
    if "direct" in preds and "dynamics" in preds:
        results["dynamics_gate"] = S.dynamics_gate(
            results["models"]["direct"]["ranking_vs_lhv2"], results["models"]["dynamics"]["ranking_vs_lhv2"])
    if "direct" in preds and "per_head" in results["models"]["direct"]:
        ret = results["models"]["direct"]["per_head"]["retention"]
        g2 = results.get("gate2", {}).get("passed_point", False)
        results["pd_activation"] = S.pd_activation_gate(
            g2, ret["sign_accuracy"], ret.get("within_state_spearman", float("nan")),
            S.tradeoff_fraction(ds))

    results["constant_baseline_note"] = (
        "ranking by a constant ties every candidate: top-1 equals chance by construction; "
        "compare per-head MAE/skill against the group-weighted training mean")
    scores = {"random": S.random_scores(ds), "greedy_immediate_oracle": S.greedy_immediate_oracle_scores(ds),
              "future_oracle (upper bound)": S.future_oracle_scores(ds)}
    for name in ("verifier_pass_rate",):
        try:
            scores["heuristic_" + name] = S.heuristic_scores(ds, name)
        except KeyError:
            pass
    for kind, p in preds.items():
        scores["forecaster_" + kind] = p.score()
    policy_table = S.compare_policies(ds, scores, n_boot=args.boot)

    if args.ablate and "direct" in preds:
        results["ablation_E8_direct"] = {}
        for g in [g.name for g in ds.layout.groups]:
            ab = ds.drop_groups(g)
            cv = cross_validate(ab, TrainConfig(kind="direct", rank_weight=args.rank_weight),
                                n_splits=args.folds, seeds=seeds)
            results["ablation_E8_direct"][g] = M.ranking_report(ab.lhv2(), cv.pred.score(), ab.state_index)
            print("ablate", g, results["ablation_E8_direct"][g]["within_state_spearman"])

    results["runtime_seconds"] = time.time() - t0
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
        with (out / "policy_table.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(policy_table[0]))
            w.writeheader()
            w.writerows(policy_table)
        print("wrote", out)
    else:
        print(json.dumps(policy_table, indent=1, default=float))


if __name__ == "__main__":
    main()
