"""Leakage-safe analysis of the multi-negative null-evidence pilot.

This script performs no retrieval or LLM inference.  It reads the completed
script_40 cache, reports the pre-specified argmax results for 1..M negatives,
mechanism-oriented one-vs-rest AUCs, paired stratified bootstrap intervals, and
nested-CV diagnostic classifiers.  Hyperparameter C is selected only inside
each outer training fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_TYPES = ["text", "table", "kg"]
SEED = 20260925


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    per = np.diag(cm) / np.bincount(y, minlength=3)
    return {
        "macro": float(per.mean()),
        "accuracy": float(np.mean(pred == y)),
        "per_type": {s: float(per[k]) for k, s in enumerate(SOURCE_TYPES)},
        "prediction_counts": {
            s: int(np.sum(pred == k)) for k, s in enumerate(SOURCE_TYPES)
        },
    }


def stratified_bootstrap_delta(y, base_pred, alt_pred, reps=20000, seed=SEED):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == k) for k in range(3)]
    values = []
    for _ in range(reps):
        per_class = []
        for k, pool in enumerate(by_class):
            ix = rng.choice(pool, len(pool), replace=True)
            per_class.append(
                np.mean(alt_pred[ix] == k) - np.mean(base_pred[ix] == k)
            )
        values.append(np.mean(per_class))
    values = np.asarray(values)
    observed = metrics(y, alt_pred)["macro"] - metrics(y, base_pred)["macro"]
    return {
        "delta": float(observed),
        "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "p_delta_le_zero": float(np.mean(values <= 0)),
        "bootstrap_reps": reps,
    }


def nested_cv_predict(X, y, outer_splits, c_grid=(0.01, 0.1, 1.0, 10.0)):
    pred = np.empty(len(y), dtype=int)
    chosen = []
    for fold, (train, test) in enumerate(outer_splits):
        inner = StratifiedKFold(4, shuffle=True, random_state=SEED + fold + 1)
        c_scores = []
        for c in c_grid:
            inner_macros = []
            for fit_rel, val_rel in inner.split(X[train], y[train]):
                fit, val = train[fit_rel], train[val_rel]
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=5000, random_state=SEED,
                                       class_weight="balanced"),
                )
                model.fit(X[fit], y[fit])
                inner_macros.append(metrics(y[val], model.predict(X[val]))["macro"])
            c_scores.append(float(np.mean(inner_macros)))
        best = int(np.argmax(c_scores))
        c = c_grid[best]
        chosen.append({"fold": fold, "C": c, "inner_macro": c_scores[best]})
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, random_state=SEED,
                               class_weight="balanced"),
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--num-negatives", type=int, default=5)
    ap.add_argument("--tag", default=None,
                    help="Cache tag such as 'all'; defaults to pc{per_class}")
    args = ap.parse_args()
    result_dir = Path("phase9_results")
    tag = args.tag or f"pc{args.per_class}"
    cache_path = result_dir / (
        f"null_evidence_cache_{tag}_m{args.num_negatives}.json"
    )
    rows = list(json.load(open(cache_path)).values())
    full_split_tag = tag == "all" or tag.startswith("all_") or tag == "dev" or tag.startswith("dev_")
    if not full_split_tag and len(rows) != args.per_class * 3:
        raise RuntimeError(f"Expected {args.per_class * 3} rows, found {len(rows)}")

    y = np.asarray([r["true_label"] for r in rows], dtype=int)
    raw = np.asarray([[r["raw"][s] for s in SOURCE_TYPES] for r in rows])
    neg = np.asarray([
        [[r["negative_scores"][s][m] for s in SOURCE_TYPES]
         for m in range(args.num_negatives)]
        for r in rows
    ])
    null_mean = neg.mean(1)
    null_sd = neg.std(1, ddof=1)
    gain = raw - null_mean
    raw_pred = raw.argmax(1)
    gain_pred = gain.argmax(1)

    by_m = {}
    for m in range(1, args.num_negatives + 1):
        score = raw - neg[:, :m].mean(1)
        pred = score.argmax(1)
        by_m[str(m)] = {
            **metrics(y, pred),
            "vs_raw": stratified_bootstrap_delta(
                y, raw_pred, pred, reps=5000, seed=SEED + m
            ),
        }

    auc = {}
    for k, s in enumerate(SOURCE_TYPES):
        target = y == k
        auc[s] = {
            "raw": float(roc_auc_score(target, raw[:, k])),
            "mean_gain": float(roc_auc_score(target, gain[:, k])),
            "null_mean": float(roc_auc_score(target, null_mean[:, k])),
        }

    outer = list(StratifiedKFold(
        5, shuffle=True, random_state=SEED
    ).split(raw, y))
    feature_sets = {
        "raw_only": raw,
        "gain_only": gain,
        "raw_plus_null_mean": np.c_[raw, null_mean],
        "raw_plus_null_mean_sd": np.c_[raw, null_mean, null_sd],
    }
    nested = {}
    for name, X in feature_sets.items():
        pred, chosen = nested_cv_predict(X, y, outer)
        nested[name] = {
            **metrics(y, pred),
            "vs_raw_argmax": stratified_bootstrap_delta(
                y, raw_pred, pred, reps=5000, seed=SEED + len(nested) + 20
            ),
            "outer_predictions": pred.tolist(),
            "selected_C": chosen,
        }

    padding = {
        s: float(np.mean([
            flag for r in rows for flag in r["negative_required_padding"][s]
        ]))
        for s in SOURCE_TYPES
    }
    summary = {
        "method": "multi-negative null-normalized evidence analysis",
        "n": len(rows),
        "per_class": args.per_class,
        "num_negatives": args.num_negatives,
        "raw_argmax": metrics(y, raw_pred),
        "mean_gain_argmax": metrics(y, gain_pred),
        "mean_gain_vs_raw": stratified_bootstrap_delta(y, raw_pred, gain_pred),
        "number_of_negatives_ablation": by_m,
        "one_vs_rest_auc": auc,
        "nested_cv_diagnostics": nested,
        "negative_padding_rate": padding,
        "notes": [
            "Negative donors use no routing labels or gold answers.",
            "Nested-CV classifiers are diagnostic; C is selected inside each outer fold.",
            "The 300-query pilot is balanced by hard benchmark label and is not the full test set.",
        ],
    }
    out = result_dir / (
        f"null_evidence_analysis_{tag}_m{args.num_negatives}.json"
    )
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"raw argmax       {summary['raw_argmax']['macro']:.4f}")
    print(f"mean-gain argmax {summary['mean_gain_argmax']['macro']:.4f}  "
          f"CI={summary['mean_gain_vs_raw']['ci95']}")
    print("AUC raw -> gain:")
    for s in SOURCE_TYPES:
        print(f"  {s:5s}: {auc[s]['raw']:.4f} -> {auc[s]['mean_gain']:.4f}")
    print("Nested-CV diagnostics:")
    for name, result in nested.items():
        print(f"  {name:24s} {result['macro']:.4f}  {result['per_type']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
