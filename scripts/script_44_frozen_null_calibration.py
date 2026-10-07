"""Dev-selected, frozen null-aware calibration for a completed reader cache.

Regularization is selected by cross-validation on dev only, separately for a
raw-score multinomial calibrator and a raw+null-mean calibrator.  Both are then
fit on all dev records and evaluated once on the untouched test cache.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from script_41_analyze_null_evidence import metrics, stratified_bootstrap_delta


RESULTS = Path("phase9_results")
SOURCES = ("text", "table", "kg")
SEED = 20260925
C_GRID = (0.01, 0.1, 1.0, 10.0)


def load(tag):
    rows = list(json.load(open(RESULTS / f"null_evidence_cache_{tag}_m5.json")).values())
    y = np.asarray([r["true_label"] for r in rows], dtype=int)
    raw = np.asarray([[r["raw"][s] for s in SOURCES] for r in rows])
    null = np.asarray([[np.mean(r["negative_scores"][s]) for s in SOURCES]
                       for r in rows])
    return y, raw, null


def model(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000, random_state=SEED,
                           class_weight="balanced"),
    )


def select_c(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for c in C_GRID:
        fold_scores = []
        for train, val in cv.split(X, y):
            clf = model(c).fit(X[train], y[train])
            fold_scores.append(metrics(y[val], clf.predict(X[val]))["macro"])
        scores[str(c)] = float(np.mean(fold_scores))
    best = max(C_GRID, key=lambda c: (scores[str(c)], -c))
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-suffix", required=True)
    args = ap.parse_args()
    suffix = args.reader_suffix
    y_dev, raw_dev, null_dev = load(f"dev_{suffix}")
    y_test, raw_test, null_test = load(f"all_{suffix}")

    feature_sets = {
        "raw_only": (raw_dev, raw_test),
        "raw_plus_null_mean": (
            np.c_[raw_dev, null_dev], np.c_[raw_test, null_test]
        ),
    }
    fits, predictions = {}, {}
    for name, (X_dev, X_test) in feature_sets.items():
        c, cv_scores = select_c(X_dev, y_dev)
        clf = model(c).fit(X_dev, y_dev)
        pred = clf.predict(X_test)
        predictions[name] = pred
        fits[name] = {
            "selected_C": c,
            "dev_cv_macro_by_C": cv_scores,
            "test": metrics(y_test, pred),
        }

    raw_argmax = raw_test.argmax(1)
    summary = {
        "analysis_role": "exploratory cross-reader extension; test protocol frozen after dev",
        "reader_suffix": suffix,
        "selection": "five-fold dev CV only; smallest C breaks ties",
        "raw_argmax_test": metrics(y_test, raw_argmax),
        "models": fits,
        "null_aware_vs_raw_calibrator": stratified_bootstrap_delta(
            y_test, predictions["raw_only"], predictions["raw_plus_null_mean"]
        ),
        "null_aware_vs_raw_argmax": stratified_bootstrap_delta(
            y_test, raw_argmax, predictions["raw_plus_null_mean"]
        ),
    }
    out = RESULTS / f"frozen_null_calibration_{suffix}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
