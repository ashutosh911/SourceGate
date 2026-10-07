"""Dev-to-test stability of null correction across donor subsets.

This reuses the five already-scored, label-free null donors.  For every nonempty
subset of donor positions, lambda is selected on dev only and then frozen on the
test set.  The analysis does not create independent donor pools, but it directly
checks whether the reported effect is carried by one favorable null example.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


OUT_DIR = Path("phase9_results")
SOURCES = ("text", "table", "kg")
GRID = np.round(np.arange(0.0, 1.5001, 0.025), 3)
FROZEN_FULL_DONOR_LAMBDA = 0.55


def load_rows(path):
    obj = json.load(open(path))
    return [obj[k] for k in sorted(obj, key=lambda x: int(x))]


def arrays(rows, subset):
    y = np.asarray([r["true_label"] for r in rows], dtype=int)
    raw = np.asarray([[r["raw"][s] for s in SOURCES] for r in rows])
    null = np.asarray([
        [np.mean([r["negative_scores"][s][j] for j in subset]) for s in SOURCES]
        for r in rows
    ])
    return y, raw, null


def macro(y, scores):
    pred = scores.argmax(1)
    per = [np.mean(pred[y == k] == k) for k in range(3)]
    return float(np.mean(per)), {s: float(per[k]) for k, s in enumerate(SOURCES)}


def aucs(y, scores):
    return {
        s: float(roc_auc_score(y == k, scores[:, k]))
        for k, s in enumerate(SOURCES)
    }


def best_lambda(y, raw, null):
    values = np.asarray([macro(y, raw - lam * null)[0] for lam in GRID])
    best = values.max()
    # Deterministic conservative tie break: smallest correction attaining best dev.
    idx = int(np.flatnonzero(np.isclose(values, best, atol=1e-12))[0])
    return float(GRID[idx]), float(best)


def summarize_group(records):
    delta = np.asarray([r["test_delta"] for r in records])
    frozen_delta = np.asarray([r["frozen_lambda_test_delta"] for r in records])
    return {
        "num_subsets": len(records),
        "test_delta_mean": float(delta.mean()),
        "test_delta_min": float(delta.min()),
        "test_delta_max": float(delta.max()),
        "fraction_positive": float(np.mean(delta > 0)),
        "frozen_lambda_0.55_delta_mean": float(frozen_delta.mean()),
        "frozen_lambda_0.55_delta_range": [
            float(frozen_delta.min()), float(frozen_delta.max())
        ],
        "frozen_lambda_0.55_fraction_positive": float(np.mean(frozen_delta > 0)),
        "selected_lambda_range": [
            float(min(r["lambda"] for r in records)),
            float(max(r["lambda"] for r in records)),
        ],
        "auc_delta_range": {
            s: [
                float(min(r["auc_delta"][s] for r in records)),
                float(max(r["auc_delta"][s] for r in records)),
            ]
            for s in SOURCES
        },
    }


def main():
    dev_rows = load_rows(OUT_DIR / "null_evidence_cache_dev_m5.json")
    test_rows = load_rows(OUT_DIR / "null_evidence_cache_all_m5.json")
    y_test = np.asarray([r["true_label"] for r in test_rows], dtype=int)
    raw_test = np.asarray([[r["raw"][s] for s in SOURCES] for r in test_rows])
    raw_macro, raw_per = macro(y_test, raw_test)
    raw_auc = aucs(y_test, raw_test)

    records = []
    for size in range(1, 6):
        for subset in itertools.combinations(range(5), size):
            y_dev, raw_dev, null_dev = arrays(dev_rows, subset)
            lam, dev_macro = best_lambda(y_dev, raw_dev, null_dev)
            _, _, null_test = arrays(test_rows, subset)
            corrected = raw_test - lam * null_test
            frozen_corrected = raw_test - FROZEN_FULL_DONOR_LAMBDA * null_test
            test_macro, test_per = macro(y_test, corrected)
            frozen_macro, frozen_per = macro(y_test, frozen_corrected)
            corrected_auc = aucs(y_test, corrected)
            records.append({
                "subset_zero_based": list(subset),
                "num_donors": size,
                "lambda": lam,
                "dev_macro": dev_macro,
                "test_macro": test_macro,
                "test_delta": test_macro - raw_macro,
                "test_per_type": test_per,
                "auc": corrected_auc,
                "auc_delta": {s: corrected_auc[s] - raw_auc[s] for s in SOURCES},
                "frozen_lambda": FROZEN_FULL_DONOR_LAMBDA,
                "frozen_lambda_test_macro": frozen_macro,
                "frozen_lambda_test_delta": frozen_macro - raw_macro,
                "frozen_lambda_test_per_type": frozen_per,
            })

    summary = {
        "analysis": "all nonempty subsets of the five pre-scored donor positions",
        "scope_note": (
            "subset sensitivity, not independent donor-pool replication; lambda "
            "selected separately on dev for each subset and frozen on test"
        ),
        "raw_test": {"macro": raw_macro, "per_type": raw_per, "auc": raw_auc},
        "by_num_donors": {
            str(size): summarize_group([r for r in records if r["num_donors"] == size])
            for size in range(1, 6)
        },
        "all_subsets": summarize_group(records),
        "records": records,
    }
    out = OUT_DIR / "donor_subset_stability_dev_to_test.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "raw_test_macro": raw_macro,
        "by_num_donors": summary["by_num_donors"],
        "all_subsets": summary["all_subsets"],
        "saved": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
