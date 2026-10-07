"""Select the null-subtraction weight on dev and optionally evaluate on test."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


RESULTS = Path("phase9_results")
SOURCES = ["text", "table", "kg"]
SEED = 42


def load(tag):
    rows = list(json.load(open(RESULTS / f"null_evidence_cache_{tag}_m5.json")).values())
    y = np.asarray([r["true_label"] for r in rows], dtype=int)
    raw = np.asarray([[r["raw"][s] for s in SOURCES] for r in rows])
    null = np.asarray([
        [np.mean(r["negative_scores"][s]) for s in SOURCES] for r in rows
    ])
    return y, raw, null


def macro_and_per_type(y, pred):
    per = [float(np.mean(pred[y == k] == k)) for k in range(3)]
    return float(np.mean(per)), {s: per[k] for k, s in enumerate(SOURCES)}


def bootstrap_macro(y, base, alternative, reps=50000):
    rng = np.random.default_rng(SEED)
    pools = [np.flatnonzero(y == k) for k in range(3)]
    values = []
    for _ in range(reps):
        per = []
        for k, pool in enumerate(pools):
            ix = rng.choice(pool, len(pool), replace=True)
            per.append(np.mean(alternative[ix] == k) - np.mean(base[ix] == k))
        values.append(np.mean(per))
    values = np.asarray(values)
    return {
        "ci95_two_sided": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "p_one_sided_delta_le_zero": float(np.mean(values <= 0)),
        "reps": reps,
    }


def bootstrap_auc(y, raw, corrected, source_idx, reps=10000):
    rng = np.random.default_rng(SEED + source_idx)
    target = y == source_idx
    pos, neg = np.flatnonzero(target), np.flatnonzero(~target)
    values = []
    for _ in range(reps):
        ix = np.r_[rng.choice(pos, len(pos), replace=True),
                   rng.choice(neg, len(neg), replace=True)]
        values.append(
            roc_auc_score(target[ix], corrected[ix, source_idx])
            - roc_auc_score(target[ix], raw[ix, source_idx])
        )
    values = np.asarray(values)
    return {
        "ci95_two_sided": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "p_one_sided_delta_le_zero": float(np.mean(values <= 0)),
        "reps": reps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-suffix", default="",
                    help="Cache suffix, e.g. qwenmath for dev_qwenmath/all_qwenmath")
    ap.add_argument("--select-only", action="store_true",
                    help="Select on dev and stop without reading the test cache")
    args = ap.parse_args()
    suffix = f"_{args.reader_suffix}" if args.reader_suffix else ""
    y_dev, raw_dev, null_dev = load(f"dev{suffix}")
    grid = np.round(np.arange(0, 1.5001, 0.025), 3)
    dev_macro = []
    for weight in grid:
        pred = (raw_dev - weight * null_dev).argmax(1)
        dev_macro.append(macro_and_per_type(y_dev, pred)[0])
    dev_macro = np.asarray(dev_macro)
    best = np.flatnonzero(np.isclose(dev_macro, dev_macro.max()))
    # Conservative tie rule fixed before inspecting test: smallest weight.
    selected_idx = best[np.argmin(grid[best])]
    weight = float(grid[selected_idx])

    corrected_dev = raw_dev - weight * null_dev
    dev_auc = {}
    for k, source in enumerate(SOURCES):
        target = y_dev == k
        raw_auc = float(roc_auc_score(target, raw_dev[:, k]))
        corrected_auc = float(roc_auc_score(target, corrected_dev[:, k]))
        dev_auc[source] = {
            "raw": raw_auc,
            "corrected": corrected_auc,
            "delta": corrected_auc - raw_auc,
        }

    selection = {
        "method": "dev-selected scalar null correction",
        "score": "raw_logprob - lambda * mean_matched_null_logprob",
        "reader_suffix": args.reader_suffix or "primary",
        "selection_split": "dev",
        "lambda_grid": {"min": 0.0, "max": 1.5, "step": 0.025},
        "selected_lambda": weight,
        "dev_best_macro": float(dev_macro[selected_idx]),
        "dev_raw_macro": float(dev_macro[0]),
        "dev_one_vs_rest_auc": dev_auc,
    }
    if args.select_only:
        out = RESULTS / f"dev_selected_null_weight{suffix}.json"
        with open(out, "w") as f:
            json.dump(selection, f, indent=2)
        print(json.dumps(selection, indent=2))
        print(f"Saved {out}")
        return

    y_test, raw_test, null_test = load(f"all{suffix}")

    base_pred = raw_test.argmax(1)
    corrected_scores = raw_test - weight * null_test
    corrected_pred = corrected_scores.argmax(1)
    raw_macro, raw_per = macro_and_per_type(y_test, base_pred)
    corrected_macro, corrected_per = macro_and_per_type(y_test, corrected_pred)

    auc = {}
    for k, source in enumerate(SOURCES):
        target = y_test == k
        raw_auc = float(roc_auc_score(target, raw_test[:, k]))
        corrected_auc = float(roc_auc_score(target, corrected_scores[:, k]))
        auc[source] = {
            "raw": raw_auc,
            "corrected": corrected_auc,
            "delta": corrected_auc - raw_auc,
            "bootstrap": bootstrap_auc(y_test, raw_test, corrected_scores, k),
        }

    summary = {
        **selection,
        "evaluation_split": "test",
        "test": {
            "raw_macro": raw_macro,
            "corrected_macro": corrected_macro,
            "macro_delta": corrected_macro - raw_macro,
            "raw_per_type": raw_per,
            "corrected_per_type": corrected_per,
            "macro_bootstrap": bootstrap_macro(y_test, base_pred, corrected_pred),
            "one_vs_rest_auc": auc,
        },
        "notes": [
            "The test labels are not used to select lambda.",
            "Negative donors use no routing labels or gold answers.",
            "The two-sided macro interval slightly overlaps zero; treat overall gain as marginal.",
        ],
    }
    out = RESULTS / f"dev_selected_null_weight_test{suffix}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
