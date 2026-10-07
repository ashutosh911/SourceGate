"""
script_27_llm_classifier_baseline.py

Zero-shot LLM classifier baseline for K=3 source routing, requested by
Reviewer 2 (point 2.3): "the baseline of asking an LLM to classify which
source a query needs was not run."

No training, no embeddings -- GPT-4o-mini reads the query and one short
description per source type, and picks text / table / kg directly.

Usage:
    python script_27_llm_classifier_baseline.py \
        --test_file mmrag_test.json \
        --out phase5_results/llm_classifier_k3_results.json

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

# ----------------------------------------------------------------------
# Source descriptions, taken verbatim from the paper's own Section 5.1
# dataset descriptions (NQ/TriviaQA -> text, OTT-QA/TAT-QA -> table,
# Freebase -> kg), so the baseline's framing matches exactly what the
# manuscript already tells the reader about each source.
# ----------------------------------------------------------------------
SOURCE_DESCRIPTIONS = {
    "text": (
        "Wikipedia-style factual passages and trivia-style factual "
        "passages (natural-language prose)."
    ),
    "table": (
        "Structured tabular data, including Wikipedia table lookups and "
        "financial tables requiring numerical/arithmetic reasoning."
    ),
    "kg": (
        "A knowledge graph of subject-predicate-object triples "
        "(e.g. 'Barack Obama -- born in -> Honolulu'), serialized to text."
    ),
}

SYSTEM_PROMPT = (
    "You are a routing component in a retrieval-augmented question-"
    "answering system. Given a user question, decide which ONE of three "
    "knowledge sources should be searched to answer it:\n\n"
    f"- text: {SOURCE_DESCRIPTIONS['text']}\n"
    f"- table: {SOURCE_DESCRIPTIONS['table']}\n"
    f"- kg: {SOURCE_DESCRIPTIONS['kg']}\n\n"
    "Respond with ONLY one word: text, table, or kg. No explanation, "
    "no punctuation, no other text."
)

VALID_LABELS = {"text", "table", "kg"}


def classify_query(client: OpenAI, question: str, model: str = "gpt-4o-mini",
                    max_retries: int = 3) -> str:
    """Return one of {'text','table','kg'}, or 'unparsed' on repeated failure."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question.strip()},
                ],
                temperature=0,
                max_tokens=5,
            )
            raw = resp.choices[0].message.content.strip().lower()
            # Strip any stray punctuation/quotes the model might add
            cleaned = re.sub(r"[^a-z]", "", raw)
            if cleaned in VALID_LABELS:
                return cleaned
            # Loose containment fallback (e.g. model says "Table." or "the table")
            for label in VALID_LABELS:
                if label in cleaned:
                    return label
            # Unparseable this attempt; retry
        except Exception as e:
            print(f"  [warn] attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return "unparsed"


def load_test_queries(path: str):
    """
    Expects the same mmRAG test file format used elsewhere in the repo:
    a list of records with at least 'question' and a gold source-type
    label. Adjust the two field names below if your mmrag_test.json uses
    different keys -- check against how script_7_baselines.py loads it.
    """
    with open(path) as f:
        data = json.load(f)
    queries = data if isinstance(data, list) else data.get("data", data)
    return queries


def gold_label_k3(record) -> str:
    """
    Derive the K=3 gold label from the record's 'id' field, whose prefix
    names the source dataset directly (e.g. 'ott_552' -> ott, 'nq_19764'
    -> nq, 'triviaqa_58922' -> triviaqa, 'tat_1039' -> tat). KG queries
    use a Freebase-style id starting with 'm.' instead of a dataset
    prefix. Mirrors the NQ+TriviaQA->text, OTT+TAT->table, KG->kg
    grouping used throughout the paper (Section 5.1).

    Falls back to argmax over 'dataset_score' (the same per-source
    relevance signal the paper's L_route soft labels are built from,
    Section 3.3) if the id prefix doesn't match a known pattern.
    """
    rid = str(record.get("id", "")).lower()

    if rid.startswith("m.") or rid.startswith("kg_"):
        return "kg"
    if rid.startswith("nq_") or rid.startswith("triviaqa_"):
        return "text"
    if rid.startswith("ott_") or rid.startswith("tat_"):
        return "table"

    # Fallback: argmax of dataset_score, grouped into K=3
    scores = record.get("dataset_score", {})
    if scores:
        text_score = scores.get("nq", 0) + scores.get("triviaqa", 0)
        table_score = scores.get("ott", 0) + scores.get("tat", 0)
        kg_score = scores.get("kg", 0)
        best = max(
            [("text", text_score), ("table", table_score), ("kg", kg_score)],
            key=lambda x: x[1],
        )
        return best[0]

    raise ValueError(f"Unrecognized id/dataset_score in record: {record}")


def compute_macro_accuracy(preds, golds):
    types = ["text", "table", "kg"]
    per_type_correct = {t: 0 for t in types}
    per_type_total = {t: 0 for t in types}
    correct = 0
    for p, g in zip(preds, golds):
        per_type_total[g] += 1
        if p == g:
            per_type_correct[g] += 1
            correct += 1
    per_type_acc = {
        t: (per_type_correct[t] / per_type_total[t] if per_type_total[t] else 0.0)
        for t in types
    }
    macro = sum(per_type_acc.values()) / len(types)
    acc = correct / len(preds) if preds else 0.0
    return acc, macro, per_type_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="mmrag_test.json")
    ap.add_argument("--out", default="phase5_results/llm_classifier_k3_results.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional cap on number of queries, for a quick smoke test.")
    args = ap.parse_args()

    client = OpenAI()  # picks up OPENAI_API_KEY from env

    queries = load_test_queries(args.test_file)
    if args.limit:
        # mmrag_test.json is grouped/sorted by dataset (confirmed: a
        # positional --limit 20 returned 20/20 table-labeled queries), so
        # a plain slice is not a representative sample. Shuffle with a
        # fixed seed instead, so the smoke test actually exercises all
        # three K=3 classes and remains reproducible across runs.
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, min(args.limit, len(queries)))

    print(f"Classifying {len(queries)} queries with {args.model} (zero-shot, "
          f"no training)...")

    # Surface class imbalance immediately -- this is what caught the
    # earlier --limit sampling bug (a positional slice returned 20/20
    # table-labeled queries because the file is grouped by dataset).
    gold_preview = [gold_label_k3(r) for r in queries]
    from collections import Counter
    balance = Counter(gold_preview)
    print(f"  Gold-label balance in this sample: {dict(balance)}")
    if any(balance.get(t, 0) == 0 for t in ("text", "table", "kg")):
        print("  [warn] at least one class has ZERO examples in this sample -- "
              "macro accuracy will be uninformative. Check sampling/schema.")

    preds, golds, raw_records = [], [], []
    for i, record in enumerate(queries):
        question = record.get("question", record.get("query", ""))
        gold = gold_label_k3(record)
        pred = classify_query(client, question, model=args.model)
        preds.append(pred)
        golds.append(gold)
        raw_records.append({"question": question, "gold": gold, "pred": pred})

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} done")

    n_unparsed = sum(1 for p in preds if p == "unparsed")
    if n_unparsed:
        print(f"  [warn] {n_unparsed} queries returned unparseable responses "
              f"after retries -- excluded from accuracy computation")
        preds_clean, golds_clean = zip(*[
            (p, g) for p, g in zip(preds, golds) if p != "unparsed"
        ])
    else:
        preds_clean, golds_clean = preds, golds

    acc, macro, per_type_acc = compute_macro_accuracy(list(preds_clean), list(golds_clean))

    print("\n" + "=" * 60)
    print(f"RESULTS -- {args.model} zero-shot classifier (K=3)")
    print("=" * 60)
    print(f"  Accuracy:       {acc:.3f}")
    print(f"  Macro accuracy: {macro:.3f}")
    for t, a in per_type_acc.items():
        print(f"    {t:6s}: {a:.3f}")
    print(f"  Unparsed:       {n_unparsed} / {len(queries)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_queries": len(queries),
            "n_unparsed": n_unparsed,
            "acc": acc,
            "macro_acc": macro,
            "per_type_acc": per_type_acc,
            "records": raw_records,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
