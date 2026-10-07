"""Deterministic evidence-preserving and destructive format interventions."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

import script_39_counterfactual_evidence_gain as S39


TRIPLE = re.compile(r"^\s*(.*?)\s+--\s+(.*?)\s+-->\s+(.*?)\s*$")


def table_to_records(text):
    lines = text.splitlines()
    out, converted, i = [], 0, 0
    while i < len(lines):
        if i + 1 < len(lines) and "|" in lines[i] and re.fullmatch(r"[\s|:-]+", lines[i + 1]):
            headers = [x.strip() for x in lines[i].strip(" |").split("|")]
            out.append("Table columns: " + "; ".join(headers) + ".")
            i += 2
            row_id = 1
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                vals = [x.strip() for x in lines[i].strip(" |").split("|")]
                pairs = [f"{h} = {v}" for h, v in zip(headers, vals)]
                out.append(f"Record {row_id}: " + "; ".join(pairs) + ".")
                converted += 1
                row_id += 1
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), converted


def kg_to_records(text):
    out, converted = [], 0
    for line in text.splitlines():
        m = TRIPLE.match(line)
        if m:
            s, r, o = (x.strip() for x in m.groups())
            out.append(f"Subject = {s}; Relation = {r}; Object = {o}.")
            converted += 1
        else:
            out.append(line)
    return "\n".join(out), converted


def reverse_fact_order(text, source):
    lines = text.splitlines()
    if source == "kg":
        pos = [i for i, line in enumerate(lines) if TRIPLE.match(line)]
    else:
        pos = [i for i, line in enumerate(lines) if "|" in line and not re.fullmatch(r"[\s|:-]+", line)]
        pos = pos[1:] if len(pos) > 1 else []  # retain the table header
    vals = [lines[i] for i in pos][::-1]
    for i, value in zip(pos, vals):
        lines[i] = value
    return "\n".join(lines), len(pos)


def break_associations(text, source):
    """Retain vocabulary but rotate values/objects across facts."""
    lines = text.splitlines()
    if source == "kg":
        parsed = [(i, TRIPLE.match(line)) for i, line in enumerate(lines)]
        parsed = [(i, m) for i, m in parsed if m]
        if len(parsed) > 1:
            objects = [m.group(3).strip() for _, m in parsed]
            objects = objects[1:] + objects[:1]
            for (i, m), obj in zip(parsed, objects):
                lines[i] = f"{m.group(1).strip()} -- {m.group(2).strip()} --> {obj}"
        return "\n".join(lines), len(parsed)
    # For tables, rotate the final column across data rows while retaining all tokens.
    row_pos = [i for i, line in enumerate(lines)
               if "|" in line and not re.fullmatch(r"[\s|:-]+", line)]
    row_pos = row_pos[1:] if len(row_pos) > 1 else []
    cells = [[x.strip() for x in lines[i].strip(" |").split("|")] for i in row_pos]
    if len(cells) > 1 and all(c for c in cells):
        last = [c[-1] for c in cells]
        last = last[1:] + last[:1]
        for i, c, value in zip(row_pos, cells, last):
            c[-1] = value
            lines[i] = " | ".join(c)
    return "\n".join(lines), len(row_pos)


def transform(text, source, mode):
    if mode == "records":
        return table_to_records(text) if source == "table" else kg_to_records(text)
    if mode == "fact_reverse":
        return reverse_fact_order(text, source)
    if mode == "broken_association":
        return break_associations(text, source)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "all"), default="dev")
    args = ap.parse_args()
    root = Path("phase9_results")
    contexts = json.load(open(root / f"retrieved_contexts_{args.split}.json"))
    data_file = "mmrag_dev.json" if args.split == "dev" else "mmrag_test.json"
    data = json.load(open(data_file))
    report = {}
    for mode in ("records", "fact_reverse", "broken_association"):
        saved, coverage, answer_preserved = {}, {"table": [], "kg": []}, []
        for key, source_contexts in contexts.items():
            i = int(key)
            saved[key] = dict(source_contexts)
            for source in ("table", "kg"):
                changed, count = transform(source_contexts[source], source, mode)
                saved[key][source] = changed
                coverage[source].append(count)
                before = S39.contains_gold(source_contexts[source], data[i])
                after = S39.contains_gold(changed, data[i])
                answer_preserved.append(before == after)
        out = root / f"retrieved_contexts_{args.split}_{mode}.json"
        with open(out, "w") as f:
            json.dump(saved, f)
        report[mode] = {
            "output": str(out),
            "answer_containment_preserved_fraction": float(np.mean(answer_preserved)),
            "mean_converted_facts": {s: float(np.mean(v)) for s, v in coverage.items()},
            "fraction_with_any_conversion": {s: float(np.mean(np.asarray(v) > 0))
                                               for s, v in coverage.items()},
        }
    out = root / f"format_intervention_audit_{args.split}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
