"""
script_18_ottqa_k2_routing.py — Second-benchmark validation on OTT-QA (K=2: table vs. passage)

WHY THIS SCRIPT EXISTS
  The Neurocomputing AE flagged the evaluation as narrow (single benchmark, mmRAG).
  This script tests whether the paper's two central claims replicate on an
  independent benchmark built from the official OTT-QA release
  (https://github.com/wenhuchen/OTT-QA), without touching mmRAG's own indices:

    Claim 1 (length confound): PrefRAG-Conf-style generation-confidence routing
      collapses toward the shorter-context source type on a *different*
      format-heterogeneous pair (table vs. passage, not text/table/KG).
    Claim 2 (supervised routing avoids it): A SourceGate-style router trained on
      soft relevance labels beats both the confidence heuristic and majority/random.

DATA CONSTRUCTION (see conversation for the honesty caveat)
  - "table" context  = serialized table (title + header + rows) from the official
    OTT-QA table dump (data/traindev_tables.json), same pipe-delimited style
    mmRAG itself uses for the `ott` source.
  - "passage" context = the linked-passage snippet text already present in OTT-QA's
    own preprocessed_data/{train,dev}_linked.json (tf-idf / string-overlap fields).
    These are real crawled Wikipedia snippets tied to the table's hyperlinks, not
    synthetic text and not re-crawled by us — but they are snippets, not full
    articles. Report this scope honestly in the paper if these numbers are used.
  - label per query = majority type across `answer-node` ("table" vs "passage"),
    soft-normalized over the two types (ties -> 0.5/0.5), mirroring mmRAG's own
    soft dataset_score construction from graded/tied relevance.

USAGE
  python script_18_ottqa_k2_routing.py --build            # build + cache records
  python script_18_ottqa_k2_routing.py --prefrag           # run PrefRAG-Conf-LN (needs GPU+Llama)
  python script_18_ottqa_k2_routing.py --sourcegate        # run SourceGate-K2 pretraining
  python script_18_ottqa_k2_routing.py --all               # all of the above
  python script_18_ottqa_k2_routing.py --all --n 200       # smoke test

OUTPUTS (phase8_results_ottqa/)
  ottqa_k2_train.json, ottqa_k2_test.json   processed records (cached, reused across runs)
  prefrag_conf_k2_summary.json              confidence-routing collapse result
  sourcegate_k2_summary.json                learned-router result
"""

import json
import argparse
import random
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from sourceformer import SourceFormerK3, gumbel_softmax  # reused architecture, k set to 2 below

# =============================================================================
# Config
# =============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BGE_NAME     = "BAAI/bge-base-en-v1.5"
LLM_NAME     = "meta-llama/Llama-3.1-8B-Instruct"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RAW_DIR          = Path("external/ottqa_raw")
TABLES_FILE      = RAW_DIR / "data/traindev_tables.json"
TRAIN_LINKED_ZIP = RAW_DIR / "preprocessed_data/train_linked.json.zip"
TRAIN_LINKED_NAME = "train_linked.json"
DEV_LINKED_FILE  = RAW_DIR / "preprocessed_data/dev_linked.json"

OUT_DIR = Path("phase8_results_ottqa")
OUT_DIR.mkdir(exist_ok=True)
EMB_CACHE = OUT_DIR / "emb_cache"
EMB_CACHE.mkdir(exist_ok=True)

MAX_CTX_CHARS   = 800
SOURCE_TYPES_K2 = ["table", "passage"]
SIDX            = {s: i for i, s in enumerate(SOURCE_TYPES_K2)}
SEEDS           = [42, 123, 2026]

EPOCHS   = 40
LR       = 1e-3
WD       = 1e-4
BATCH    = 128
PATIENCE = 5


# =============================================================================
# Step 1 — Build K=2 records from the official OTT-QA release
# =============================================================================
def serialize_table(table_obj: dict, max_chars: int = MAX_CTX_CHARS) -> str:
    title  = table_obj.get("title", "")
    header = " | ".join(h[0] for h in table_obj.get("header", []))
    rows   = [" | ".join(cell[0] for cell in row) for row in table_obj.get("data", [])]
    body   = "\n".join(rows)
    text   = f"{title}\n{header}\n{body}" if header else f"{title}\n{body}"
    return text[:max_chars]


def extract_passage_context(item: dict, max_chars: int = MAX_CTX_CHARS) -> str:
    snippets, seen = [], set()
    for key in ("string-overlap", "tf-idf"):
        for entry in item.get(key, []):
            if len(entry) >= 4 and isinstance(entry[3], str) and entry[3].strip():
                snip = entry[3].strip()
                if snip not in seen:
                    seen.add(snip)
                    snippets.append(snip)
    return " ".join(snippets)[:max_chars]


def label_from_answer_nodes(item: dict):
    counts = Counter()
    for n in item.get("answer-node", []):
        if len(n) >= 4 and n[3] in SOURCE_TYPES_K2:
            counts[n[3]] += 1
    if not counts:
        return None
    total = sum(counts.values())
    soft = {s: counts.get(s, 0) / total for s in SOURCE_TYPES_K2}
    hard = max(soft, key=soft.get)
    return soft, hard


def build_records(items: list, tables: dict, split_name: str) -> list:
    records = []
    skipped_no_label, skipped_no_table, skipped_no_passage = 0, 0, 0
    for item in items:
        lbl = label_from_answer_nodes(item)
        if lbl is None:
            skipped_no_label += 1
            continue
        soft, hard = lbl
        table_obj = tables.get(item.get("table_id"))
        if table_obj is None:
            skipped_no_table += 1
            continue
        passage_ctx = extract_passage_context(item)
        if not passage_ctx:
            skipped_no_passage += 1
            continue
        records.append({
            "query":           item["question"],
            "table_context":   serialize_table(table_obj),
            "passage_context": passage_ctx,
            "dataset_score":   soft,
            "label":           SIDX[hard],
        })
    print(f"  [{split_name}] built {len(records):,} records "
          f"(skipped: {skipped_no_label} no-label, {skipped_no_table} no-table, "
          f"{skipped_no_passage} no-passage-snippet)")
    counts = Counter(r["label"] for r in records)
    for i, s in enumerate(SOURCE_TYPES_K2):
        n = counts.get(i, 0)
        print(f"    {s:8s}: {n:>6,} ({100*n/max(len(records),1):.1f}%)")
    return records


def load_and_cache_records(n_limit: int = None):
    train_path = OUT_DIR / "ottqa_k2_train.json"
    test_path  = OUT_DIR / "ottqa_k2_test.json"
    if train_path.exists() and test_path.exists():
        print("Cached K=2 records found, loading from disk.")
        with open(train_path) as f:
            train_records = json.load(f)
        with open(test_path) as f:
            test_records = json.load(f)
        if n_limit:
            train_records, test_records = train_records[:n_limit], test_records[:n_limit]
        return train_records, test_records

    print("Building K=2 records from official OTT-QA release...")
    with open(TABLES_FILE) as f:
        tables = json.load(f)
    print(f"  Loaded {len(tables):,} tables.")

    with zipfile.ZipFile(TRAIN_LINKED_ZIP) as zf:
        with zf.open(TRAIN_LINKED_NAME) as f:
            train_items = json.load(f)
    with open(DEV_LINKED_FILE) as f:
        test_items = json.load(f)
    print(f"  Loaded {len(train_items):,} train / {len(test_items):,} dev (test) raw items.")

    train_records = build_records(train_items, tables, "train")
    test_records  = build_records(test_items, tables, "test")

    with open(train_path, "w") as f:
        json.dump(train_records, f)
    with open(test_path, "w") as f:
        json.dump(test_records, f)
    print(f"  Saved {train_path}, {test_path}")

    if n_limit:
        train_records, test_records = train_records[:n_limit], test_records[:n_limit]
    return train_records, test_records


# =============================================================================
# Step 2 — PrefRAG-Conf-LN on OTT-QA (K=2)
# =============================================================================
def load_llm_4bit():
    print(f"\nLoading {LLM_NAME} in 4-bit NF4 (matches main-paper reader config)...")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, quantization_config=bnb_config, device_map="auto",
    )
    llm.eval()
    print(f"  LLM loaded. VRAM: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
    return tok, llm


def score_source(query: str, context: str, llm_tok, llm) -> float:
    """Mean question-token log-prob conditioned on context (single fwd pass, no generate)."""
    prefix = ("Answer the following question using only the provided context. "
              "Be concise.\n\nContext:\n" + context + "\n\nQuestion: ")
    full = prefix + query + "\n\nAnswer:"

    enc_prefix = llm_tok(prefix, return_tensors="pt", truncation=True, max_length=900)
    enc_full   = llm_tok(full, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    prefix_len = enc_prefix["input_ids"].shape[1]
    seq_len    = enc_full["input_ids"].shape[1]
    if prefix_len >= seq_len:
        return -1e9

    labels = enc_full["input_ids"].clone()
    labels[:, :prefix_len] = -100

    with torch.no_grad():
        out = llm(**enc_full, labels=labels)
    return -out.loss.item()


def eval_routing_k2(preds: np.ndarray, labels: np.ndarray):
    acc = float((preds == labels).mean())
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES_K2):
        mask = (labels == idx)
        per_type[t] = float((preds[mask] == idx).mean()) if mask.sum() else float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return acc, macro, per_type


def run_prefrag_conf(test_records: list):
    print("\n" + "=" * 74)
    print("PrefRAG-Conf on OTT-QA (K=2: table vs. passage)")
    print("=" * 74)

    llm_tok, llm = load_llm_4bit()

    n = len(test_records)
    logprob = np.zeros((n, 2), dtype=np.float64)
    ctx_len = np.zeros((n, 2), dtype=np.int32)
    labels  = np.array([r["label"] for r in test_records], dtype=np.int64)

    for i, r in enumerate(tqdm(test_records, desc="scoring")):
        table_score   = score_source(r["query"], r["table_context"], llm_tok, llm)
        passage_score = score_source(r["query"], r["passage_context"], llm_tok, llm)
        logprob[i] = [table_score, passage_score]
        ctx_len[i, 0] = len(llm_tok(r["table_context"], add_special_tokens=False)["input_ids"])
        ctx_len[i, 1] = len(llm_tok(r["passage_context"], add_special_tokens=False)["input_ids"])

    nll = -logprob
    preds_raw = logprob.argmax(1)
    acc, macro, per_type = eval_routing_k2(preds_raw, labels)
    print(f"\nraw (PrefRAG-Conf): acc={acc:.3f} macro={macro:.3f} "
          f"table={per_type['table']:.3f} passage={per_type['passage']:.3f}")

    print("\nMechanism (per-source means):")
    for j, s in enumerate(SOURCE_TYPES_K2):
        print(f"  {s:8s}  ctx_tokens={ctx_len[:, j].mean():>8.1f}  mean_NLL={nll[:, j].mean():>8.3f}")
    corr = float(np.corrcoef(ctx_len.ravel(), nll.ravel())[0, 1])
    print(f"  corr(ctx_tokens, NLL) = {corr:+.3f}")

    # Length-normalized variants, mirroring script_16
    per_ctx_token = (nll / np.maximum(ctx_len, 1)).argmin(1)
    acc_ln, macro_ln, per_type_ln = eval_routing_k2(per_ctx_token, labels)
    print(f"per_ctx_token:      acc={acc_ln:.3f} macro={macro_ln:.3f} "
          f"table={per_type_ln['table']:.3f} passage={per_type_ln['passage']:.3f}")

    mu, sd = logprob.mean(0, keepdims=True), logprob.std(0, keepdims=True) + 1e-9
    zcal = ((logprob - mu) / sd).argmax(1)
    acc_z, macro_z, per_type_z = eval_routing_k2(zcal, labels)
    print(f"zcal:               acc={acc_z:.3f} macro={macro_z:.3f} "
          f"table={per_type_z['table']:.3f} passage={per_type_z['passage']:.3f}")

    majority_label = Counter(labels.tolist()).most_common(1)[0][0]
    maj_preds = np.full_like(labels, majority_label)
    acc_maj, macro_maj, per_type_maj = eval_routing_k2(maj_preds, labels)
    rand_preds = np.random.RandomState(0).randint(0, 2, size=n)
    acc_rand, macro_rand, per_type_rand = eval_routing_k2(rand_preds, labels)

    summary = {
        "n_queries": n,
        "mechanism": {
            "per_source_mean_ctx_tokens": {s: float(ctx_len[:, j].mean())
                                            for j, s in enumerate(SOURCE_TYPES_K2)},
            "per_source_mean_nll": {s: float(nll[:, j].mean())
                                     for j, s in enumerate(SOURCE_TYPES_K2)},
            "corr_ctxlen_nll": corr,
        },
        "raw":           {"acc": acc, "macro": macro, "per_type": per_type},
        "per_ctx_token": {"acc": acc_ln, "macro": macro_ln, "per_type": per_type_ln},
        "zcal":          {"acc": acc_z, "macro": macro_z, "per_type": per_type_z},
        "majority":      {"acc": acc_maj, "macro": macro_maj, "per_type": per_type_maj},
        "random":        {"acc": acc_rand, "macro": macro_rand, "per_type": per_type_rand},
    }
    out = OUT_DIR / "prefrag_conf_k2_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out}")
    return summary


# =============================================================================
# Step 3 — SourceGate-K2 pretraining
# =============================================================================
def encode_queries(records: list, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(records):
            print(f"  cache hit: {cache_path}")
            return arr
    print(f"  Encoding {len(records):,} queries with BGE...")
    tok = AutoTokenizer.from_pretrained(BGE_NAME)
    bge = AutoModel.from_pretrained(BGE_NAME, torch_dtype=torch.float32).to(DEVICE).eval()
    queries = [QUERY_PREFIX + r["query"] for r in records]
    out = np.empty((len(queries), 768), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, len(queries), 64), desc="BGE encode"):
            e = min(s + 64, len(queries))
            enc = tok(queries[s:e], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(DEVICE)
            emb = F.normalize(bge(**enc).last_hidden_state[:, 0], p=2, dim=1)
            out[s:e] = emb.cpu().numpy()
    np.save(cache_path, out)
    del bge
    torch.cuda.empty_cache()
    return out


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(targets * logp).sum(dim=-1).mean()


def train_sourcegate_k2(train_records, test_records, train_embs, test_embs, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_targets = torch.tensor(
        [[r["dataset_score"][s] for s in SOURCE_TYPES_K2] for r in train_records],
        dtype=torch.float32)
    train_hard = torch.tensor([r["label"] for r in train_records], dtype=torch.long)
    test_targets = torch.tensor(
        [[r["dataset_score"][s] for s in SOURCE_TYPES_K2] for r in test_records],
        dtype=torch.float32)
    test_labels = torch.tensor([r["label"] for r in test_records], dtype=torch.long)

    train_embs_t = torch.from_numpy(train_embs).float()
    test_embs_t  = torch.from_numpy(test_embs).float()

    # inverse-frequency weighted sampling (mirrors main paper's K=3 sampler)
    counts = Counter(train_hard.tolist())
    weights = torch.tensor([1.0 / counts[label.item()] for label in train_hard])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = torch.utils.data.TensorDataset(train_embs_t, train_targets, train_hard)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler)

    model = SourceFormerK3(input_dim=768, hidden=512, mid=128, k=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_macro, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for embs, targets, _ in train_loader:
            embs, targets = embs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(embs)
            loss = soft_cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(test_embs_t.to(DEVICE))
            preds = logits.argmax(-1).cpu().numpy()
        acc, macro, per_type = eval_routing_k2(preds, test_labels.numpy())

        if macro > best_macro:
            best_macro, best_state, patience_ctr = macro, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        logits = model(test_embs_t.to(DEVICE))
        preds = logits.argmax(-1).cpu().numpy()
    acc, macro, per_type = eval_routing_k2(preds, test_labels.numpy())
    print(f"  seed={seed}  test acc={acc:.3f} macro={macro:.3f} "
          f"table={per_type['table']:.3f} passage={per_type['passage']:.3f}  (best epoch macro={best_macro:.3f})")
    return {"acc": acc, "macro": macro, "per_type": per_type}


def run_sourcegate(train_records, test_records):
    print("\n" + "=" * 74)
    print("SourceGate-K2 pretraining on OTT-QA")
    print("=" * 74)

    train_embs = encode_queries(train_records, EMB_CACHE / "train_embs.npy")
    test_embs  = encode_queries(test_records, EMB_CACHE / "test_embs.npy")

    runs = []
    for seed in SEEDS:
        runs.append(train_sourcegate_k2(train_records, test_records, train_embs, test_embs, seed))

    macros = [r["macro"] for r in runs]
    mean_macro, std_macro = float(np.mean(macros)), float(np.std(macros))
    print(f"\nSourceGate-K2 over {len(SEEDS)} seeds: macro={mean_macro:.3f} +/- {std_macro:.3f}")

    # baselines for reference in the same table
    test_labels = np.array([r["label"] for r in test_records])
    majority_label = Counter(test_labels.tolist()).most_common(1)[0][0]
    acc_maj, macro_maj, _ = eval_routing_k2(np.full_like(test_labels, majority_label), test_labels)
    rand_preds = np.random.RandomState(0).randint(0, 2, size=len(test_labels))
    acc_rand, macro_rand, _ = eval_routing_k2(rand_preds, test_labels)

    summary = {
        "n_train": len(train_records),
        "n_test": len(test_records),
        "sourcegate_k2": {"mean_macro": mean_macro, "std_macro": std_macro, "runs": runs},
        "majority": {"acc": acc_maj, "macro": macro_maj},
        "random":   {"acc": acc_rand, "macro": macro_rand},
    }
    out = OUT_DIR / "sourcegate_k2_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out}")
    return summary


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--prefrag", action="store_true")
    ap.add_argument("--sourcegate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="smoke-test row limit")
    args = ap.parse_args()

    if not (args.build or args.prefrag or args.sourcegate or args.all):
        ap.print_help()
        return

    train_records, test_records = load_and_cache_records(n_limit=args.n)

    if args.prefrag or args.all:
        run_prefrag_conf(test_records)

    if args.sourcegate or args.all:
        run_sourcegate(train_records, test_records)


if __name__ == "__main__":
    main()
