# script_confidence_baseline.py
"""
BGE-confidence routing baseline.
Routes each query to the source whose top-1 BGE retrieval score is highest.
No training. Tests whether SourceFormer beats a trivial similarity heuristic.
"""
import json, numpy as np, torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda"
BGE = "BAAI/bge-base-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
INDICES_DIR = Path("faiss_indices")

SOURCE_TYPES_K3 = ["text", "table", "kg"]
TYPE_TO_DATASETS_K3 = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
SOURCE_TYPES_K5 = ["nq", "triviaqa", "ott", "tat", "kg"]

with open("mmrag_test.json") as f:
    test = json.load(f)
print(f"Test queries: {len(test)}")

tok = AutoTokenizer.from_pretrained(BGE)
bge = AutoModel.from_pretrained(BGE, torch_dtype=torch.float16).to(DEVICE).eval()

indices = {}
for ds in ["nq", "triviaqa", "ott", "tat", "kg"]:
    indices[ds] = faiss.read_index(str(INDICES_DIR / ds / "index.faiss"))

queries = [PREFIX + t["query"] for t in test]
embs = []
B = 64
with torch.no_grad():
    for i in tqdm(range(0, len(queries), B), desc="encoding"):
        enc = tok(queries[i:i+B], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(DEVICE)
        e = bge(**enc).last_hidden_state[:, 0]
        embs.append(F.normalize(e.float(), p=2, dim=1).cpu().numpy())
embs = np.vstack(embs).astype(np.float32)

def hard_label_k3(item):
    scores = item["dataset_score"]
    text = scores.get("nq", 0) + scores.get("triviaqa", 0)
    table = scores.get("ott", 0) + scores.get("tat", 0)
    kg = scores.get("kg", 0)
    return int(np.argmax([text, table, kg]))

def hard_label_k5(item):
    scores = item["dataset_score"]
    return int(np.argmax([scores.get(s, 0) for s in SOURCE_TYPES_K5]))

# K=3 routing
print("\n=== K=3 confidence routing ===")
type_scores = np.zeros((len(test), 3), dtype=np.float32)
for j, t in enumerate(SOURCE_TYPES_K3):
    best = -np.inf * np.ones(len(test))
    for ds in TYPE_TO_DATASETS_K3[t]:
        s, _ = indices[ds].search(embs, 1)
        best = np.maximum(best, s[:, 0])
    type_scores[:, j] = best
decisions_k3 = np.argmax(type_scores, axis=1)
labels_k3 = np.array([hard_label_k3(t) for t in test])

acc_k3 = (decisions_k3 == labels_k3).mean()
per_type_k3 = {}
for j, t in enumerate(SOURCE_TYPES_K3):
    mask = labels_k3 == j
    per_type_k3[t] = (decisions_k3[mask] == j).mean() if mask.sum() else 0
macro_k3 = np.mean(list(per_type_k3.values()))

# K=5 routing
print("\n=== K=5 confidence routing ===")
type_scores_k5 = np.zeros((len(test), 5), dtype=np.float32)
for j, ds in enumerate(SOURCE_TYPES_K5):
    s, _ = indices[ds].search(embs, 1)
    type_scores_k5[:, j] = s[:, 0]
decisions_k5 = np.argmax(type_scores_k5, axis=1)
labels_k5 = np.array([hard_label_k5(t) for t in test])

acc_k5 = (decisions_k5 == labels_k5).mean()
per_type_k5 = {}
for j, t in enumerate(SOURCE_TYPES_K5):
    mask = labels_k5 == j
    per_type_k5[t] = (decisions_k5[mask] == j).mean() if mask.sum() else 0
macro_k5 = np.mean(list(per_type_k5.values()))

# Report
print(f"\n{'='*60}")
print(f"BGE-CONFIDENCE BASELINE RESULTS (test set)")
print(f"{'='*60}")
print(f"\n  K=3:  acc={acc_k3:.4f}  macro={macro_k3:.4f}")
print(f"        per-type: {per_type_k3}")
print(f"\n  K=5:  acc={acc_k5:.4f}  macro={macro_k5:.4f}")
print(f"        per-type: {per_type_k5}")

print(f"\n  For comparison (test set):")
print(f"    Random:       acc=0.333  macro=0.337")
print(f"    Majority:     acc=0.558  macro=0.333")
print(f"    Phase 3 K=3:  acc=0.728  macro=0.737")
print(f"    Phase 4 K=3:  acc=0.751  macro=0.717")
print(f"    Phase 3 K=5:  --        macro=0.709")
print(f"    Phase 4 K=5:  acc=0.666  macro=0.696")

# Save decisions for downstream F1/EM evaluation
np.save("phase5_results/confidence_decisions_k3.npy", decisions_k3)
np.save("phase5_results/confidence_decisions_k5.npy", decisions_k5)

# Save summary
with open("phase5_results/confidence_summary.json", "w") as f:
    json.dump({
        "k3": {"acc": float(acc_k3), "macro": float(macro_k3),
               "per_type": {k: float(v) for k, v in per_type_k3.items()}},
        "k5": {"acc": float(acc_k5), "macro": float(macro_k5),
               "per_type": {k: float(v) for k, v in per_type_k5.items()}},
    }, f, indent=2)

print(f"\n  Saved: confidence_decisions_k3.npy, _k5.npy, confidence_summary.json")
