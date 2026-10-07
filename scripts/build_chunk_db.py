import json
import sqlite3
from pathlib import Path

DB_PATH = Path("chunk_texts.db")
CHUNKS_DIR = Path("chunks_by_source")
SOURCES = ["nq", "triviaqa", "ott", "tat", "kg"]

if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
    print(f"{DB_PATH} already exists and is non-empty — delete it first to rebuild.")
    exit()

print("Building chunk_texts.db...")
con = sqlite3.connect(DB_PATH)
cur = con.cursor()
cur.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT)")
cur.execute("CREATE INDEX idx_id ON chunks(id)")

BATCH = 10_000
total = 0
for src in SOURCES:
    buf = []
    path = CHUNKS_DIR / f"{src}.jsonl"
    if not path.exists():
        print(f"  [skip] {path} not found")
        continue
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content")
            if isinstance(text, dict):
                text = text.get("text", "")
            if not text or not isinstance(text, str):
                continue
            buf.append((obj["id"], text))
            if len(buf) == BATCH:
                cur.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?)", buf)
                con.commit()
                total += len(buf)
                buf = []
    if buf:
        cur.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?)", buf)
        con.commit()
        total += len(buf)
    print(f"  {src}: done")

con.close()
print(f"Done. {total:,} chunks in {DB_PATH} ({DB_PATH.stat().st_size/1e9:.2f} GB)")

