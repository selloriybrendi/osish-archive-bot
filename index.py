"""
messages.json dan chunklar yasash + local sentence-transformers bilan embedding qilish.

Local model — internet kerak emas, kunlik limit yo'q, bepul abadiy.
Birinchi run-da model yuklanadi (~470 MB), keyin lokalda kesh saqlanadi.

Output:
  - chunks.json — chunk metadata (id, sender, sana, matn)
  - embeddings.npy — numpy array (N x 768)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
INPUT = Path(os.getenv("INPUT_FILE", "messages.json"))
CHUNKS_OUT = Path("chunks.json")
EMB_OUT = Path("embeddings.npy")

# Ko'p tilli model (o'zbek, rus, ingliz — barchasini qo'llaydi)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_GAP_SEC = 300        # 5 daqiqa
CHUNK_MAX_CHARS = 1500
BATCH_SIZE = 64

if not INPUT.exists():
    print(f"[XATO] {INPUT} fayl topilmadi. Avval extract.py ni ishga tushiring.")
    sys.exit(1)


def parse_iso(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def make_chunks(messages):
    chunks = []
    current = None
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        sender = m.get("sender_name") or f"id={m.get('sender_id')}"
        date = m.get("date")
        if not date:
            continue

        if current is None:
            current = {"sender": sender, "first_date": date, "last_date": date,
                       "first_id": m["id"], "last_id": m["id"], "text": text}
            continue

        same_sender = current["sender"] == sender
        try:
            gap = (parse_iso(date) - parse_iso(current["last_date"])).total_seconds()
        except Exception:
            gap = 0

        if same_sender and gap <= CHUNK_GAP_SEC and len(current["text"]) + len(text) + 2 < CHUNK_MAX_CHARS:
            current["text"] += "\n" + text
            current["last_date"] = date
            current["last_id"] = m["id"]
        else:
            chunks.append(current)
            current = {"sender": sender, "first_date": date, "last_date": date,
                       "first_id": m["id"], "last_id": m["id"], "text": text}
    if current:
        chunks.append(current)
    return chunks


def main():
    print(f"[INFO] {INPUT} o'qilmoqda...")
    messages = json.loads(INPUT.read_text(encoding="utf-8"))
    print(f"  {len(messages)} ta xabar")

    print(f"[INFO] Chunklarga guruhlanmoqda...")
    chunks = make_chunks(messages)
    print(f"  {len(chunks)} ta chunk hosil bo'ldi")

    print(f"[INFO] Local model yuklanmoqda: {MODEL_NAME}")
    print(f"  (birinchi marta ~470 MB yuklab olinadi, keyin keshlanadi)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model tayyor (dim={model.get_sentence_embedding_dimension()})")

    # Matnlarni tayyorlash
    texts = [f"{ch['sender']} ({ch['first_date'][:10]}): {ch['text']}" for ch in chunks]

    print(f"[INFO] Embedding boshlandi (lokal CPU/GPU, internetsiz)...")
    print(f"  Batch size: {BATCH_SIZE}")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    arr = np.array(embeddings, dtype=np.float32)

    np.save(EMB_OUT, arr)
    CHUNKS_OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Saqlandi:")
    print(f"  {CHUNKS_OUT} ({len(chunks)} chunk)")
    print(f"  {EMB_OUT} ({arr.shape}, {arr.nbytes/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
