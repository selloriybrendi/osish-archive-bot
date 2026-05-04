"""
messages.json dan ChunkLar yasash + Gemini orqali embedding qilish.

Chunking strategiyasi:
  - Bir xil sender + 5 daqiqa ichidagi ketma-ket xabarlar = bitta chunk
  - Bo'sh matnli xabarlar tashlab yuboriladi
  - Har chunk maks 1500 belgi, oshsa bo'linadi

Output:
  - chunks.json — chunk metadata (id, sender, sana, matn)
  - embeddings.npy — numpy array (N x 768)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
INPUT = Path(os.getenv("INPUT_FILE", "messages.json"))
CHUNKS_OUT = Path("chunks.json")
EMB_OUT = Path("embeddings.npy")

GEMINI_MODEL = "models/gemini-embedding-001"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:embedContent"
CHUNK_GAP_SEC = 300        # 5 daqiqa
CHUNK_MAX_CHARS = 1500
BATCH_SIZE = 50            # API'ga jo'natiladigan bir bach

if not GEMINI_KEY:
    print("[XATO] .env faylida GEMINI_API_KEY bo'lishi kerak.")
    sys.exit(1)
if not INPUT.exists():
    print(f"[XATO] {INPUT} fayl topilmadi. Avval extract.py ni ishga tushiring.")
    sys.exit(1)


def parse_iso(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def make_chunks(messages):
    """Xabarlarni mantiqiy chunkka guruhlash."""
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


def embed(text: str) -> list:
    """Bitta matn uchun Gemini embedding."""
    payload = {
        "model": GEMINI_MODEL,
        "content": {"parts": [{"text": text}]},
    }
    r = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_KEY},
        json=payload,
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Embed xatosi: {r.status_code} {r.text[:200]}")
    return r.json()["embedding"]["values"]


def main():
    print(f"[INFO] {INPUT} o'qilmoqda...")
    messages = json.loads(INPUT.read_text(encoding="utf-8"))
    print(f"  {len(messages)} ta xabar")

    print(f"[INFO] Chunklarga guruhlanmoqda...")
    chunks = make_chunks(messages)
    print(f"  {len(chunks)} ta chunk hosil bo'ldi")

    print(f"[INFO] Gemini embedding boshlandi (free tier: ~1500 RPM)...")
    embeddings = []
    failed = 0
    for i, ch in enumerate(chunks, 1):
        # Sender + sana ham embedding'ga kiritamiz — qidiruv yaxshilanadi
        text_for_emb = f"{ch['sender']} ({ch['first_date'][:10]}): {ch['text']}"
        try:
            vec = embed(text_for_emb)
            embeddings.append(vec)
        except Exception as e:
            print(f"  [{i}] xato: {e}")
            embeddings.append([0.0] * 3072)  # placeholder
            failed += 1
        if i % 50 == 0:
            print(f"  {i}/{len(chunks)} ({100*i//len(chunks)}%) — failed: {failed}")
        time.sleep(0.05)  # rate-limit hurmati uchun

    arr = np.array(embeddings, dtype=np.float32)
    np.save(EMB_OUT, arr)
    CHUNKS_OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saqlandi:")
    print(f"  {CHUNKS_OUT} ({len(chunks)} chunk)")
    print(f"  {EMB_OUT} ({arr.shape}, {arr.nbytes/1024/1024:.2f} MB)")
    if failed:
        print(f"  [DIQQAT] {failed} ta chunk uchun embedding muvaffaqiyatsiz bo'ldi")


if __name__ == "__main__":
    main()
