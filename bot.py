"""
RAG Telegram bot — "O'sish nuqtasi" arxivi bilan suhbat.

Foydalanuvchi savol bersa:
  1) Savolni embedding'ga aylantiramiz (Gemini)
  2) Eng mos top-K chunkni topamiz (cosine similarity)
  3) Ularni context sifatida Gemini'ga jo'natamiz
  4) Javobni yuboramiz

DM (Direct Message) — bot bilan shaxsiy suhbatda
Group — @bot_username deb chaqirilganda javob beradi
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ALLOWED_USER_IDS = [int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()]
TOP_K = int(os.getenv("TOP_K", "10"))

CHUNKS_FILE = Path("chunks.json")
EMB_FILE = Path("embeddings.npy")

GEMINI_EMB_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("bot")

if not BOT_TOKEN or not GEMINI_KEY:
    log.error(".env: BOT_TOKEN va GEMINI_API_KEY bo'lishi kerak.")
    sys.exit(1)

# Indeksni yuklash
if not CHUNKS_FILE.exists() or not EMB_FILE.exists():
    log.error("chunks.json yoki embeddings.npy topilmadi. Avval index.py ni ishga tushiring.")
    sys.exit(1)

CHUNKS = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
EMBS = np.load(EMB_FILE)
EMB_NORMS = np.linalg.norm(EMBS, axis=1, keepdims=True) + 1e-12
EMBS_NORMALIZED = EMBS / EMB_NORMS
log.info(f"Yuklandi: {len(CHUNKS)} chunk, embeddings shape: {EMBS.shape}")


def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        GEMINI_EMB_URL,
        params={"key": GEMINI_KEY},
        json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}},
        timeout=30,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"]["values"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def search(question: str, top_k: int = TOP_K):
    q = embed_query(question)
    sims = EMBS_NORMALIZED @ q  # cosine similarity
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), CHUNKS[i]) for i in idxs]


def build_prompt(question: str, hits) -> str:
    context_lines = []
    for rank, (_, score, ch) in enumerate(hits, 1):
        date = (ch.get("first_date") or "")[:10]
        sender = ch.get("sender", "Unknown")
        text = ch.get("text", "")
        context_lines.append(f"[{rank}] {date} — {sender}:\n{text}")
    context = "\n\n".join(context_lines)
    return f"""Sen "O'sish nuqtasi" Telegram guruhi arxivi bilan ishlaydigan yordamchisan.
Foydalanuvchi savolini guruh tarixidan kelib chiqib javob ber.

QOIDALAR:
- Faqat quyidagi MATERIAL'dan foydalanib javob ber. Yo'q narsani o'ylab topma.
- Agar savolga material'da javob bo'lmasa: "Bu haqida arxivda ma'lumot topmadim" deb ayt.
- Sana va kim aytganini eslab o'ting.
- Qisqa va aniq javob ber, mosini ko'paytirma.

SAVOL: {question}

MATERIAL (guruh xabarlari):
{context}

JAVOB (o'zbek tilida):"""


def gemini_generate(prompt: str) -> str:
    r = requests.post(
        GEMINI_GEN_URL,
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}},
        timeout=60,
    )
    if r.status_code != 200:
        return f"⚠️ Gemini xatosi: {r.status_code}"
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return "⚠️ Javob bo'sh keldi."


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


# ---------- Handler'lar ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Salom! Men 'O'sish nuqtasi' guruhi arxivi bilan ishlovchi botman.\n\n"
        f"Indeksda {len(CHUNKS)} ta xabar guruhlangan. Savol bering — "
        "guruh tarixidan javob topib beraman.\n\n"
        "Misol: \"Eng ko'p kim gapiradi?\" yoki "
        "\"Loyiha haqida nima muhokama qilingan?\""
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    senders = {}
    for ch in CHUNKS:
        s = ch.get("sender", "?")
        senders[s] = senders.get(s, 0) + 1
    top = sorted(senders.items(), key=lambda x: -x[1])[:5]
    txt = f"📊 Arxiv: {len(CHUNKS)} chunk, {sum(senders.values())} xabar\n\nEng faol:\n"
    for s, c in top:
        txt += f"  • {s}: {c} chunk\n"
    if CHUNKS:
        txt += f"\nDavr: {CHUNKS[0]['first_date'][:10]} → {CHUNKS[-1]['last_date'][:10]}"
    await update.message.reply_text(txt)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return
    user = msg.from_user
    if not is_allowed(user.id):
        await msg.reply_text("Sizga bu botdan foydalanishga ruxsat yo'q.")
        return

    # Group'da faqat bot mention qilingan paytda javob beramiz
    bot_username = (await context.bot.get_me()).username
    if msg.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        if not (msg.text.lower().startswith(f"@{bot_username.lower()}") or
                (msg.reply_to_message and msg.reply_to_message.from_user
                 and msg.reply_to_message.from_user.username == bot_username)):
            return
        question = re.sub(rf"@{bot_username}\s*", "", msg.text, count=1, flags=re.I).strip()
    else:
        question = msg.text.strip()

    if not question:
        return

    log.info(f"Savol [{user.username or user.id}]: {question[:60]!r}")
    await context.bot.send_chat_action(chat_id=msg.chat.id, action="typing")

    try:
        hits = search(question)
        prompt = build_prompt(question, hits)
        answer = gemini_generate(prompt)
        # Manbalarni ham qo'shamiz
        sources = "\n\n📚 <i>Manbalar (top-3):</i>\n"
        for i, (_, score, ch) in enumerate(hits[:3], 1):
            date = ch.get("first_date", "")[:10]
            sources += f"{i}. {ch.get('sender', '?')} — {date} (mos: {score:.2f})\n"
        await msg.reply_text(answer + sources, parse_mode="HTML")
    except Exception as e:
        log.exception("Xato")
        await msg.reply_text(f"⚠️ Xato: {e}")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Bot ishga tushdi (polling rejimi)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
