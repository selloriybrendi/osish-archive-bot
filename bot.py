"""
RAG Telegram bot — "O'sish nuqtasi" arxivi bilan suhbat.
Tabiiy javob, manbalar ko'rsatilmaydi.
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
TOP_K = int(os.getenv("TOP_K", "15"))

CHUNKS_FILE = Path("chunks.json")
EMB_FILE = Path("embeddings.npy")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("bot")

if not BOT_TOKEN or not GEMINI_KEY:
    log.error(".env: BOT_TOKEN va GEMINI_API_KEY bo'lishi kerak.")
    sys.exit(1)

if not CHUNKS_FILE.exists() or not EMB_FILE.exists():
    log.error("chunks.json yoki embeddings.npy topilmadi. Avval index.py ni ishga tushiring.")
    sys.exit(1)

CHUNKS = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
EMBS = np.load(EMB_FILE)
EMB_NORMS = np.linalg.norm(EMBS, axis=1, keepdims=True) + 1e-12
EMBS_NORMALIZED = EMBS / EMB_NORMS
log.info(f"Yuklandi: {len(CHUNKS)} chunk, embeddings shape: {EMBS.shape}")

log.info(f"Local model yuklanmoqda: {MODEL_NAME}")
from sentence_transformers import SentenceTransformer
EMBED_MODEL = SentenceTransformer(MODEL_NAME)
log.info(f"Embed model tayyor (dim={EMBED_MODEL.get_sentence_embedding_dimension()})")


def embed_query(text: str) -> np.ndarray:
    v = EMBED_MODEL.encode(text, convert_to_numpy=True)
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def search(question, top_k=TOP_K):
    q = embed_query(question)
    sims = EMBS_NORMALIZED @ q
    idxs = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), CHUNKS[i]) for i in idxs]


def build_prompt(question, hits):
    context_lines = []
    for rank, (_, score, ch) in enumerate(hits, 1):
        date = (ch.get("first_date") or "")[:10]
        sender = ch.get("sender", "Unknown")
        text = ch.get("text", "")
        context_lines.append(f"[{rank}] {date} - {sender}:\n{text}")
    context = "\n\n".join(context_lines)
    return (
        "Sen 'O'sish nuqtasi' guruhi a'zolarining suhbatlarini biluvchi aqlli yordamchi botsan.\n"
        "Foydalanuvching - Otaxon (Intelektos), guruh adminlaridan biri.\n\n"
        "Vazifa: foydalanuvchi savoliga MATERIAL'dan kelib chiqib tabiiy, suhbatdoshlik ohangida javob ber.\n\n"
        "QOIDALAR:\n"
        "- Material asosida xulosa chiqar, sintez qil. Bir nechta xabardan birlashtirib, o'z so'zlaringizda ayt.\n"
        "- Quruq hisobot emas, suhbatdoshlik uslubida yoz - do'st bilan gaplashayotgandek.\n"
        "- Faqat aniq ma'lumot bo'lmaganda 'Bu haqida arxivda aniq ma'lumot yo'q' de.\n"
        "- Kim aytgani yoki sanasi muhim bo'lsa gap ichida tabiiy ko'rsat (masalan: 'Maftuna 5-aprelda aytgancha...'), 'Manbalar' ro'yxati ko'rinishida emas.\n"
        "- O'zbek tilida, Otaxon'ga 'siz' deb murojaat qil.\n"
        "- Qisqa va aniq, lekin sovuq emas. Inson kabi gaplash.\n"
        "- Markdown belgilarni ishlatma (** yulduzcha, # va boshqalar). Faqat oddiy matn.\n\n"
        f"Savol: {question}\n\n"
        "Guruh xabarlari (siz uchun eng mos keladiganlari):\n"
        f"{context}\n\n"
        "Javob (faqat o'zbek tilida, suhbatdoshlik bilan):"
    )


def gemini_generate(prompt):
    r = requests.post(
        GEMINI_GEN_URL,
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}},
        timeout=60,
    )
    if r.status_code != 200:
        return f"Gemini xatosi: {r.status_code}"
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return "Javob bo'sh keldi."


def is_allowed(user_id):
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Salom! Men 'O'sish nuqtasi' guruhi xabarlarini yodimda tutgan yordamchingizman.\n\n"
        f"Indeksda {len(CHUNKS)} ta gap bor. Savol bering - javob topib beraman.\n\n"
        "Misol: \"cvb so'nggi haftada nima yozdi?\", \"Maftuna narxlar haqida nima degan?\""
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    senders = {}
    for ch in CHUNKS:
        s = ch.get("sender", "?")
        senders[s] = senders.get(s, 0) + 1
    top = sorted(senders.items(), key=lambda x: -x[1])[:10]
    txt = f"Arxiv: {len(CHUNKS)} chunk\n\nEng faol yozuvchilar:\n"
    for s, c in top:
        txt += f"  - {s}: {c}\n"
    if CHUNKS:
        txt += f"\nDavr: {CHUNKS[0]['first_date'][:10]} -> {CHUNKS[-1]['last_date'][:10]}"
    await update.message.reply_text(txt)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return
    user = msg.from_user
    if not is_allowed(user.id):
        await msg.reply_text("Sizga bu botdan foydalanishga ruxsat yo'q.")
        return

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
        await msg.reply_text(answer)
    except Exception as e:
        log.exception("Xato")
        await msg.reply_text(f"Xato: {e}")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Bot ishga tushdi (polling rejimi)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
