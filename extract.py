"""
Telethon orqali "O'sish nuqtasi" guruhining BUTUN tarixini yuklab olish.

Ishlash:
  1) Birinchi run-da telefon raqami va Telegram'dan kelgan kod so'raladi (interaktiv).
  2) Session fayli saqlanadi — keyingi run-da kod kerak emas.
  3) Guruhdagi har bir xabar messages.json fayliga yoziladi.

Ishga tushirish:
  python extract.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat

load_dotenv()

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()
GROUP_NAME = os.getenv("GROUP_NAME", "O'sish nuqtasi").strip()
SESSION = os.getenv("SESSION_NAME", "extractor")
OUTPUT = Path(os.getenv("OUTPUT_FILE", "messages.json"))

if not API_ID or not API_HASH:
    print("[XATO] .env faylida API_ID va API_HASH bo'lishi kerak.")
    sys.exit(1)


def msg_to_dict(m) -> dict:
    """Xabarni JSON-friendly dict ga aylantirish."""
    sender_name = ""
    sender_username = ""
    if hasattr(m, "sender") and m.sender:
        sender_username = getattr(m.sender, "username", "") or ""
        first = getattr(m.sender, "first_name", "") or ""
        last = getattr(m.sender, "last_name", "") or ""
        sender_name = (first + " " + last).strip()
    return {
        "id": m.id,
        "date": m.date.isoformat() if m.date else None,
        "sender_id": getattr(m, "sender_id", None),
        "sender_name": sender_name,
        "sender_username": sender_username,
        "text": (m.message or "").strip(),
        "reply_to_msg_id": getattr(m.reply_to, "reply_to_msg_id", None) if m.reply_to else None,
        "has_media": bool(m.media),
        "media_type": type(m.media).__name__ if m.media else None,
    }


async def find_group(client, name: str):
    """Dialog ro'yxatidan guruhni nomi bo'yicha topish."""
    print(f"[INFO] Dialog ro'yxati tekshirilmoqda... ('{name}' qidirilmoqda)")
    async for dialog in client.iter_dialogs():
        if dialog.is_group or dialog.is_channel:
            title = (dialog.title or "").lower()
            if name.lower() in title or title in name.lower():
                print(f"[OK] Topildi: '{dialog.title}' (id={dialog.id}, type={type(dialog.entity).__name__})")
                return dialog.entity
    return None


async def main():
    client = TelegramClient(SESSION, API_ID, API_HASH)
    print("[INFO] Telegram'ga ulanmoqda...")
    await client.start()  # interaktiv: telefon, kod, password
    me = await client.get_me()
    print(f"[OK] Login: {me.first_name} (@{me.username}, id={me.id})")

    target = await find_group(client, GROUP_NAME)
    if not target:
        print(f"[XATO] '{GROUP_NAME}' topilmadi. Quyidagi dialoglardan tanlang:")
        async for d in client.iter_dialogs(limit=30):
            if d.is_group or d.is_channel:
                print(f"  - {d.title!r} (id={d.id})")
        await client.disconnect()
        sys.exit(2)

    # Xabarlarni yuklab olish
    print(f"[INFO] Xabarlar yuklab olinmoqda...")
    messages = []
    total = 0
    async for m in client.iter_messages(target, limit=None):
        if not m.message and not m.media:
            continue  # bo'sh service xabarlar (join/leave)
        messages.append(msg_to_dict(m))
        total += 1
        if total % 500 == 0:
            print(f"  ... {total} ta xabar yuklab olindi")

    print(f"[OK] Jami: {total} ta xabar")
    # Eng eskidan eng yangigacha tartiblash
    messages.sort(key=lambda x: x["id"])

    OUTPUT.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"[OK] Fayl saqlandi: {OUTPUT.absolute()} ({size_mb:.2f} MB)")

    # Statistika
    senders = {}
    for m in messages:
        key = m["sender_name"] or f"id={m['sender_id']}"
        senders[key] = senders.get(key, 0) + 1
    print(f"\n[STATISTIKA]")
    print(f"  Eng faol 5 ta yozuvchi:")
    for sender, count in sorted(senders.items(), key=lambda x: -x[1])[:5]:
        print(f"    {sender}: {count} ta xabar")
    if messages:
        print(f"  Sana oralig'i: {messages[0]['date']} -> {messages[-1]['date']}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
