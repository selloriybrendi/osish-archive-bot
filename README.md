# O'sish Nuqtasi Archive Bot

"O'sish nuqtasi" Telegram guruhining butun xabarlar tarixini o'qiydigan va savollarga javob beradigan AI bot. Telethon (tarix yuklab olish) + Gemini API (embedding + javob).

## Arxitektura

```
extract.py  →  messages.json   (BIR MARTA, sizning Mac'ingizda)
                    ↓
index.py    →  chunks.json + embeddings.npy   (BIR MARTA)
                    ↓
bot.py      →  24/7 ishlaydi (Telegram bot polling)
```

## Sozlash (qadamma-qadam)

### 1-bosqich: Mac'da loyihani sozlash

```bash
# Loyiha papkasiga kiring (yoki yaratib oling)
git clone https://github.com/selloriybrendi/osish-archive-bot.git
cd osish-archive-bot

# Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env faylini yarating
cp .env.example .env
nano .env
```

`.env` ichini to'ldiring (5 ta qiymat):
- `API_ID` — my.telegram.org dan
- `API_HASH` — my.telegram.org dan
- `BOT_TOKEN` — @BotFather'dan
- `GEMINI_API_KEY` — aistudio.google.com/apikey dan
- `GROUP_NAME` — guruh nomi (`O'sish nuqtasi`)

### 2-bosqich: Guruh tarixini yuklab olish (BIR MARTA)

```bash
python extract.py
```

Birinchi marta ishga tushganda:
1. Telefon raqamingizni so'raydi → kiriting
2. Telegram'dan kelgan kodni so'raydi → kiriting
3. (Agar 2FA yoqilgan bo'lsa) parolni so'raydi
4. Guruhni topadi va xabarlarni yuklab olishni boshlaydi

Natija: `messages.json` fayl (KOMMITGA TUSHMAYDI — `.gitignore`'da).

### 3-bosqich: Xabarlarni indekslash

```bash
python index.py
```

Gemini API orqali har bir xabarni embedding'ga aylantiradi. Vaqti: ~10 daqiqa har 1000 xabar uchun.

Natija: `chunks.json` + `embeddings.npy`.

### 4-bosqich: Botni ishga tushirish

```bash
python bot.py
```

Endi botingiz @WorkArchive_*_bot bilan suhbat qilib savollar berishingiz mumkin.

## Bot bilan ishlash

**DM (private chat):**
- `/start` — botni qarshilash
- `/stats` — arxiv statistikasi
- har qanday xabar — savol sifatida ishlanadi

**Guruhda:**
- Botni guruhga qo'shing (admin bo'lishi shart emas)
- `@WorkArchive_yourname_bot eng ko'p kim gapiradi?` — javob beradi
- Yoki bot xabariga reply qilish — savol sifatida ishlaydi

## 24/7 hosting

Mac'ingiz har doim ochiq bo'lmagani uchun, botni serverga deploy qilish kerak. Variantlar:

1. **Railway.app** (Recommended): bepul $5 kreditga 1 oy yetadi
2. **Fly.io**: bepul tier
3. **VPS** (Hetzner CX11 €4/oy): to'liq nazorat

Har bittasi uchun `.env` o'zgaruvchilarini va `chunks.json` + `embeddings.npy` fayllarini yuklash kerak (lekin `messages.json` shart emas — chunks ichida hammasi bor).

## Xavfsizlik

- `.env` faylini **hech qachon** commit qilmang.
- `messages.json` ichida sizning shaxsiy guruh tarixingiz — uni **hech kimga** bermang.
- Token'lar chiqib ketsa — darhol revoke qiling:
  - Bot: @BotFather → `/revoke`
  - Telegram API: my.telegram.org → eski app'ni o'chirib, yangisi
  - Gemini: aistudio.google.com/apikey → eski kalitni delete

## Litsenziya

Shaxsiy foydalanish uchun. MIT.
