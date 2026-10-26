"""Jarayon darajasidagi (process-wide) BIR XIL lock — BARCHA GPU-bog'liq
model-chaqiruvlar (Whisper til-aniqlash/decode, WeSpeaker/ECAPA ovoz
embedding) shu orqali ketma-ket (bir vaqtda FAQAT BITTASI) ishlaydi.

REAL TESTDA TOPILGAN MUAMMO (latency spike): har bir WebSocket sessiyasi
O'ZINING mustaqil turn_worker()iga ega (dialog.py) — bitta sessiya ICHIDA
GPU chaqiruvlar ketma-ket (bir worker) bo'lsa ham, IKKI (yoki undan ko'p)
BIR VAQTDA OCHIQ sessiya BOSHQA-BOSHQA worker'larda BIR XIL GPU'ga
PARALLEL murojaat qilishi mumkin edi — bu CUDA kontekst almashinuvi/
xotira talashish tufayli BIR CHAQIRUVNING o'zini o'n(lab) soniyagacha
sekinlashtirib yuborishi mumkin (real logda: 4.7s audio uchun 26.18s STT,
odatdagi ~1.5s o'rniga — ikkinchi sessiyaning navbati BIRINCHISI bilan
BIR VAQTDA GPU so'ragan payt kuzatilgan).

Shu BITTA, process-wide lock BARCHA sessiyalar uchun GPU chaqiruvlarni
to'liq ketma-ketlashtiradi — bitta GPU baribir bir vaqtning o'zida bitta
ishni bajaradi, shuning uchun bu HAQIQIY parallellikdan hech qanday
yo'qotish EMAS (aslida GPU allaqachon shunday ishlagan, faqat ENDI
Python/CUDA tarafida ham mos ravishda KUTIB turiladi — thrashing o'rniga).

Oddiy (reentrant EMAS) `threading.Lock` — chaqiruvchi funksiyalar hech
qachon shu lock ushlab turilgan paytda O'ZINI qayta chaqirmasligi SHART
(tekshirildi: stt.py va speaker_diart.py bir-birini import qilmaydi,
har bir lock-o'ralgan funksiya o'zi ichida qayta kirmaydi)."""
import threading

GPU_LOCK = threading.Lock()
