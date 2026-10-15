import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws/dialog";

// ===========================================================================
// DIALOG_PLAN.md 3-bosqich (C1-C5): audio yozib olish MediaRecorder/WebM'dan
// AudioWorkletNode orqali XOM PCM'ga o'tkazildi — qarang frontend/public/
// pcm-worklet.js va pastdagi startListenLoop(). Sabablari:
//   - C3 (pre-roll): MediaRecorder "nutq aniqlangandan KEYIN" ishga
//     tushirilardi — gap boshidagi 100-300ms doim kesilib ketardi.
//     AudioWorklet orqali XOM namunalarga uzluksiz kirish bo'lgani uchun
//     endi "nutq boshlangan" deb topilgan PAYTDAN OLDINGI bir necha
//     bloqni ham (pre-roll) turnga qo'shib yuborish mumkin.
//   - C3 (aniq kesish): WebM/MediaRecorder konteyner formatida ixtiyoriy
//     vaqt nuqtasidan "kesib olish" ishonchli emas (faqat BIRINCHI chunk'da
//     konteyner header bor) — xom PCM massividan esa istalgan namunadan
//     kesib olish oddiy massiv-amal.
// Backend tarafida bu _decode_waveform_bytes() (webm, ffmpeg orqali decode)
// dan _decode_pcm_bytes() (to'g'ridan-to'g'ri np.frombuffer)ga o'tishni
// talab qildi — qarang stt.py va dialog.py'dagi handle_audio_chunk().
// ===========================================================================

// Backend (SAMPLE_RATE, stt.py) 16kHz mono kutadi — AudioContext ANIQ shu
// tezlikda yaratiladi (quyida, startListenLoop'da), shunda hech qanday
// JS-tarafidagi resampling kerak emas.
const TARGET_SAMPLE_RATE = 16000;

// pcm-worklet.js'dagi PCM_CHUNK_SIZE bilan BIR XIL bo'lishi SHART — bu
// yerda faqat bitta worklet-xabarining necha millisoniyaga teng ekanini
// hisoblash uchun kerak (pre-roll/kalibratsiya oynalarini blok-sonlariga
// aylantirish uchun). Ikkisi qo'lda sinxronlanadi — birini o'zgartirsangiz,
// ikkinchisini ham o'zgartiring.
const WORKLET_CHUNK_SAMPLES = 2048;
const CHUNK_DURATION_MS = (WORKLET_CHUNK_SAMPLES / TARGET_SAMPLE_RATE) * 1000; // ~128ms

// --- C1: yengil clipping-himoyasi (og'ir DynamicsCompressorNode o'rniga) ---
// Ilgari bitta DynamicsCompressorNode (threshold/knee/ratio/attack/release
// bilan to'liq dinamika-ishlov berish) HAM VAD-tahlil, HAM yozib olish
// yo'liga ulangan edi — bu nafaqat og'ir, balki ovoz tembrini tekislab,
// ikki spikerning embeddinglarini bir-biriga yaqinlashtirib yuborardi (A1
// muammosini kuchaytirardi). Endi: faqat past gain + yumshoq (tanh-asosidagi)
// soft-clip — signalni "ezmaydi", faqat haddan tashqari balandlikda
// raqamli clipping'ning OLDINI oladi.
const LIMITER_PRE_GAIN = 0.9;
const SOFT_CLIP_DRIVE = 1.5;

// --- C2: moslashuvchan VAD ---
// Sessiya boshida shuncha millisoniya fon-shovqin o'lchanadi (bu vaqtda
// turn aniqlash mantiqi ishlamaydi) va NOISE_FLOOR_MULTIPLIER'ga
// ko'paytirilib dinamik chegaraga aylanadi — qattiq SPEECH_RMS_THRESHOLD
// o'rniga. Shovqin darajasi vaqt o'tishi bilan sekin (EMA) yangilanib
// boradi, FAQAT "gapirilmayotgan" paytlarda (aks holda nutqning o'zi
// "shovqin" deb hisoblanib ketardi).
const NOISE_FLOOR_CALIBRATION_MS = 1000;
const NOISE_FLOOR_MULTIPLIER = 3;
const NOISE_FLOOR_EMA_ALPHA = 0.05;

// REGRESSIYA TUZATISHI (real testda topilgan: bitta uzun gap 3-4 ta
// ~1.5s'lik bo'lakka bo'linib ketgan edi). Ildiz sabab: "listening_started"
// ko'ringandan keyin odam odatda DARHOL gapira boshlaydi — 1 soniyalik
// jim turish HECH QACHON kafolatlanmagan. Eski kod kalibratsiya oynasidagi
// HAMMA namunaning ODDIY O'RTACHASINI olardi — agar odam shu oyna ICHIDA
// gapirsa, nutq energiyasi noiseFloor'ni haddan tashqari YUQORI qilib
// qo'yardi. Natijada dynamicThreshold (=noiseFloor*3) ham shunchalik
// yuqori chiqib ketardi — HAQIQIY, davom etayotgan nutqning o'zi
// (so'zlar orasidagi, undosh-oldi pastroq energiyali lahzalarida) doimiy
// shu (noto'g'ri yuqori) threshold'dan PAST qolib, 1100ms ichida "jimlik"
// deb (noto'g'ri) topilib, turn vaqtidan oldin kesilardi — bu ANIQ
// kuzatilgan simptomga mos keladi (qisqa, baland bo'lak threshold'dan
// o'tadi, keyin qolgan GAP shu threshold'dan past qolib ketadi).
//
// Tuzatish 1 — PERSENTIL, O'RTACHA EMAS: kalibratsiya namunalarining eng
// past NOISE_FLOOR_PERCENTILE qismi olinadi. Odam gapirayotganda ham,
// so'zlar orasida/undosh paytida har doim nisbatan PASTROQ energiyali
// lahzalar bo'ladi — past persentil shularni tutadi, nutq cho'qqilariga
// "aldanib qolmaydi" (oddiy o'rtachadan farqli).
const NOISE_FLOOR_PERCENTILE = 0.2;
// Tuzatish 2 — QATTIQ SHIP (ceiling): persentil ham, keyingi EMA
// yangilanishi ham HECH QACHON shu qiymatdan oshmaydi. Bu feedback-loop'ni
// (threshold juda yuqori bo'lsa, ko'proq haqiqiy nutq "jimlik" deb
// topiladi va EMA'ga qo'shilib, noiseFloor'ni YANADA oshiradi) BUTUNLAY
// to'xtatadi. 0.05 — real suhbat energiyasidan (odatda 0.02-0.1 RMS)
// pastroq, lekin haqiqiy jim xona shovqinidan ancha yuqori.
const MAX_NOISE_FLOOR = 0.05;
// Juda jim xonada (noise_floor deyarli 0) threshold ham deyarli 0'ga
// tushib, eng kichik havo shovqinini ham "nutq" deb hisoblab yubormasligi
// uchun past chegara (backend'dagi RMS_SILENCE_THRESHOLD bilan bir xil
// kattalik tartibida).
const MIN_NOISE_FLOOR = 0.003;

// Tuzatish 3 — RMS SMOOTHING: C1 noiseSuppression/autoGainControl'ni
// o'chirgani uchun (brauzerning o'zi ENDI signalni tekislamaydi) xom
// RMS bitta-bittalab (har 128ms blokda) ANCHA "shovqinli"/tez
// o'zgaruvchan bo'lib qoldi — bitta tasodifiy past bloq (masalan
// undosh/nafas) o'zi hech narsaga ta'sir qilmasa ham, calibration/EMA
// tufayli natijani buzishi mumkin edi. Shu sababli "speaking" qarori
// endi XOM emas, EKSPONENSIAL SILLIQLANGAN RMS asosida qabul qilinadi
// (audio-level ko'rsatkichi uchun ishlatiladigan AUDIO_LEVEL_SMOOTHING
// bilan BIR XIL mexanizm, shu VAD qarori uchun alohida nusxa).
const VAD_RMS_SMOOTHING_ALPHA = 0.5; // 0..1, kattaroq = tezroq (kamroq silliqlash)

// --- D1: NISBIY (near-field) VAD — shovqinli muhit tuzatishi ---
// MUAMMO: xonada boshqa odamlar ham gapirayotganda, ularning ovozi
// NOISE_FLOOR_MULTIPLIER asosidagi bazaviy `dynamicThreshold`dan baland
// bo'lib qolishi mumkin (chunki bu threshold faqat gaplar ORASIDAGI haqiqiy
// jimlikka moslashadi, "boshqa odam gapirmoqda" holatiga emas). Natijada
// foydalanuvchi gapirib to'xtagandan keyin ham, fondagi ovoz shu bazaviy
// threshold'dan baland bo'lib qolgani uchun VAD "hali gapirilmoqda" deb
// hisoblaydi, SILENCE_DURATION_MS hech qachon to'lmaydi va turn abadiy
// ochiq qolib ketadi.
//
// YECHIM: mikrofonga ENG YAQIN turgan odam deyarli doim ENG BALAND RMS'ga
// ega bo'ladi (masofa kvadrati bo'yicha signal kuchi kamayadi). Shu sababli
// faol turn davomida "joriy spiker"ning darajasini (`speechLevelRef`, sekin
// EMA) kuzatib boramiz va navbatdagi freym shu darajaning NEAR_FIELD_RATIO
// ulushidan PAST bo'lsa — garchi bazaviy threshold'dan baland bo'lsa ham —
// buni "fon" (jimlik) deb hisoblaymiz. Bazaviy threshold HAR DOIM pastki
// chegara sifatida saqlanadi (`Math.max(dynamicThreshold, ...)`), shuning
// uchun jim xonada (fon yo'q) xatti-harakat O'ZGARMAYDI: speechLevel past
// bo'lsa, nearThreshold = dynamicThreshold bilan bir xil bo'lib qoladi.
const NEAR_FIELD_RATIO = 0.35; // 0..1 — joriy spiker darajasining shuncha ulushidan past freym "fon" deb hisoblanadi
const SPEECH_LEVEL_EMA_ALPHA = 0.1; // sekin EMA — bir martalik baland fon-impulsi speechLevel'ni darhol ko'tarib yubormasin

// --- D1: turn boshida "fon shovqinidan tasodifan ochilish"ning oldini
// olish — nutq hech bo'lmasa shuncha millisoniya UZLUKSIZ chegaradan
// yuqori bo'lgandagina turn RASMAN boshlanadi. PRE_ROLL_MS (300ms) bu
// qiymatdan katta bo'lishi SHART — shunda tasdiqlash davridagi audio ham
// (preRoll orqali) yo'qolmaydi, alohida bufer kerak emas.
const MIN_SPEECH_START_MS = 150;

// Muqobil/qo'shimcha variant (bu bosqichda ISHLATILMAGAN, faqat
// hujjatlashtirilgan — qarang fayl oxiridagi "@ricky0123/vad-web" izohi):
// Silero VAD (@ricky0123/vad-web) neyron-tarmoq asosidagi, shovqinga
// chidamliroq nutq aniqlashni taqdim etadi, lekin alohida model
// yuklanishini (WASM+ONNX, ~1-2MB) va butunlay boshqa integratsiya
// modelini talab qiladi.

// TARIX (1100 → 1400 → 2000ms): bu qiymat ilgari asta-sekin oshirilgan edi,
// chunki ikkinchi tilda gapirayotgan odam ko'pincha gap O'RTASIDA 1.4-2s
// "o'ylanib" to'xtab qoladi — qisqa chegara buni (noto'g'ri) "gap tugadi"
// deb bo'lib yuborardi, backend esa (eski MERGE_GAP_S, faqat <0.5s'lik
// davomlarni ushlagan) buni tuzatolmasdi.
//
// TEZLIK TUZATISHI (2000 → 1200, maqsad: gap tugashi -> tarjima ~2s): ikki
// sabab bilan endi xavfsiz pasaytirildi:
//   1. useDialogSession.js'dagi noiseFloor "qulflanib qolish" xatosi
//      tuzatildi (handleWorkletChunk'dagi EMA endi faqat faol turn YO'Q
//      paytda yangilanadi) — avvalgi gap bo'linishlarining ko'pi aslida
//      shu xatodan (VAD butunlay "nutq"ni yo'qotib qo'yishi) bo'lgan
//      bo'lishi mumkin, 2000ms esa shuni yashirib turgan.
//   2. Backend endi RETRO_MERGE_WINDOW_S (dialog.py, 2.5s) bilan
//      RETROAKTIV birlashtirishni qo'llaydi: agar shu past chegara
//      tufayli gap baribir ikkiga bo'linib ketsa, backend buni HECH
//      QANDAY QO'SHIMCHA KUTISHSIZ (STT darhol ishlaydi) keyinroq
//      avtomatik qo'shib, qayta tarjima qiladi — xavfsizlik to'ri endi
//      frontend tarafida emas, backend tarafida.
const SILENCE_DURATION_MS = 1200;

// --- C3: pre-roll ---
// "Nutq boshlandi" deb topilgan ONDAN OLDINGI shuncha millisoniya ham
// yangi turnga qo'shiladi — VAD energiya-chegarasi kesib o'tilguncha
// odatda allaqachon bir necha o'nlab millisoniya haqiqiy nutq o'tib
// ketgan bo'ladi (masalan yumshoq undosh bilan boshlangan so'z).
const PRE_ROLL_MS = 300;
const PRE_ROLL_CHUNK_COUNT = Math.ceil(PRE_ROLL_MS / CHUNK_DURATION_MS);

// --- C4: majburiy kesishni yumshatish ---
// Ilgari MAX_TURN_DURATION_MS=12000'da SO'ZSIZ, so'z o'rtasida ham kesib
// yuborilardi. Endi: shu vaqtdan keyin birinchi QISQA (SOFT_CUT_PAUSE_MS)
// pauza paytida kesiladi — odatda bu nafas olish/gap orasidagi tabiiy
// to'xtash bo'ladi, SILENCE_DURATION_MS (1400ms, "turn tugadi" degani)
// kabi uzoq emas — bu ikkisi ATAYLAB farq qiladi, lekin ZID EMAS: ular
// vaqt bo'yicha bir-biridan AJRATILGAN ikki rejim. SOFT_CUT_PAUSE_MS
// FAQAT turnElapsed >= SOFT_CUT_AFTER_MS (15s) bo'lgandan KEYIN
// tekshiriladi (handleWorkletChunk'dagi `else if`ga qarang) — shu
// vaqtgacha YAGONA qaror SILENCE_DURATION_MS asosida beriladi, ikkisi
// HECH QACHON bir vaqtda ishlamaydi. Agar hech qanday pauza (hatto qisqa ham) umuman
// bo'lmasa, HARD_CUT_MAX_MS'da baribir majburan kesiladi — cheksiz
// cho'zilib ketmasligi uchun oxirgi chora.
const SOFT_CUT_AFTER_MS = 15000;
const SOFT_CUT_PAUSE_MS = 300;
const HARD_CUT_MAX_MS = 18000;

const AUDIO_LEVEL_RMS_MAX = 0.15; // shu RMS qiymatida audioLevel = 1 ga yetadi
const AUDIO_LEVEL_SMOOTHING = 0.4; // 0..1, kattaroq = tezroq (kamroq silliqlash)

const DEBUG_VAD = false;

// DIALOG_PLAN.md 1-bosqich (D2): backend endi turn_skipped bilan birga aniq
// `reason` yuboradi (dialog.py'ga qarang) — ilgari bularning HAMMASI bir xil
// "Ovoz aniq eshitilmadi" xabari bilan ko'rsatilardi, garchi sabablar
// (masalan "juda qisqa" va "tarjima xatosi qildi") foydalanuvchi uchun
// butunlay boshqa narsa bo'lsa ham. Ro'yxatda yo'q/kutilmagan reason uchun
// DEFAULT ishlatiladi.
//
// Spiker aniqlash endi cheklanmagan sonli, FAQAT ovoz asosidagi onlayn
// klasterlash orqali ishlaydi (speaker_diart.py) — har bir turn ALBATTA
// biror spikerga (yangisi bo'lsa ham) tayinlanadi, shuning uchun eski
// "ambiguous_speaker" skip-sababi ENDI HECH QACHON yuborilmaydi (olib
// tashlandi) — noaniq hollarda backend endi turnni ko'rsatadi, faqat
// `uncertain=true` bayrog'i bilan (qarang DialogBubble.jsx'dagi "?"
// belgisi va "spikerni almashtirish" menyusi).
const SKIP_REASON_MESSAGES = {
  too_short: "Juda qisqa gap — tashlab yuborildi.",
  unknown_speaker: "Ovoz aniq eshitilmadi — yana gapirib ko'ring.",
  empty_transcription: "Hech narsa eshitilmadi.",
  processing_error: "Ichki xatolik yuz berdi — qayta urinib ko'ring.",
  translation_error: "Tarjima qilishda xatolik yuz berdi.",
  empty_audio: "Audio bo'sh keldi — tashlab yuborildi.",
  // Real testda topilgan 2-muammo: tanlangan ikki tildan TASHQARI tilda
  // gapirilganda (masalan o'zbekcha, ikkala speaker en/ru tanlagan bo'lsa)
  // backend endi buni alohida sabab bilan belgilaydi (stt.py'dagi
  // pair_mass/raw_top_lang tekshiruvi, qarang dialog.py).
  unsupported_language: "Tanlangan tillardan boshqa tilda gapirildi.",
};
const SKIP_REASON_DEFAULT = "Ovoz aniq eshitilmadi — yana gapirib ko'ring.";

// --- C1: yengil soft-clip egri chizig'i (WaveShaperNode uchun) ---
// tanh(x * DRIVE) — kichik amplitudalarda deyarli chiziqli (signal
// "ezilmaydi"), ±1'ga yaqinlashganda esa yumshoq to'yinadi (qattiq raqamli
// kesish o'rniga). tanh(DRIVE)'ga bo'lish — egri chiziq ±1 nuqtalarida
// ANIQ ±1'ga tegishini ta'minlaydi (unity gain o'rtacha signal uchun).
function _buildSoftClipCurve(samples = 1024) {
  const curve = new Float32Array(samples);
  const normalizer = Math.tanh(SOFT_CLIP_DRIVE);
  for (let i = 0; i < samples; i++) {
    const x = (i / (samples - 1)) * 2 - 1;
    curve[i] = Math.tanh(x * SOFT_CLIP_DRIVE) / normalizer;
  }
  return curve;
}

function _computeRms(float32Chunk) {
  let sumSquares = 0;
  for (let i = 0; i < float32Chunk.length; i++) {
    sumSquares += float32Chunk[i] * float32Chunk[i];
  }
  return Math.sqrt(sumSquares / float32Chunk.length);
}

function _mergeFloat32Chunks(chunks) {
  let totalLength = 0;
  for (const c of chunks) totalLength += c.length;
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }
  return merged;
}

export function useDialogSession() {
  const [phase, setPhase] = useState("idle"); // idle | listening | ended | error
  const [messages, setMessages] = useState([]);
  // Backend'ning avtomatik spiker-birlashtirish ("speakers_merged") va
  // spiker-o'chirish ("speaker_removed") voqealari — DialogSessionView.jsx
  // o'zining spiker ism xaritasini (speakerNames) shu orqali yangilaydi
  // (qarang undagi useEffect). Append-only ro'yxat — eski voqealar
  // sessiya davomida saqlanadi, chunki DialogSessionView ularni allaqachon
  // qayta ishlangan deb kuzatib boradi (indeks orqali), shuning uchun
  // bu yerda cheklov/tozalash shart emas (sessiyaning o'zi uzoq davom
  // etmaydi).
  const [speakerEvents, setSpeakerEvents] = useState([]);
  // Spiker rangi UZLUKSIZ saqlanishi uchun: {speakerId: order} — har bir
  // spiker ID birinchi marta "turn" statusida ko'ringanda shu yerga
  // navbatdagi BUTUN SON ("order") bilan QO'SHILADI va keyin HECH QACHON
  // o'zgartirilmaydi/qayta hisoblanmaydi (DialogSessionView.jsx shu
  // order'ni `% palette.length` orqali aniq rangga aylantiradi). ESKI
  // yondashuv — har render'da `messages` massividan "kim birinchi
  // uchraganini" qayta hisoblash — bubble o'chirilganda/birlashtirilganda
  // qolgan spikerlarning NISBIY tartibi (demak rangi) siljib ketishi
  // MUMKIN edi (masalan ikkita spiker orasidagi "ko'prik" bubble
  // o'chirilsa). Bu yerda, aksincha, tayinlash FAQAT bitta joyda —
  // handleServerMessage'dagi "turn" case'ida, YANGI (hali xaritada yo'q)
  // ID birinchi ko'ringanda — sodir bo'ladi, shuning uchun keyingi HECH
  // QANDAY o'chirish/birlashtirish buni qayta tartiblay olmaydi.
  const [speakerColorOrder, setSpeakerColorOrder] = useState({});
  // React state YANGILANISHI asinxron bo'lgani uchun (bir nechta "turn"
  // xabari bitta render oynasida ketma-ket kelsa, `speakerColorOrder`ning
  // "eski" qiymatiga tayanib ikkinchi marta BIR XIL order'ni berib
  // qo'yish xavfi bor edi) — "keyingi band bo'lmagan order" alohida,
  // SINXRON ref'da yuritiladi (bu ref FAQAT event-handler ichida, render
  // vaqtida EMAS, o'qiladi/yangilanadi — React hook qoidalariga mos).
  const nextSpeakerColorOrderRef = useRef(0);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null); // vaqtinchalik "eshitilmadi" kabi xabarlar uchun, error state'dan alohida
  const [audioLevel, setAudioLevel] = useState(0);
  // DIALOG_PLAN.md 4-bosqich: backend navbatdagi turnni qayta ishlay
  // boshlaganda (`processing` statusi) true, natija ("turn"/"turn_skipped")
  // kelganda yana false — DialogSessionView shu bilan "Qayta ishlanmoqda…"
  // ko'rsatkichini chiqaradi.
  const [isProcessing, setIsProcessing] = useState(false);

  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const stoppedRef = useRef(false);
  const noticeTimeoutRef = useRef(null);

  const audioContextRef = useRef(null);
  const workletNodeRef = useRef(null);

  // --- VAD/turn holati (React state EMAS — har worklet-xabarida
  // o'zgaradi, re-render kerak emas) ---
  const smoothedLevelRef = useRef(0);
  // REGRESSIYA TUZATISHI: endi sum/count (oddiy o'rtacha) EMAS — persentil
  // hisoblash uchun kalibratsiya oynasidagi barcha XOM RMS namunalari
  // massiv sifatida saqlanadi (qarang NOISE_FLOOR_PERCENTILE izohi).
  const calibrationStartedAtRef = useRef(null);
  const calibrationSamplesRef = useRef([]);
  const noiseFloorRef = useRef(null);
  // REGRESSIYA TUZATISHI: "speaking" qarori endi shu silliqlangan qiymat
  // asosida qabul qilinadi, xom `rms`ning o'zi emas (qarang
  // VAD_RMS_SMOOTHING_ALPHA izohi).
  const smoothedRmsRef = useRef(0);
  // D1: faol turn davomida "joriy (eng yaqin/baland) spiker"ning
  // darajasi — qarang NEAR_FIELD_RATIO izohi yuqorida. Faol turn YO'Q
  // paytda 0 (neytral — nearThreshold shunda dynamicThreshold bilan
  // bir xil bo'lib qoladi).
  const speechLevelRef = useRef(0);
  // D1: hali RASMIY boshlanmagan ("tasdiqlanmagan") nutq qachon
  // boshlanganini belgilaydi — MIN_SPEECH_START_MS to'lmaguncha turn
  // ochilmaydi (qarang handleWorkletChunk). `null` = hozir nomzod yo'q.
  const pendingSpeechStartAtRef = useRef(null);
  const preRollChunksRef = useRef([]);
  const turnChunksRef = useRef(null); // null = hozir faol turn yo'q
  const turnStartedAtRef = useRef(null);
  const silenceStartedAtRef = useRef(null);
  // C5: TTS ijro etilayotganda true — shu paytda hech qanday audio
  // VAD/turn mantiqiga kiritilmaydi (qarang pauseMic/resumeMic).
  const micPausedRef = useRef(false);
  // Diagnostika: har DIAG_LOG_INTERVAL_MSda bir marta holatni "[VAD]" bilan
  // konsolga chiqarish uchun.
  const lastDiagLogAtRef = useRef(0);

  const cleanupMedia = useCallback(() => {
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => track.stop());
      } catch (err) {
        console.error("[useDialogSession] error stopping media tracks", err);
      }
      streamRef.current = null;
    }
  }, []);

  const closeSocket = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (err) {
        console.error("[useDialogSession] error closing socket", err);
      }
      wsRef.current = null;
    }
  }, []);

  const resetVadState = useCallback(() => {
    smoothedLevelRef.current = 0;
    calibrationStartedAtRef.current = null;
    calibrationSamplesRef.current = [];
    noiseFloorRef.current = null;
    smoothedRmsRef.current = 0;
    speechLevelRef.current = 0;
    pendingSpeechStartAtRef.current = null;
    preRollChunksRef.current = [];
    turnChunksRef.current = null;
    turnStartedAtRef.current = null;
    silenceStartedAtRef.current = null;
    micPausedRef.current = false;
  }, []);

  const stopListenLoop = useCallback(() => {
    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.port.onmessage = null;
        workletNodeRef.current.disconnect();
      } catch (err) {
        console.error("[useDialogSession] error disconnecting worklet node", err);
      }
      workletNodeRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (err) {
        console.error("[useDialogSession] error closing audio context", err);
      }
      audioContextRef.current = null;
    }

    resetVadState();
    setAudioLevel(0);
  }, [resetVadState]);

  // TEZLIK TUZATISHI (backend MERGE_GAP_S'ni faqat kerak bo'lganda
  // qo'llashi uchun): ichki (batafsil) `reason`larni backend kutadigan
  // IKKI toifaga qisqartiradi — dialog.py'dagi dispatch qarori faqat
  // shuni biladi. "silence" — odam o'zi jimlik bilan (SILENCE_DURATION_MS)
  // to'xtadi, demak davomi (agar bo'lsa) ALLAQACHON shu muddat ichida
  // YANGI, alohida audio sifatida kelib bo'lardi — backend qo'shimcha
  // kutishi shart emas. "max_duration" — frontend gapni MAJBURAN kesdi
  // (soft/hard cut) yoki boshqa abrupt sabab (TTS pauzasi) — davom etish
  // ehtimoli yuqori, backend eskicha qisqa muddat kutib ko'radi.
  function _categorizeFinalizeReason(reason) {
    if (reason === "silence_timeout") return "silence";
    return "max_duration";
  }

  const sendTurnAudio = useCallback((float32Array, durationMs, reason) => {
    const bytes = float32Array.byteLength;
    if (
      !stoppedRef.current &&
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      float32Array.length > 0
    ) {
      try {
        const wireReason = _categorizeFinalizeReason(reason);
        // Backend (dialog.py'dagi handle_client_message, action="turn_meta")
        // buni AYNAN keyingi binary chunk uchun amal qiladi deb kutadi —
        // shuning uchun matnli xabar har doim audio'dan OLDIN yuboriladi
        // (WebSocket bitta ulanishda xabarlarni ketma-ket saqlaydi).
        wsRef.current.send(JSON.stringify({ action: "turn_meta", reason: wireReason }));
        wsRef.current.send(float32Array.buffer);
        console.log(
          `[VAD] turn SENT to backend durationMs=${durationMs.toFixed(0)} bytes=${bytes} ` +
            `reason=${reason} wireReason=${wireReason}`
        );
      } catch (err) {
        console.error("[useDialogSession] error sending turn audio", err);
      }
    } else {
      console.warn(
        `[VAD] turn NOT SENT (ws closed or empty) durationMs=${durationMs.toFixed(0)} bytes=${bytes} ` +
          `wsState=${wsRef.current ? wsRef.current.readyState : "null"} stopped=${stoppedRef.current}`
      );
    }
  }, []);

  const finalizeTurn = useCallback(
    (reason) => {
      const chunks = turnChunksRef.current;
      const turnStartedAt = turnStartedAtRef.current;
      turnChunksRef.current = null;
      turnStartedAtRef.current = null;
      silenceStartedAtRef.current = null;
      speechLevelRef.current = 0;
      if (!chunks || chunks.length === 0) return;

      const merged = _mergeFloat32Chunks(chunks);
      const durationMs = (merged.length / TARGET_SAMPLE_RATE) * 1000;
      console.log(
        `[VAD] speech ENDED at=${new Date().toISOString()} reason=${reason} ` +
          `turnDurationMs=${durationMs.toFixed(0)} wallClockMs=${
            turnStartedAt ? Date.now() - turnStartedAt : "?"
          }`
      );
      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] turn FINALIZED reason=${reason} samples=${merged.length} ` +
            `durationMs=${durationMs.toFixed(0)}`
        );
      }
      sendTurnAudio(merged, durationMs, reason);
    },
    [sendTurnAudio]
  );

  const handleWorkletChunk = useCallback(
    (chunk) => {
      if (stoppedRef.current) return;
      const now = Date.now();
      const rms = _computeRms(chunk);

      // --- audio-level ko'rsatkichi (UI) — mikrofon pauza qilingan
      // bo'lsa ham ko'rsatkich 0'ga yaqinlashib davom etadi, zararsiz ---
      const normalizedLevel = Math.min(1, rms / AUDIO_LEVEL_RMS_MAX);
      smoothedLevelRef.current += (normalizedLevel - smoothedLevelRef.current) * AUDIO_LEVEL_SMOOTHING;
      setAudioLevel(smoothedLevelRef.current);

      // --- C5: TTS ijro etilayotganda hech narsa qabul qilinmaydi/yozilmaydi ---
      if (micPausedRef.current) {
        return;
      }

      // --- C2: fon-shovqin kalibratsiyasi (sessiya boshida bir marta) ---
      // REGRESSIYA TUZATISHI: endi XOM rms'ning o'zi emas, massivga
      // to'planadi — foydalanuvchi shu 1 soniya ICHIDA gapira boshlagan
      // taqdirda ham, pastdagi persentil-hisob nutq cho'qqilariga
      // "aldanib qolmaydi" (qarang NOISE_FLOOR_PERCENTILE izohi yuqorida).
      if (calibrationStartedAtRef.current === null) {
        calibrationStartedAtRef.current = now;
      }
      const inCalibration = now - calibrationStartedAtRef.current < NOISE_FLOOR_CALIBRATION_MS;
      if (inCalibration) {
        calibrationSamplesRef.current.push(rms);
        return;
      }
      if (noiseFloorRef.current === null) {
        const samples = calibrationSamplesRef.current.slice().sort((a, b) => a - b);
        const idx =
          samples.length > 0
            ? Math.min(samples.length - 1, Math.floor(samples.length * NOISE_FLOOR_PERCENTILE))
            : -1;
        const estimate = idx >= 0 ? samples[idx] : MIN_NOISE_FLOOR;
        noiseFloorRef.current = Math.min(MAX_NOISE_FLOOR, Math.max(MIN_NOISE_FLOOR, estimate));
        if (DEBUG_VAD) {
          console.log(
            `[VAD-DEBUG] noise floor calibrated (p${NOISE_FLOOR_PERCENTILE * 100} of ` +
              `${samples.length} samples): ${noiseFloorRef.current.toFixed(5)}`
          );
        }
      }

      // REGRESSIYA TUZATISHI: xom `rms` o'rniga EKSPONENSIAL SILLIQLANGAN
      // qiymat threshold bilan solishtiriladi — C1 noiseSuppression/AGC'ni
      // o'chirgani uchun xom RMS blokdan-blokka ancha "shovqinli" bo'lib
      // qoldi, bitta tasodifiy past bloq butun gapni erta "jimlik" deb
      // ko'rsatib yuborishi mumkin edi.
      smoothedRmsRef.current += VAD_RMS_SMOOTHING_ALPHA * (rms - smoothedRmsRef.current);

      const dynamicThreshold = Math.max(MIN_NOISE_FLOOR, noiseFloorRef.current) * NOISE_FLOOR_MULTIPLIER;
      const speaking = smoothedRmsRef.current >= dynamicThreshold;
      const hasActiveTurn = turnChunksRef.current !== null;

      // --- D1: nisbiy (near-field) VAD — qarang NEAR_FIELD_RATIO izohi
      // yuqorida. FAQAT faol turn davomida qo'llaniladi: turn hali
      // boshlanmagan bo'lsa (fon-shovqin kalibratsiyasi/gaplar orasi),
      // bazaviy `speaking` ham hozircha yetarli signal.
      let nearThreshold = dynamicThreshold;
      let nearFieldSpeaking = speaking;
      let silenceReason = speaking ? "above_threshold" : "below_threshold";
      if (hasActiveTurn) {
        nearThreshold = Math.max(dynamicThreshold, speechLevelRef.current * NEAR_FIELD_RATIO);
        nearFieldSpeaking = smoothedRmsRef.current >= nearThreshold;
        silenceReason = nearFieldSpeaking ? "near_field_speech" : "far_field_background";
        if (nearFieldSpeaking) {
          // Sekin EMA — joriy ("eng yaqin") spikerning darajasini vaqt
          // bo'yicha kuzatib boradi, bir martalik baland fon-impulsidan
          // darhol "aldanib" yuqoriga sirg'alib ketmaydi.
          speechLevelRef.current += SPEECH_LEVEL_EMA_ALPHA * (smoothedRmsRef.current - speechLevelRef.current);
        }
      }

      // Diagnostika: har 2 soniyada joriy holatni chiqaramiz.
      if (now - lastDiagLogAtRef.current >= 2000) {
        lastDiagLogAtRef.current = now;
        console.log(
          `[VAD] status rms=${rms.toFixed(5)} smoothedRms=${smoothedRmsRef.current.toFixed(5)} ` +
            `noiseFloor=${noiseFloorRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
            `speechLevel=${speechLevelRef.current.toFixed(5)} nearThreshold=${nearThreshold.toFixed(5)} ` +
            `frame=${nearFieldSpeaking ? "speech" : "silence"}(${silenceReason}) ` +
            `turn=${hasActiveTurn ? "recording" : "idle"}`
        );
      }

      // Shovqin darajasi FAQAT "gapirilmayotgan" paytlarda sekin
      // yangilanadi — aks holda nutqning o'zi "yangi shovqin darajasi"
      // deb qabul qilinib, threshold nutq energiyasiga qarab o'sib
      // ketardi (va keyingi gaplarni tutib qololmay qolardi). QATTIQ SHIP
      // (MAX_NOISE_FLOOR) shu yerda ham qo'llanadi — EMA hech qachon
      // noiseFloor'ni xavfsiz chegaradan tashqariga chiqarib yubormaydi.
      //
      // XATO TUZATISHI (uzoq, to'xtovsiz gapda VAD butunlay "qulflanib
      // qolishi"): bu yangilanish ILGARI turnChunksRef faol bo'lsa HAM
      // ishlardi. "noiseFloor" EMA'si har doim `smoothedRmsRef`GA QARAB
      // intiladi (target aynan shu), shuning uchun agar nutq ICHIDAGI
      // bitta bo'g'in/so'z orasidagi tabiiy pastroq-energiyali lahza
      // noto'g'ri "jimlik" deb topilsa (masalan undosh yoki nafas),
      // noiseFloor shu PAYTDAGI (hali ham nutq bo'lgan) energiyaga qarab
      // yuqoriga suriladi. dynamicThreshold = noiseFloor*3 bo'lgani uchun,
      // bu doim o'z-o'zini qondiruvchi holatga olib keladi: "joriy energiya
      // = yangi shovqin darajasi", va 3x shart HECH QACHON qondirilmaydi —
      // VAD go'yo odam to'xtab qolganini "aniqlaydi" va undan keyin
      // boshqa hech qanday nutqni "nutq" deb topa olmay qoladi (chunki
      // threshold doim joriy ovoz darajasidan yuqori bo'lib qolaveradi).
      // Bu aynan amaliyotda ko'rilgan "20-30s gapirilgan, lekin faqat 1ta
      // ~4-5s turn kelgan, qolgani umuman yuborilmagan" nosozlikka mos
      // keladi. Tuzatish: shovqin darajasini FAQAT faol turn YO'Q paytda
      // (haqiqiy, gaplar ORASIDAGI jimlikda) moslashtiramiz — nutq
      // ICHIDAGI vaqtinchalik pastroq lahzalar endi floor'ni yuqoriga
      // surmaydi.
      if (!speaking && !hasActiveTurn) {
        const updated = noiseFloorRef.current + NOISE_FLOOR_EMA_ALPHA * (smoothedRmsRef.current - noiseFloorRef.current);
        noiseFloorRef.current = Math.min(MAX_NOISE_FLOOR, Math.max(MIN_NOISE_FLOOR, updated));
      }

      if (DEBUG_VAD) {
        console.log(
          `[VAD-DEBUG] rms=${rms.toFixed(5)} smoothedRms=${smoothedRmsRef.current.toFixed(5)} ` +
            `noiseFloor=${noiseFloorRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
            `speechLevel=${speechLevelRef.current.toFixed(5)} nearThreshold=${nearThreshold.toFixed(5)} ` +
            `speaking=${speaking} nearFieldSpeaking=${nearFieldSpeaking} activeTurn=${hasActiveTurn}`
        );
      }

      // --- C3: pre-roll — faol turn bo'lmagandayam doim so'nggi
      // PRE_ROLL_CHUNK_COUNT blokni saqlab turamiz ---
      preRollChunksRef.current.push(chunk);
      if (preRollChunksRef.current.length > PRE_ROLL_CHUNK_COUNT) {
        preRollChunksRef.current.shift();
      }

      if (!hasActiveTurn) {
        if (speaking) {
          // D1: turn FAQAT nutq kamida MIN_SPEECH_START_MS UZLUKSIZ
          // chegaradan yuqori bo'lgandan KEYIN rasman boshlanadi — bitta
          // qisqa fon-impulsi (masalan uzoqdagi odamning bir so'zi) shu
          // bilan turn ochib yubormaydi. `pendingSpeechStartAtRef` — shu
          // nomzod davrining boshlanish vaqti; har qanday "gapirilmayapti"
          // freymi buni DARHOL bekor qiladi (pastdagi else branch).
          if (pendingSpeechStartAtRef.current === null) {
            pendingSpeechStartAtRef.current = now;
          }
          const pendingElapsedMs = now - pendingSpeechStartAtRef.current;
          if (pendingElapsedMs >= MIN_SPEECH_START_MS) {
            // Yangi turn — PRE-ROLL BILAN boshlanadi (C3): nutq energiya-
            // chegarasidan "rasman" o'tgunga qadar allaqachon o'tib ketgan
            // bir necha bloq (odatda so'z boshidagi yumshoq tovush, SHU
            // JUMLADAN tasdiqlash davridagi audio — PRE_ROLL_MS >
            // MIN_SPEECH_START_MS bo'lgani uchun allaqachon preRoll'da) ham
            // qo'shiladi.
            turnChunksRef.current = [...preRollChunksRef.current];
            turnStartedAtRef.current = pendingSpeechStartAtRef.current;
            silenceStartedAtRef.current = null;
            speechLevelRef.current = smoothedRmsRef.current;
            pendingSpeechStartAtRef.current = null;
            console.log(
              `[VAD] speech STARTED at=${new Date(now).toISOString()} rms=${rms.toFixed(5)} ` +
                `smoothedRms=${smoothedRmsRef.current.toFixed(5)} threshold=${dynamicThreshold.toFixed(5)} ` +
                `confirmedAfterMs=${pendingElapsedMs.toFixed(0)}`
            );
            if (DEBUG_VAD) {
              console.log(`[VAD-DEBUG] turn STARTED (preRollChunks=${preRollChunksRef.current.length})`);
            }
          }
        } else {
          pendingSpeechStartAtRef.current = null;
        }
        return;
      }

      // Faol turn davom etadi — gapirilmayotgan bo'lsa ham audio
      // qo'shib boriladi (turn SILENCE_DURATION_MS to'liq o'tgunga qadar
      // yopilmaydi, shuning uchun qisqa pauzalar gap ICHIDA qoladi).
      turnChunksRef.current.push(chunk);

      // D1: silence-taymer endi BAZAVIY `speaking` emas, NISBIY
      // `nearFieldSpeaking`ga qarab yuritiladi — fondagi odam(lar)ning
      // ovozi bazaviy threshold'dan baland bo'lsa ham, bu ENDI "nutq
      // davom etmoqda" deb hisoblanmaydi, taymer shovqin sifatida ishlayveradi.
      if (nearFieldSpeaking) {
        silenceStartedAtRef.current = null;
      } else if (silenceStartedAtRef.current === null) {
        silenceStartedAtRef.current = now;
      }

      const turnElapsedMs = now - turnStartedAtRef.current;
      const silenceElapsedMs = silenceStartedAtRef.current === null ? 0 : now - silenceStartedAtRef.current;

      if (silenceElapsedMs >= SILENCE_DURATION_MS) {
        finalizeTurn("silence_timeout");
      } else if (turnElapsedMs >= SOFT_CUT_AFTER_MS && silenceElapsedMs >= SOFT_CUT_PAUSE_MS) {
        // C4: qattiq belgilangan vaqtda emas, SOFT_CUT_AFTER_MS'dan keyingi
        // BIRINCHI qisqa pauzada kesiladi — odatda bu so'z/gap chegarasi.
        finalizeTurn("soft_cut_pause");
      } else if (turnElapsedMs >= HARD_CUT_MAX_MS) {
        // C4: hech qanday pauza (hatto qisqa ham) topilmadi — oxirgi
        // chora, cheksiz cho'zilib ketmasligi uchun majburan kesiladi.
        finalizeTurn("hard_cut_max_duration");
      }
    },
    [finalizeTurn]
  );

  const startListenLoop = useCallback(async () => {
    if (!streamRef.current) return;

    stopListenLoop();

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    let audioContext;
    try {
      audioContext = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
    } catch (err) {
      console.error(
        "[useDialogSession] failed to create AudioContext at 16kHz, falling back to default rate",
        err
      );
      audioContext = new AudioContextClass();
    }

    // MUHIM XATO TUZATISHI: yangi yaratilgan AudioContext ko'p brauzerlarda
    // (ayniqsa Chrome) "suspended" holatda boshlanadi, AGAR u to'g'ridan-
    // to'g'ri, SINXRON tugma-bosish ichida yaratilmagan bo'lsa — bu yerda
    // esa AudioContext "listening_started" WebSocket xabari kelgandan
    // keyin (asinxron) yaratiladi. "suspended" holatda AudioWorkletNode'ning
    // process() funksiyasi UMUMAN chaqirilmaydi — hech qanday xato
    // chiqmaydi, shunchaki hech narsa ishlamaydi (audioLevel/to'lqin
    // hech qachon o'zgarmaydi, birorta ham turn yuborilmaydi). Shu sababli
    // resume() ANIQ, qo'lda chaqiriladi va natija tekshiriladi.
    try {
      await audioContext.resume();
    } catch (err) {
      console.error("[useDialogSession] audioContext.resume() failed", err);
    }
    if (audioContext.state !== "running") {
      console.error(
        `[useDialogSession] AudioContext holati "${audioContext.state}" ("running" kutilgan) — ` +
          "mikrofon tinglanmaydi."
      );
      setError("Audio tizimini ishga tushirishda xatolik yuz berdi — sahifani qayta yuklab ko'ring.");
      setPhase("error");
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
      // Zamonaviy Chrome/Firefox/Edge AudioContext konstruktoridagi
      // sampleRate so'rovini hurmat qiladi. Agar brauzer buni rad etsa,
      // backend'ga yuborilayotgan PCM NOTO'G'RI tezlikda bo'ladi (ovoz
      // tezlashtirilgan/sekinlashtirilgan eshitiladi, STT/diarizatsiya
      // ishlamaydi) — bu holat uchun JS-tarafida resampler YOZILMAGAN
      // (bu ishning ko'lamidan tashqarida qoldirildi), shuning uchun
      // bu xabar darhol ko'rinadigan, qattiq xato sifatida chiqariladi.
      console.error(
        `[useDialogSession] AudioContext sampleRate=${audioContext.sampleRate}, kutilgan ` +
          `${TARGET_SAMPLE_RATE} — backend'ga yuborilayotgan audio tezligi NOTO'G'RI bo'lishi mumkin!`
      );
    }

    try {
      await audioContext.audioWorklet.addModule("/pcm-worklet.js");
    } catch (err) {
      console.error("[useDialogSession] failed to load pcm-worklet.js", err);
      setError("Audio modulini yuklashda xatolik yuz berdi.");
      setPhase("error");
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    if (stoppedRef.current) {
      // Worklet yuklanayotgan paytda foydalanuvchi suhbatni to'xtatgan
      // bo'lishi mumkin — shu holda endi hech narsa ulanmaydi.
      try {
        audioContext.close();
      } catch {
        // e'tiborsiz qoldiriladi
      }
      return;
    }

    const source = audioContext.createMediaStreamSource(streamRef.current);

    // --- C1: yengil clipping-himoyasi — faqat YOZIB OLINADIGAN (va shu
    // bilan birga energiya/VAD hisoblash uchun ishlatiladigan, chunki
    // endi ikkisi BIR XIL signal) yo'lga ulanadi. Og'ir DynamicsCompressor
    // butunlay olib tashlandi — pastdagi "ESKI VA YANGI ZANJIRNI
    // SOLISHTIRISH" izohiga qarang.
    const limiterGain = audioContext.createGain();
    limiterGain.gain.value = LIMITER_PRE_GAIN;
    const limiterShaper = audioContext.createWaveShaper();
    limiterShaper.curve = _buildSoftClipCurve();
    source.connect(limiterGain);
    limiterGain.connect(limiterShaper);

    const workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
    limiterShaper.connect(workletNode);

    // AudioWorkletNode faqat audio-grafga (destination'gacha) ulangan
    // bo'lsagina process() muntazam chaqiriladi — biz mikrofonni
    // EHTIYOTDAN eshittirmaymiz, shuning uchun 0-gain orqali "jimlikka"
    // ulaymiz (standart AudioWorklet pattern).
    const silentSink = audioContext.createGain();
    silentSink.gain.value = 0;
    workletNode.connect(silentSink);
    silentSink.connect(audioContext.destination);

    workletNode.port.onmessage = (event) => handleWorkletChunk(event.data);

    audioContextRef.current = audioContext;
    workletNodeRef.current = workletNode;

    if (DEBUG_VAD) {
      console.log(`[VAD-DEBUG] listen loop STARTED, actual sampleRate=${audioContext.sampleRate}`);
    }
  }, [handleWorkletChunk, stopListenLoop]);

  // --- C5: TTS ijro etilayotganda VAD/yozib olish to'xtatiladi ---
  // DialogSessionView.jsx (usePlayAudio() bilan birga) TTS ijrosi
  // boshlanganda pauseMic()ni, tugaganda resumeMic()ni chaqiradi — shunda
  // karnaydan chiqayotgan tarjima ovozi mikrofon orqali yangi turn
  // sifatida qayta eshitilib ketmaydi.
  const pauseMic = useCallback(
    (reason = "unknown") => {
      if (micPausedRef.current) {
        console.log(`[VAD] pauseMic called again while already paused reason=${reason}`);
        return;
      }
      micPausedRef.current = true;
      console.log(`[VAD] pauseMic reason=${reason} at=${new Date().toISOString()}`);
      // Agar pauza paytida tasodifan faol turn bo'lsa (masalan foydalanuvchi
      // ijro tugmasini aynan gapirayotganda bossa) — uni yo'qotib
      // yubormasdan, darhol yakunlab yuboramiz.
      if (turnChunksRef.current !== null) {
        finalizeTurn("mic_paused_for_tts");
      }
      // TTS tovushi pre-roll'ga "ko'rinmas" tarzda kirib qolmasligi uchun
      // tozalanadi — pauza tugagach, pre-roll nolldan to'planadi.
      preRollChunksRef.current = [];
    },
    [finalizeTurn]
  );

  const resumeMic = useCallback((reason = "unknown") => {
    if (!micPausedRef.current) {
      console.log(`[VAD] resumeMic called while not paused reason=${reason}`);
    }
    micPausedRef.current = false;
    console.log(`[VAD] resumeMic reason=${reason} at=${new Date().toISOString()}`);
  }, []);

  const handleServerMessage = useCallback(
    (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw);
      } catch (err) {
        console.error("[useDialogSession] bad JSON from server", err, raw);
        return;
      }

      switch (msg.status) {
        case "listening_started":
          stoppedRef.current = false;
          setPhase("listening");
          startListenLoop();
          break;

        case "processing":
          // DIALOG_PLAN.md 4-bosqich: backend turn_worker() navbatdan bitta
          // turnni olib qayta ishlay boshlaganda yuboradi — STT/tarjima bir
          // necha soniya cho'zilishi mumkin, shu payt foydalanuvchiga biror
          // narsa ko'rinmasa "ishlamayapti" degan taassurot qoladi. "turn"
          // yoki "turn_skipped" kelganda (pastga qarang) avtomatik yopiladi.
          setIsProcessing(true);
          break;

        case "turn":
          setNotice(null);
          setIsProcessing(false);
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
            noticeTimeoutRef.current = null;
          }
          // Backend turnlarni KETMA-KET (bitta navbat/worker, dialog.py'ga
          // qarang) qayta ishlaydi, shuning uchun natijalar odatda KELISH
          // tartibida ham keladi — `seq` (audio backend'ga kelgan tartib)
          // bo'yicha saralash shunga qaramay xavfsizlik to'ri sifatida
          // saqlanadi.
          //
          // Qabul qilingan yondashuv: DARHOL ko'rsatish + sort, navbatga
          // qo'yib KUTDIRISH emas. Sabab: bu jonli suhbat tarjimoni —
          // asosiy qiymati tezlik (deyarli real vaqtda tarjima).
          //
          // DIALOG_PLAN.md 4-bosqich: endi shu `seq` MAVJUD bo'lsa
          // (correct_speaker orqali spiker tuzatilgandan keyin backend
          // XUDDI SHU seq bilan qayta "turn" yuboradi) — yangi bubble
          // qo'shish o'rniga MAVJUDINI almashtiramiz (upsert).
          setMessages((prev) => {
            const entry = {
              seq: msg.seq,
              // Spiker endi A/B emas, cheklanmagan sonli butun son ID
              // (speaker_diart.py'dagi onlayn klasterlash) — DialogSessionView
              // shuni "N-spiker" deb formatlab, rangli yorliq bilan ko'rsatadi.
              speaker: msg.speaker,
              originalText: msg.original,
              translation: msg.translation,
              sourceLang: msg.source_lang,
              targetLang: msg.target_lang,
              // Spiker aniqlash ishonchi past bo'lsa (uncertain band, juda
              // qisqa turn, yoki hali tasdiqlanmagan yangi spiker nomzodi)
              // — DialogBubble shuni ko'rib yorliq yonida "?" belgisini
              // chiqaradi, foydalanuvchi kerak bo'lsa qo'lda to'g'rilaydi.
              uncertain: Boolean(msg.uncertain),
              // Tarjima: DeepL (asosiy) + NLLB-600M (zaxira) — "deepl",
              // "nllb" yoki null (tarjima umuman chaqirilmagan, masalan
              // source==target). DialogBubble shuni ko'rib juda kichik
              // "DeepL"/"lokal" belgisini chiqaradi.
              mtProvider: msg.mt_provider ?? null,
              // IKKISI HAM (DeepL va NLLB) muvaffaqiyatsiz bo'lsa — true,
              // DialogBubble tarjima o'rniga kulrang "Tarjima vaqtincha
              // mavjud emas" matnini ko'rsatadi.
              translationError: Boolean(msg.translation_error),
            };
            const existingIdx = prev.findIndex((m) => m.seq === msg.seq);
            const next =
              existingIdx === -1
                ? [...prev, entry]
                : prev.map((m, i) => (i === existingIdx ? entry : m));
            next.sort((a, b) => a.seq - b.seq);
            return next;
          });

          // Rangni UZLUKSIZ saqlash uchun: shu spiker ID birinchi marta
          // ko'rinishi — qarang `speakerColorOrder` izohi yuqorida. Agar
          // ID ALLAQACHON xaritada bo'lsa (odatiy hol — mavjud spiker
          // yana gapirdi) hech narsa qilinmaydi, order O'ZGARMAYDI.
          setSpeakerColorOrder((prev) => {
            if (msg.speaker in prev) return prev;
            const order = nextSpeakerColorOrderRef.current;
            nextSpeakerColorOrderRef.current += 1;
            return { ...prev, [msg.speaker]: order };
          });
          break;

        case "turn_deleted":
          // Har bir bubble'dagi 🗑 tugmasi — backend turnni turn_history'dan
          // o'chirgandan keyin shu statusni yuboradi, bubble ekrandan
          // olib tashlanadi.
          setMessages((prev) => prev.filter((m) => m.seq !== msg.seq));
          break;

        case "speakers_merged": {
          // Backend ikkita spikerni (prototiplari juda o'xshash chiqqani
          // uchun) AVTOMATIK bitta spikerga birlashtirgan — BARCHA "from"
          // bubble'lari darhol "into"ga ko'chiriladi. Ism/rang meros olish
          // qarori (DialogSessionView.jsx'dagi speakerNames xaritasi) shu
          // voqeani kuzatib turgan useEffect orqali hal qilinadi.
          const { from, into } = msg;
          setMessages((prev) => prev.map((m) => (m.speaker === from ? { ...m, speaker: into } : m)));
          setSpeakerEvents((prev) => [...prev, { type: "merged", from, into }]);
          break;
        }

        case "speaker_removed":
          // Bitta spikerning YAGONA ovoz namunasi tuzatish/o'chirish orqali
          // olib tashlangandan keyin backend shu spiker profilini butunlay
          // o'chiradi — DialogSessionView.jsx shu spikerning ismini/
          // ro'yxatini o'z xaritasidan tozalaydi (bubble'larning o'zi
          // ekranda QOLADI, faqat endi default ism/rang bilan).
          setSpeakerEvents((prev) => [...prev, { type: "removed", speaker: msg.speaker }]);
          break;

        case "turn_skipped":
          console.warn("[useDialogSession] turn skipped", msg.reason);
          setIsProcessing(false);
          setNotice(SKIP_REASON_MESSAGES[msg.reason] || SKIP_REASON_DEFAULT);
          if (noticeTimeoutRef.current) {
            clearTimeout(noticeTimeoutRef.current);
          }
          noticeTimeoutRef.current = setTimeout(() => {
            setNotice(null);
            noticeTimeoutRef.current = null;
          }, 3000);
          break;

        default:
          console.warn("[useDialogSession] unknown status", msg);
      }
    },
    [startListenLoop]
  );

  const startDialog = useCallback(async () => {
    if (wsRef.current) {
      return;
    }

    setError(null);
    setMessages([]);
    // Yangi suhbat — rang tayinlovi ham nol'dan boshlanishi SHART, aks
    // holda oldingi sessiyadagi spiker ID'lar bilan "band" bo'lgan order
    // qiymatlari yangi sessiyada ham saqlanib qolib, birinchi spiker
    // palitra boshidan EMAS, o'rtasidan rang olishi mumkin edi.
    setSpeakerColorOrder({});
    nextSpeakerColorOrderRef.current = 0;
    stoppedRef.current = false;

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          // DIALOG_PLAN.md 3-bosqich (C1) ASL NIYATI: noiseSuppression/
          // autoGainControl ikkisi ham o'CHIRILGAN edi — bular brauzer-ichida,
          // bizning nazoratimizdan tashqarida signalni o'zgartirib, ovoz
          // embeddinglarini (A1) ishonchsizlashtirishi mumkin edi.
          //
          // REGRESSIYA TUZATISHI (real testda topilgan: oddiy ovozda gap
          // umuman backendga yetib bormay qoldi, faqat juda baland/qichqirib
          // gapirganda ishladi — debug_audio/ dagi yozuvlarga qarang, AGC
          // o'chirilgandan keyingi sessiyada muvaffaqiyatli turnlar orasida
          // bir necha daqiqalik bo'shliqlar paydo bo'ldi). Sabab: autoGainControl
          // xom mikrofon signalini odatiy gapirish balandligida ishlatish
          // mumkin bo'lgan darajaga olib keladi — buni o'chirib, C2'dagi
          // moslashuvchan threshold (noiseFloor*3)ga ishonib qoldirish amalda
          // YETARLI emas ekan: noiseSuppression ham o'chirilgani uchun fon
          // shovqini filtrlanmay qoladi, signal esa AGC'siz past bo'lib qoladi
          // — natijada nutq/shovqin NISBATI yomonlashadi, threshold nazariy
          // jihatdan "moslashuvchan" bo'lsa ham amalda haddan tashqari baland
          // chiqib qoladi. Shu sababli autoGainControl QAYTARILDI (true) — bu
          // signal darajasini VAD ishlashi uchun yetarli qiladi.
          // noiseSuppression ATAYLAB false qolgan — bu spektral/dinamika
          // jihatidan ko'proq "qayta shakllantiradigan" ishlov, AGC kabi
          // shunchaki daraja normallashtirish emas, shuning uchun A1 xavfi
          // asosan shu bilan bog'liq edi.
          //
          // echoCancellation true qoldi — karnaydan chiqadigan tovushni
          // mikrofon qaytadan "eshitib" yubormasligi uchun kerak (C5'dagi
          // VAD-pauza bilan BIR XIL maqsadga, lekin boshqa qatlamda —
          // apparat/brauzer darajasida — xizmat qiladi).
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
    } catch (err) {
      console.error("[useDialogSession] mic access denied", err);
      setError("Mikrofonga ruxsat berilmadi");
      setPhase("error");
      return;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (event) => handleServerMessage(event.data);

      ws.onerror = (event) => {
        console.error("[useDialogSession] websocket error", event);
        setError("Ulanishda xatolik yuz berdi");
        setPhase("error");
      };

      ws.onclose = (event) => {
        if (!stoppedRef.current) {
          if (!event.wasClean) {
            console.error(
              "[useDialogSession] websocket closed unexpectedly",
              event.code,
              event.reason
            );
            setError("Ulanish kutilmaganda uzildi.");
            setPhase("error");
          } else {
            setPhase("ended");
          }
        }
        stoppedRef.current = true;
        stopListenLoop();
        cleanupMedia();
      };
    } catch (err) {
      console.error("[useDialogSession] failed to open websocket", err);
      setError(err?.message || String(err));
      setPhase("error");
    }
  }, [cleanupMedia, handleServerMessage, stopListenLoop]);

  // Har bir bubble'dagi "spikerni almashtirish" menyusidan bitta variant
  // tanlanganda shu funksiya chaqiriladi — `targetSpeaker` mavjud spiker
  // ID'si (butun son) yoki literal "new" satri ("Yangi spiker" tanlansa).
  // Backend shu `seq` uchun ovoz profilini yangilaydi (dialog.py,
  // `correct_speaker` action) va natija XUDDI SHU seq bilan yangi "turn"
  // sifatida qaytadi (handleServerMessage'dagi upsert shuni almashtiradi).
  // Tarjima QAYTA QILINMAYDI — yo'nalish spikerga emas, aniqlangan tilga
  // bog'liq bo'lgani uchun o'zgarmaydi.
  const correctSpeaker = useCallback((seq, targetSpeaker) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({ action: "correct_speaker", seq, target_speaker: targetSpeaker })
        );
      }
    } catch (err) {
      console.error("[useDialogSession] error sending correct_speaker", err);
    }
  }, []);

  // Har bir bubble'dagi 🗑 tugmasi — backend shu seq'ni turn_history'dan
  // o'chiradi (va agar shu turn yagona namuna sifatida biror spikerni
  // "ochgan" bo'lsa, o'sha spiker profilini ham) — natija "turn_deleted"
  // statusi bilan tasdiqlanadi (handleServerMessage shuni ko'rib bubble'ni
  // ekrandan olib tashlaydi).
  const deleteTurn = useCallback((seq) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "delete_turn", seq }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending delete_turn", err);
    }
  }, []);

  // Spiker ismlarini tahrirlash (✏️) — ism o'zi FAQAT frontend holatida
  // (DialogSessionView.jsx'dagi speakerNames xaritasi) saqlanadi, backend
  // mantig'iga ta'sir qilmaydi. Shunga qaramay backend'ga xabar beriladi —
  // FAQAT [/ws/dialog] loglarida spiker ID'sini inson ismiga bog'lab
  // qo'yish uchun (diagnostika).
  const renameSpeaker = useCallback((speaker, name) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "rename_speaker", speaker, name }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending rename_speaker", err);
    }
  }, []);

  // Foydalanuvchi ikkita spikerga BIR XIL ism bergandan va "bitta odam
  // sifatida birlashtiraymi?" so'roviga "Ha" deb javob berganidan keyin
  // chaqiriladi (DialogSessionView.jsx) — backend shu ikki spikerning
  // ovoz profillarini birlashtiradi (dialog.py, `merge_speakers` action)
  // va natijada "speakers_merged" statusi qaytadi (handleServerMessage
  // shuni ko'rib bubble'larni ko'chiradi).
  const mergeSpeakers = useCallback((from, into) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "merge_speakers", from, into }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending merge_speakers", err);
    }
  }, []);

  const stopDialog = useCallback(() => {
    stoppedRef.current = true;

    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "stop_dialog" }));
      }
    } catch (err) {
      console.error("[useDialogSession] error sending stop_dialog", err);
    }

    closeSocket();
    stopListenLoop();
    cleanupMedia();
    setPhase("ended");
    setAudioLevel(0);
  }, [closeSocket, cleanupMedia, stopListenLoop]);

  // Xavfsizlik to'ri: komponent kutilmaganda unmount bo'lsa ham (masalan
  // foydalanuvchi "to'xtatish"ni bosmasdan orqaga qaytsa), mikrofon va socket
  // doim tozalanadi — sizib chiqishning oldi olinadi.
  useEffect(() => {
    return () => {
      stoppedRef.current = true;
      stopListenLoop();
      cleanupMedia();
      closeSocket();
      if (noticeTimeoutRef.current) {
        clearTimeout(noticeTimeoutRef.current);
        noticeTimeoutRef.current = null;
      }
    };
  }, [cleanupMedia, closeSocket, stopListenLoop]);

  return {
    phase,
    messages,
    speakerEvents,
    speakerColorOrder,
    error,
    notice,
    audioLevel,
    isProcessing,
    startDialog,
    stopDialog,
    correctSpeaker,
    deleteTurn,
    renameSpeaker,
    mergeSpeakers,
    // C5: DialogSessionView.jsx TTS pleybek boshlanishi/tugashiga ulaydi.
    pauseMic,
    resumeMic,
  };
}

// ===========================================================================
// ESKI VA YANGI ZANJIRNI SOLISHTIRISH (C1) — to'liq eval shart emas, lekin
// qanday solishtirish mumkinligi:
//
// 1. Joriy (yangi, limiter+xom) kod bilan backend/debug_audio/ ga 1 suhbat
//    yozib oling (SAVE_DEBUG_AUDIO=True), keyin backend/tools/eval_dialog.py
//    orqali labels.json yaratib, qo'lda to'g'rilang (oldingi ishlarga
//    qarang — DIALOG_PLAN.md 0-bosqich).
// 2. `git stash` / vaqtincha shu faylni eski holatiga qaytarib (compressor
//    + noiseSuppression/AGC true), XUDDI SHU ikki kishi, XUDDI SHU
//    jumlalarni boshqa suhbatda qayta yozib oling (alohida papkaga, masalan
//    debug_audio_old_chain).
// 3. Ikkala papka uchun eval_dialog.py'ni ishga tushiring va "spiker
//    aniqligi" / "tashlab yuborilgan turnlar" / WER ko'rsatkichlarini
//    solishtiring:
//        python backend/tools/eval_dialog.py backend/debug_audio_new_chain
//        python backend/tools/eval_dialog.py backend/debug_audio_old_chain
//    Kutilgan farq: yangi zanjirda spiker aniqligi yuqoriroq (ovoz
//    embeddinglari kompressiya tufayli bir-biriga yaqinlashmagani uchun),
//    WER taxminan bir xil yoki yaxshiroq bo'lishi kerak.
//
// MUQOBIL VAD KUTUBXONASI (C2, hozircha ISHLATILMAGAN):
// @ricky0123/vad-web — Silero VAD (ONNX, brauzerda WASM orqali ishlaydi)
// neyron-tarmoq asosidagi nutq aniqlashni taqdim etadi, RMS-chegaradan
// (hatto moslashuvchan bo'lsa ham) ko'proq shovqin turlariga (masalan
// doimiy fon musiqasi) chidamli bo'lishi mumkin. Integratsiya uchun
// taxminan: `npm install @ricky0123/vad-web`, keyin
// `MicVAD.new({onSpeechStart, onSpeechEnd, onFrameProcessed})` orqali
// hozirgi `handleWorkletChunk`'dagi RMS-asosidagi `speaking` qarorini
// almashtirish mumkin (PCM capture/pre-roll/C3/C4/C5 mantiqi o'zgarishsiz
// qolaveradi — faqat "speaking" signalining MANBAI o'zgaradi). Bu ishning
// ko'lamida ISHLATILMAGAN, chunki C2'ning oddiy noise_floor×3 yechimi
// birinchi navbatda so'ralgan va qo'shimcha kutubxona/model yuklanishi
// talab qilmaydi.
// ===========================================================================
