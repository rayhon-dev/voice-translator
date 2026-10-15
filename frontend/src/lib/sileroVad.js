// Mentor demo'da topilgan 1-muammo tuzatishi: gap boshlanishi/tugashi
// endi RMS energiya chegarasi o'rniga Silero (neyron-tarmoq) VAD orqali
// aniqlanadi — shovqinli xonada ham oddiy ovozdagi nutqni tutish uchun.
//
// @ricky0123/vad-web'ning yuqori-darajali API'si (`MicVAD`) O'ZI alohida
// `getUserMedia()` chaqirib, o'ZINING AudioContext/AudioWorklet zanjirini
// yaratadi — bu useDialogSession.js'dagi MAVJUD PCM oqimi (pre-roll, C1
// limiter, 16kHz AudioWorkletNode) bilan ikkita alohida mikrofon ulanishi
// degani bo'lardi. Shu sababli bu yerda past-darajali `Silero` model
// klassi TO'G'RIDAN-TO'G'RI ishlatiladi — faqat ONNX inference, hech
// qanday o'z audio oqimi/mikrofon ochilmaydi; freymlar useDialogSession.js
// tomonidan MAVJUD worklet oqimidan push qilinadi (qarang pushSamples()).
import * as ort from "onnxruntime-web/wasm";
import { defaultModelFetcher } from "@ricky0123/vad-web";
// MUHIM: `Silero` klassi @ricky0123/vad-web'ning TEPA DARAJADAGI
// ("@ricky0123/vad-web") export ro'yxatida YO'Q (paketning dist/index.js/
// index.d.ts fayliga qarang — u faqat MicVAD/FrameProcessor/
// defaultModelFetcher va boshqalarni eksport qiladi) — FAQAT ichki
// `dist/models/` submoduli orqali ochiq. Avval tepa darajadan import
// qilingani uchun `Silero` HAR DOIM `undefined` bo'lib, keyinchalik
// `Silero.new(...)` chaqiruvi SINXRON TypeError tashlar edi — bu esa
// (Promise ichida EMAS, to'g'ridan-to'g'ri getSileroModel() tanasida
// sodir bo'lgani uchun) hech qanday `.catch()` ushlay olmaydi va
// useDialogSession.js'dagi mount useEffect'ini xato bilan to'xtatib,
// ErrorBoundary'siz butun React daraxtini "oq sahifa"ga aylantirardi —
// aynan mentor demo'da ko'rilgan nosozlik. Shu sababli bu yerda to'g'ri,
// ICHKI yo'ldan import qilinadi.
import { Silero } from "@ricky0123/vad-web/dist/models/silero";

// ONNX model fayli (~2.3MB) va onnxruntime-web'ning WASM ikkilik fayli
// (~14MB) ikkisi HAM `node_modules`'dagi paketlardan (@ricky0123/vad-web,
// onnxruntime-web) to'g'ridan-to'g'ri, `new URL(..., import.meta.url)`
// orqali olinadi — qo'lda public/'ga ko'chirish SHART EMAS (loyihada
// *.onnx fayllar .gitignore'da — qarang .gitignore:59 — model binary'larini
// git'ga COMMIT QILMASLIK repo konvensiyasi, backend'dagi boshqa modellar
// ham xuddi shunday HuggingFace/node_modules'dan runtime'da olinadi).
// Vite bu patternni (xuddi onnxruntime-web'ning o'z WASM fayli uchun
// ichki qiladigani kabi) build vaqtida avtomatik aniqlab, hashlangan
// statik asset sifatida dist/'ga qo'shadi; dev serverda esa to'g'ridan-
// to'g'ri node_modules'dan beriladi.
const MODEL_URL = new URL(
  "../../node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx",
  import.meta.url
).href;

// Silero v5/v6 modeli 16kHz'da 512-namunalik (~32ms) freymlar bilan
// ishlaydi (qarang @ricky0123/vad-web/dist/real-time-vad.js:
// `frameSamples = model === "legacy" ? 1536 : 512`). Bu yerda qat'iy
// konstanta sifatida takrorlanadi — kutubxonaning ichki implementatsiya
// detaliga qarab o'zgarishi mumkin bo'lgani uchun pastda komment bilan
// belgilab qo'yilgan.
export const SILERO_FRAME_SAMPLES = 512;

let configured = false;
function configureOrt() {
  if (configured) return;
  configured = true;
  ort.env.logLevel = "error";
}

let modelPromise = null;

/**
 * Silero VAD modelini (bitta marta, butun sahifa umri davomida) yuklaydi
 * va keshlaydi — bir nechta suhbat sessiyasi (startDialog/stopDialog)
 * bitta yuklangan modeldan qayta foydalanadi. Muvaffaqiyatsiz bo'lsa xato
 * uloqtiriladi — chaqiruvchi (useDialogSession.js) buni ushlab, energiya
 * VAD'ga qaytishi SHART.
 */
export function getSileroModel() {
  if (!modelPromise) {
    // Butun tana `async` IIFE ichida — shu bilan ICHKARIDA (shu jumladan
    // `configureOrt()`/`Silero.new(...)` chaqiruvining o'ZIDA, ya'ni HALI
    // birorta `await`ga yetmasdan) sodir bo'lgan HAR QANDAY SINXRON throw
    // ham avtomatik rad etilgan promise'ga aylanadi — chaqiruvchi
    // (useDialogSession.js) doim `.catch()` bilan ushlay oladi deb ISHONCH
    // bilan tayanishi mumkin, hech qachon xato useEffect'dan "sizib
    // chiqib", ErrorBoundary'siz butun ilovani qotirmaydi.
    modelPromise = (async () => {
      configureOrt();
      return Silero.new(ort, () => defaultModelFetcher(MODEL_URL));
    })().catch((err) => {
      // Keyingi chaqiruv qayta urinib ko'rsin (masalan vaqtinchalik tarmoq
      // xatosi) — doimiy muvaffaqiyatsiz promise'ni keshlab qo'ymaymiz.
      modelPromise = null;
      throw err;
    });
  }
  return modelPromise;
}
