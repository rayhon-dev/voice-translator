// DIALOG_PLAN.md 3-bosqich (C3): MediaRecorder/WebM o'rniga — pre-roll va
// aniq vaqt-chegarali "turn" kesish uchun XOM PCM namunalarga to'g'ridan-
// to'g'ri kirish kerak, bu esa faqat AudioWorklet orqali mumkin (asosiy
// JS thread'da emas, alohida audio-render thread'ida ishlaydi).
//
// Har bir process() chaqiruvi 128 namunalik blok beradi (Web Audio
// standarti) — har safar main thread'ga xabar yubormasdan (bu 16kHz'da
// soniyada ~125 marta bo'lardi), PCM_CHUNK_SIZE namunaga yetguncha ICHKI
// bufer'da yig'ib, keyin BITTA postMessage bilan yuboradi (~128ms'da bir
// marta, eski VAD_CHECK_INTERVAL_MS=100ga yaqin).
const PCM_CHUNK_SIZE = 2048;

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(PCM_CHUNK_SIZE);
    this._writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        this._buffer[this._writeIndex++] = channel[i];
        if (this._writeIndex >= PCM_CHUNK_SIZE) {
          // .slice() — nusxa: shu AudioWorkletProcessor o'ziga xos
          // buferni qayta ishlatishni davom ettiradi, main thread esa
          // mustaqil nusxa oladi (aks holda ma'lumot ustidan "poyga"
          // bo'lardi).
          this.port.postMessage(this._buffer.slice());
          this._writeIndex = 0;
        }
      }
    }
    // `true` — processor faol qolishi kerakligini bildiradi (node
    // uzilib qolmasligi uchun).
    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
