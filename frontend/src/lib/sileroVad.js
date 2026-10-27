import * as ort from "onnxruntime-web/wasm";
import { defaultModelFetcher } from "@ricky0123/vad-web";
import { Silero } from "@ricky0123/vad-web/dist/models/silero";

const MODEL_URL = new URL(
  "../../node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx",
  import.meta.url
).href;

export const SILERO_FRAME_SAMPLES = 512;

let configured = false;
function configureOrt() {
  if (configured) return;
  configured = true;
  ort.env.logLevel = "error";
}

let modelPromise = null;

export function getSileroModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      configureOrt();
      return Silero.new(ort, () => defaultModelFetcher(MODEL_URL));
    })().catch((err) => {
      modelPromise = null;
      throw err;
    });
  }
  return modelPromise;
}
