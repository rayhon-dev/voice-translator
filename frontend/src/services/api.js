import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");

  const response = await axios.post(`${API_BASE_URL}/transcribe`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data.text;
}

export async function translateText(text, targetLang) {
  const response = await axios.post(`${API_BASE_URL}/translate`, {
    text,
    target_lang: targetLang,
  });

  return response.data.translation;
}

export async function speakText(text, lang) {
  const response = await axios.post(
    `${API_BASE_URL}/speak`,
    { text, lang },
    { responseType: "blob" }
  );

  return response.data;
}

export async function identifySpeaker(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");

  const response = await axios.post(`${API_BASE_URL}/identify-speaker`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data.speaker;
}

export async function resetDialog() {
  await axios.post(`${API_BASE_URL}/reset-dialog`);
}

