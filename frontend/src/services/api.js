import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export async function translateText(text, targetLang, sourceLang = "en") {
  const response = await axios.post(`${API_BASE_URL}/translate`, {
    text,
    target_lang: targetLang,
    source_lang: sourceLang,
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