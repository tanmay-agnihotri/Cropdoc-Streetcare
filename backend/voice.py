from gtts import gTTS
import os, uuid, tempfile

LANG_CODES = {
    "en": "en", "hi": "hi", "mr": "mr",
    "ta": "ta", "te": "te", "bn": "bn", "gu": "gu"
}

def generate_voice(text: str, lang: str = "hi") -> str:
    code     = LANG_CODES.get(lang, "hi")
    filename = os.path.join(tempfile.gettempdir(), f"voice_{uuid.uuid4().hex}.mp3")

    # Clean text for TTS
    clean = text.replace('\n', ' ').replace('•', '').strip()
    if len(clean) > 500:
        clean = clean[:500] + "..."

    tts = gTTS(text=clean, lang=code, slow=False)
    tts.save(filename)
    return filename