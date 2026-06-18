from gtts import gTTS
import os, uuid

LANG_CODES = {
    "en": "en",
    "hi": "hi",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
    "gu": "gu"
}

def generate_voice(text: str, lang: str = "hi") -> str:
    code     = LANG_CODES.get(lang, "hi")
    filename = f"/tmp/voice_{uuid.uuid4().hex}.mp3"
    tts      = gTTS(text=text, lang=code, slow=False)
    tts.save(filename)
    return filename