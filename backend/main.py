# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from crop_predict  import predict_crop
from animal_predict import predict_animal
from sms_handler   import handle_sms
from weather        import get_disease_risk
from voice          import generate_voice
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="CropDoc + StreetCare API", version="1.0.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "CropDoc + StreetCare API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- CROP ENDPOINTS ----

@app.post("/predict/crop")
async def predict_crop_endpoint(
    file: UploadFile = File(...),
    lang: str = Form(default="en"),
    lat:  float = Form(default=None),
    lon:  float = Form(default=None)
):
    image_bytes = await file.read()
    result = predict_crop(image_bytes, lang)

    # Add weather-based outbreak risk if location provided
    if lat and lon:
        risk = get_disease_risk(lat, lon, result["disease_id"])
        result["weather_risk"] = risk

    return JSONResponse(content=result)

@app.post("/predict/animal")
async def predict_animal_endpoint(
    file: UploadFile = File(...),
    lang: str = Form(default="en"),
    lat:  float = Form(default=None),
    lon:  float = Form(default=None)
):
    image_bytes = await file.read()
    result = predict_animal(image_bytes, lang)

    # Add nearest vet/shelter if location provided
    if lat and lon:
        from maps import get_nearby_vets
        result["nearby_vets"] = get_nearby_vets(lat, lon)

    return JSONResponse(content=result)

# ---- VOICE ENDPOINT ----

@app.post("/voice")
async def voice_endpoint(text: str = Form(...), lang: str = Form(default="hi")):
    audio_path = generate_voice(text, lang)
    from fastapi.responses import FileResponse
    return FileResponse(audio_path, media_type="audio/mpeg")

# ---- SMS/WHATSAPP WEBHOOK (Twilio calls this) ----

@app.post("/webhook/sms")
async def sms_webhook(request: dict):
    return await handle_sms(request)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)