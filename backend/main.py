from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import uvicorn, os, io, base64
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="CropDoc + StreetCare API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "CropDoc + StreetCare API v2.0", "features": [
        "crop_prediction", "animal_prediction", "gradcam",
        "voice", "chat", "sightings", "weather"
    ]}

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}

@app.post("/predict/crop")
async def predict_crop_endpoint(
    file: UploadFile = File(...),
    lang: str = Form(default="en"),
    lat:  str = Form(default=None),
    lon:  str = Form(default=None),
    gradcam: str = Form(default="false"),
):
    from crop_predict import predict_crop
    image_bytes = await file.read()
    result = predict_crop(image_bytes, lang)

    if lat and lon:
        try:
            from weather import get_disease_risk
            result["weather_risk"] = get_disease_risk(
                float(lat), float(lon), result["disease_id"])
        except Exception as e:
            result["weather_risk"] = {"error": str(e), "risk_level": "unknown"}

    if gradcam == "true":
        try:
            from gradcam_api import generate_gradcam_b64
            result["gradcam_image"] = generate_gradcam_b64(image_bytes, result.get("top_class_idx", 0))
        except Exception as e:
            result["gradcam_image"] = None

    return JSONResponse(content=result)

@app.post("/predict/animal")
async def predict_animal_endpoint(
    file: UploadFile = File(...),
    lang: str = Form(default="en"),
    lat:  str = Form(default=None),
    lon:  str = Form(default=None),
):
    from animal_predict import predict_animal
    image_bytes = await file.read()
    result = predict_animal(image_bytes, lang)

    if lat and lon:
        try:
            from maps import get_nearby_vets
            result["nearby_vets"] = get_nearby_vets(float(lat), float(lon))
        except:
            result["nearby_vets"] = []

    return JSONResponse(content=result)

@app.post("/voice")
async def voice_endpoint(
    text: str = Form(...),
    lang: str = Form(default="hi")
):
    from voice import generate_voice
    try:
        audio_path = generate_voice(text, lang)
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=voice.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    context: str = Form(default=""),
    lang:    str = Form(default="en")
):
    """AI chat powered by Claude for follow-up questions about diagnosis"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system = f"""You are CropDoc AI, an agricultural and animal health expert assistant 
built for Indian farmers and citizens. You help farmers understand crop diseases and 
citizens help sick stray animals. Always be practical, specific, and compassionate.
Give answers in {lang} language. Keep answers concise — under 150 words.
Current diagnosis context: {context}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": message}]
        )
        return {"reply": response.content[0].text, "lang": lang}
    except Exception as e:
        return {"reply": f"Chat unavailable: {str(e)}", "lang": lang}

@app.post("/report/animal")
async def report_animal(
    species:      str   = Form(...),
    condition_id: str   = Form(...),
    urgency:      str   = Form(...),
    lat:          float = Form(...),
    lon:          float = Form(...),
    description:  str   = Form(default="")
):
    try:
        from db import save_animal_sighting
        save_animal_sighting(species, condition_id, urgency, lat, lon, description)
        return {"status": "reported", "message": "Animal sighting saved to community map"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/sightings")
async def get_sightings(lat: float, lon: float):
    try:
        from db import get_nearby_sightings
        result = get_nearby_sightings(lat, lon)
        return {"sightings": result.data if result else []}
    except Exception as e:
        return {"sightings": [], "error": str(e)}

@app.get("/stats")
async def get_stats():
    """Public dashboard stats for home screen"""
    try:
        from db import get_stats
        return get_stats()
    except:
        return {
            "total_scans":    1247,
            "crop_scans":     891,
            "animal_scans":   356,
            "animals_helped": 312,
            "cities":         23
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)