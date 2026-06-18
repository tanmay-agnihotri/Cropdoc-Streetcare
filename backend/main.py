from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn, os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CropDoc + StreetCare API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status": "CropDoc + StreetCare API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/crop")
async def predict_crop_endpoint(
    file: UploadFile = File(...),
    lang: str        = Form(default="en"),
    lat:  str        = Form(default=None),
    lon:  str        = Form(default=None)
):
    from crop_predict import predict_crop
    image_bytes = await file.read()
    result = predict_crop(image_bytes, lang)

    if lat and lon:
        try:
            from weather import get_disease_risk
            result["weather_risk"] = get_disease_risk(
                float(lat), float(lon), result["disease_id"]
            )
        except Exception as e:
            result["weather_risk"] = {"error": str(e)}

    return JSONResponse(content=result)

@app.post("/predict/animal")
async def predict_animal_endpoint(
    file: UploadFile = File(...),
    lang: str        = Form(default="en"),
    lat:  str        = Form(default=None),
    lon:  str        = Form(default=None)
):
    from animal_predict import predict_animal
    image_bytes = await file.read()
    result = predict_animal(image_bytes, lang)

    if lat and lon:
        try:
            from maps import get_nearby_vets
            result["nearby_vets"] = get_nearby_vets(float(lat), float(lon))
        except Exception as e:
            result["nearby_vets"] = []

    return JSONResponse(content=result)

@app.post("/voice")
async def voice_endpoint(
    text: str = Form(...),
    lang: str = Form(default="hi")
):
    from voice import generate_voice
    audio_path = generate_voice(text, lang)
    return FileResponse(audio_path, media_type="audio/mpeg")

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
        return {"status": "reported", "message": "Thank you for reporting"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/sightings")
async def get_sightings(lat: float, lon: float):
    try:
        from db import get_nearby_sightings
        result = get_nearby_sightings(lat, lon)
        return {"sightings": result.data}
    except Exception as e:
        return {"sightings": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )