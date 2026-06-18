#import requests, os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

DISEASE_TRIGGERS = {
    "Tomato___Early_blight":    {"temp_min":24, "temp_max":29, "humidity":85},
    "Tomato___Late_blight":     {"temp_min":10, "temp_max":24, "humidity":90},
    "Potato___Late_blight":     {"temp_min":10, "temp_max":20, "humidity":90},
    "Corn_(maize)___Common_rust_": {"temp_min":16, "temp_max":25, "humidity":80},
}

def get_disease_risk(lat: float, lon: float, disease_id: str) -> dict:
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat":lat,"lon":lon,"appid":API_KEY,"units":"metric"},
            timeout=5
        )
        data     = r.json()
        temp     = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        city     = data["name"]

        triggers   = DISEASE_TRIGGERS.get(disease_id, {})
        risk_level = "low"

        if triggers:
            t_ok = triggers["temp_min"] <= temp <= triggers["temp_max"]
            h_ok = humidity >= triggers["humidity"]
            if t_ok and h_ok:
                risk_level = "high"
            elif t_ok or h_ok:
                risk_level = "medium"

        messages = {
            "high":   f"High outbreak risk in {city}! Spray preventive fungicide NOW.",
            "medium": f"Moderate risk in {city}. Monitor crops daily.",
            "low":    f"Low outbreak risk in {city} currently."
        }

        return {
            "city":        city,
            "temperature": temp,
            "humidity":    humidity,
            "risk_level":  risk_level,
            "message":     messages.get(risk_level, "")
        }
    except Exception as e:
        return {"error": str(e), "risk_level": "unknown"}