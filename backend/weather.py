# backend/weather.py
import requests, os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("55332cc6333061f44620145a7f6a2d85  ")

# Weather conditions that spike disease risk
DISEASE_WEATHER_TRIGGERS = {
    "Tomato___Early_blight":  {"temp_min":24,"temp_max":29,"humidity":85},
    "Tomato___Late_blight":   {"temp_min":10,"temp_max":24,"humidity":90},
    "Potato___Late_blight":   {"temp_min":10,"temp_max":20,"humidity":90},
    "Rice___Leaf_Blast":      {"temp_min":25,"temp_max":35,"humidity":89},
    "Wheat___Stripe_Rust":    {"temp_min":10,"temp_max":15,"humidity":80},
    "Cotton___Bacterial_Blight":{"temp_min":30,"temp_max":40,"humidity":70},
}

def get_disease_risk(lat: float, lon: float, disease_id: str) -> dict:
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        temp     = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        city     = data["name"]

        triggers = DISEASE_WEATHER_TRIGGERS.get(disease_id, {})
        risk_level = "low"

        if triggers:
            temp_ok     = triggers["temp_min"] <= temp <= triggers["temp_max"]
            humidity_ok = humidity >= triggers["humidity"]
            if temp_ok and humidity_ok:
                risk_level = "high"
            elif temp_ok or humidity_ok:
                risk_level = "medium"

        return {
            "city":        city,
            "temperature": temp,
            "humidity":    humidity,
            "risk_level":  risk_level,
            "message": {
                "high":   f"⚠️ High outbreak risk in {city}! Temp {temp}°C, Humidity {humidity}% — ideal for this disease. Spray preventive fungicide NOW.",
                "medium": f"Moderate risk in {city}. Monitor your crops daily.",
                "low":    f"Low outbreak risk in {city} currently."
            }.get(risk_level, "")
        }
    except Exception as e:
        return {"error": str(e), "risk_level": "unknown"}