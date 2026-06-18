import requests, os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

def get_nearby_vets(lat: float, lon: float, radius: int = 5000) -> list:
    if not GOOGLE_KEY:
        return []
    results = []
    for keyword in ["veterinary hospital", "animal shelter", "animal welfare"]:
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params={
                    "location": f"{lat},{lon}",
                    "radius":   radius,
                    "keyword":  keyword,
                    "key":      GOOGLE_KEY
                },
                timeout=5
            )
            for p in r.json().get("results", [])[:2]:
                results.append({
                    "name":    p.get("name"),
                    "address": p.get("vicinity"),
                    "rating":  p.get("rating", "N/A"),
                    "open":    p.get("opening_hours", {}).get("open_now"),
                    "lat":     p["geometry"]["location"]["lat"],
                    "lon":     p["geometry"]["location"]["lng"]
                })
        except:
            pass
    return results