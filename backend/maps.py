# backend/maps.py
import requests, os

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_nearby_vets(lat: float, lon: float, radius: int = 5000) -> list:
    """Find vets and animal shelters within radius meters"""
    results = []

    for place_type, keyword in [
        ("veterinary_care", "veterinary hospital"),
        ("local_government_office", "animal welfare"),
    ]:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius":   radius,
            "keyword":  keyword,
            "key":      GOOGLE_API_KEY
        }
        r = requests.get(url, params=params, timeout=5)
        places = r.json().get("results", [])

        for p in places[:3]:
            results.append({
                "name":    p.get("name"),
                "address": p.get("vicinity"),
                "rating":  p.get("rating", "N/A"),
                "open":    p.get("opening_hours", {}).get("open_now", None),
                "lat":     p["geometry"]["location"]["lat"],
                "lon":     p["geometry"]["location"]["lng"],
                "type":    place_type
            })

    return results