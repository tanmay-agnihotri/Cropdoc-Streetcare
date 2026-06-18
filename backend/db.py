# backend/db.py
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

supabase = create_client(
    os.getenv("https://klvxpkinwcyvdqqfqzvi.supabase.co"),
    os.getenv("sb_publishable_qrI-wKc_FV42WVIoQdQCSg_RWjyoDbp")
)

def save_crop_report(condition, confidence, severity, lat, lon, city):
    return supabase.table("reports").insert({
        "type":       "crop",
        "condition":  condition,
        "confidence": confidence,
        "severity":   severity,
        "lat":        lat,
        "lon":        lon,
        "city":       city
    }).execute()

def save_animal_sighting(species, condition_id, urgency, lat, lon, description):
    return supabase.table("animal_sightings").insert({
        "species":      species,
        "condition_id": condition_id,
        "urgency":      urgency,
        "lat":          lat,
        "lon":          lon,
        "description":  description
    }).execute()

def get_nearby_sightings(lat, lon, radius_deg=0.1):
    # Simple bounding box query
    return supabase.table("animal_sightings")\
        .select("*")\
        .gte("lat", lat - radius_deg)\
        .lte("lat", lat + radius_deg)\
        .gte("lon", lon - radius_deg)\
        .lte("lon", lon + radius_deg)\
        .eq("is_resolved", False)\
        .order("created_at", desc=True)\
        .limit(20)\
        .execute()