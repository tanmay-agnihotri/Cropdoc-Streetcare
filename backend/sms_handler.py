# backend/sms_handler.py
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os, base64, requests
from dotenv import load_dotenv
load_dotenv()

client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

async def handle_sms(form_data) -> str:
    """
    Twilio sends a POST with:
      Body     = text message
      MediaUrl0 = image URL (if user sent a photo)
      From     = sender's number
    """
    body      = form_data.get("Body", "").strip().lower()
    media_url = form_data.get("MediaUrl0", None)
    from_num  = form_data.get("From", "")

    resp = MessagingResponse()

    if media_url is None:
        resp.message(
            "🌿 CropDoc / StreetCare\n\n"
            "Send a photo of:\n"
            "• A crop leaf → get disease diagnosis\n"
            "• A stray animal → get health advice\n\n"
            "Reply with C for crops, A for animals."
        )
        return str(resp)

    # Download the image Twilio received
    r = requests.get(media_url, auth=(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    ))
    image_bytes = r.content

    # Determine mode from recent message body
    mode = "crop" if "c" in body else "animal"

    # Call internal prediction
    if mode == "crop":
        from crop_predict import predict_crop
        result = predict_crop(image_bytes, lang="hi")
        reply = (
            f"🌾 फसल रोग: {result['condition']}\n"
            f"📊 सटीकता: {result['confidence']}%\n"
            f"⚕️ उपचार: {result['treatment']}\n"
            f"🛡️ बचाव: {result['prevention']}"
        )
    else:
        from animal_predict import predict_animal
        result = predict_animal(image_bytes, lang="hi")
        reply = (
            f"🐾 स्थिति: {result['condition']}\n"
            f"🚨 तुरंत करें: {result['immediate_action'][:200]}\n"
            f"🍚 खाना: {result['feeding'][:150]}"
        )

    resp.message(reply)
    return str(resp)