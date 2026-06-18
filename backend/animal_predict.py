# backend/animal_predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json
from animal_db import ANIMAL_CONDITIONS

INTERPRETER = tf.lite.Interpreter(model_path="animal_model.tflite")
INTERPRETER.allocate_tensors()
INPUT_DETAILS  = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()

with open("animal_class_names.json") as f:
    CLASS_NAMES = json.load(f)

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)

def predict_animal(image_bytes: bytes, lang: str = "en") -> dict:
    input_data = preprocess(image_bytes)
    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    INTERPRETER.invoke()

    output = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
    scale, zero_point = OUTPUT_DETAILS[0]['quantization']
    output = (output.astype(np.float32) - zero_point) * scale

    top_idx  = int(np.argmax(output))
    confidence = float(output[top_idx])
    condition_id = CLASS_NAMES[top_idx]

    info      = ANIMAL_CONDITIONS.get(condition_id, {})
    lang_info = info.get(lang, info.get("en", {}))

    urgency_map = {
        "none":     {"color": "green",  "label": "No action needed"},
        "low":      {"color": "yellow", "label": "See vet within 48 hours"},
        "medium":   {"color": "orange", "label": "See vet within 24 hours"},
        "high":     {"color": "red",    "label": "See vet TODAY"},
        "critical": {"color": "purple", "label": "EMERGENCY — act now"},
    }
    urgency = info.get("urgency", "low")

    return {
        "condition_id":      condition_id,
        "confidence":        round(confidence * 100, 1),
        "species":           info.get("species", "unknown"),
        "urgency":           urgency,
        "urgency_display":   urgency_map.get(urgency, {}),
        "severity":          info.get("severity", "unknown"),
        "condition":         lang_info.get("condition", condition_id),
        "what_you_see":      lang_info.get("what_you_see", ""),
        "immediate_action":  lang_info.get("immediate_action", ""),
        "first_aid":         lang_info.get("first_aid", ""),
        "feeding":           lang_info.get("feeding", ""),
        "what_not_to_do":    lang_info.get("what_not_to_do", ""),
        "vet_needed":        info.get("vet_needed", False),
        "vet_urgency":       info.get("vet_urgency", ""),
        "shelter_needed":    info.get("shelter_needed", False),
        "resources_needed":  info.get("resources_needed", []),
        "treatment_info":    info.get("en", {}).get("treatment_info", "")
    }