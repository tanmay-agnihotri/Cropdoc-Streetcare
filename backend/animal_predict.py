import numpy as np
from PIL import Image
import io, json, os

with open(os.path.join(os.path.dirname(__file__), "class_names_animal.json")) as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

try:
    from ai_edge_litert.interpreter import Interpreter
    INTERPRETER = Interpreter(
        model_path=os.path.join(os.path.dirname(__file__), "animal_model.tflite")
    )
    print("Animal model: using LiteRT")
except:
    import tensorflow as tf
    INTERPRETER = tf.lite.Interpreter(
        model_path=os.path.join(os.path.dirname(__file__), "animal_model.tflite")
    )
    print("Animal model: using tf.lite")

INTERPRETER.allocate_tensors()
INPUT_DETAILS  = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()

print("Animal model loaded. Classes:", NUM_CLASSES)

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_animal(image_bytes: bytes, lang: str = "en") -> dict:
    from animal_db import ANIMAL_CONDITIONS

    input_data = preprocess(image_bytes)

    if INPUT_DETAILS[0]['dtype'] == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)

    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    INTERPRETER.invoke()

    raw = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

    if OUTPUT_DETAILS[0]['dtype'] == np.uint8:
        scale, zero_point = OUTPUT_DETAILS[0]['quantization']
        output = (raw.astype(np.float32) - zero_point) * scale
    else:
        output = raw.astype(np.float32)

    top_idx    = int(np.argmax(output))
    confidence = float(output[top_idx])
    condition_id = CLASS_NAMES[top_idx]

    info      = ANIMAL_CONDITIONS.get(condition_id, {})
    lang_info = info.get(lang, info.get("en", {}))

    urgency_display = {
        "none":     {"color": "green",  "label": "No action needed"},
        "low":      {"color": "yellow", "label": "See vet within 48 hours"},
        "medium":   {"color": "orange", "label": "See vet within 24 hours"},
        "high":     {"color": "red",    "label": "See vet TODAY"},
        "critical": {"color": "purple", "label": "EMERGENCY — act now"},
    }

    urgency = info.get("urgency", "low")

    return {
        "condition_id":     condition_id,
        "confidence":       round(confidence * 100, 1),
        "species":          info.get("species", "unknown"),
        "urgency":          urgency,
        "urgency_display":  urgency_display.get(urgency, {}),
        "severity":         info.get("severity", "unknown"),
        "condition":        lang_info.get("condition", condition_id),
        "what_you_see":     lang_info.get("what_you_see", ""),
        "immediate_action": lang_info.get("immediate_action", ""),
        "first_aid":        lang_info.get("first_aid", ""),
        "feeding":          lang_info.get("feeding", ""),
        "what_not_to_do":   lang_info.get("what_not_to_do", ""),
        "vet_needed":       info.get("vet_needed", False),
        "vet_urgency":      info.get("vet_urgency", ""),
        "shelter_needed":   info.get("shelter_needed", False),
        "treatment_info":   info.get("en", {}).get("treatment_info", "")
    }