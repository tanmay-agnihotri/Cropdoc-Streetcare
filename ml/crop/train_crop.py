# backend/crop_predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json, os
from crop_db import CROP_CONDITIONS

# Load TFLite model once at startup
INTERPRETER = tf.lite.Interpreter(model_path="crop_model.tflite")
INTERPRETER.allocate_tensors()
INPUT_DETAILS  = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()

with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

def predict_crop(image_bytes: bytes, lang: str = "en") -> dict:
    input_data = preprocess_image(image_bytes)

    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    INTERPRETER.invoke()

    output = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
    # Dequantize output
    scale, zero_point = OUTPUT_DETAILS[0]['quantization']
    output = (output.astype(np.float32) - zero_point) * scale

    top3_idx  = np.argsort(output)[::-1][:3]
    top3_conf = output[top3_idx]

    disease_id = CLASS_NAMES[top3_idx[0]]
    confidence = float(top3_conf[0])

    info = CROP_CONDITIONS.get(disease_id, {})
    lang_info = info.get(lang, info.get("en", {}))

    return {
        "disease_id":   disease_id,
        "confidence":   round(confidence * 100, 1),
        "severity":     info.get("severity", "unknown"),
        "is_healthy":   "healthy" in disease_id.lower(),
        "condition":    lang_info.get("name", disease_id),
        "symptoms":     lang_info.get("symptoms", ""),
        "treatment":    lang_info.get("treatment", ""),
        "prevention":   lang_info.get("prevention", ""),
        "top3": [
            {"disease": CLASS_NAMES[i], "confidence": round(float(c)*100, 1)}
            for i, c in zip(top3_idx, top3_conf)
        ]
    }