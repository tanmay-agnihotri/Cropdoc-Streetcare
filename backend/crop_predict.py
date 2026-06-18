import numpy as np
from PIL import Image
import io, json, os

# Load class names
with open(os.path.join(os.path.dirname(__file__), "class_names_crop.json")) as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

# Load TFLite model
try:
    from ai_edge_litert.interpreter import Interpreter
    INTERPRETER = Interpreter(
        model_path=os.path.join(os.path.dirname(__file__), "crop_model.tflite")
    )
    print("Crop model: using LiteRT")
except:
    import tensorflow as tf
    INTERPRETER = tf.lite.Interpreter(
        model_path=os.path.join(os.path.dirname(__file__), "crop_model.tflite")
    )
    print("Crop model: using tf.lite")

INTERPRETER.allocate_tensors()
INPUT_DETAILS  = INTERPRETER.get_input_details()
OUTPUT_DETAILS = INTERPRETER.get_output_details()

print("Crop model loaded. Classes:", NUM_CLASSES)
print("Input dtype:", INPUT_DETAILS[0]['dtype'])

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_crop(image_bytes: bytes, lang: str = "en") -> dict:
    from crop_db import CROP_CONDITIONS, DEFAULT_CONDITION

    input_data = preprocess(image_bytes)

    # Handle both float32 and uint8 models
    if INPUT_DETAILS[0]['dtype'] == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)

    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    INTERPRETER.invoke()

    raw = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

    # Dequantize if uint8 output
    if OUTPUT_DETAILS[0]['dtype'] == np.uint8:
        scale, zero_point = OUTPUT_DETAILS[0]['quantization']
        output = (raw.astype(np.float32) - zero_point) * scale
    else:
        output = raw.astype(np.float32)

    top3_idx  = np.argsort(output)[::-1][:3]
    top3_conf = output[top3_idx]

    disease_id = CLASS_NAMES[top3_idx[0]]
    confidence = float(top3_conf[0])

    info      = CROP_CONDITIONS.get(disease_id, DEFAULT_CONDITION)
    lang_info = info.get(lang, info.get("en", {}))

    return {
        "disease_id":  disease_id,
        "confidence":  round(confidence * 100, 1),
        "severity":    info.get("severity", "unknown"),
        "is_healthy":  "healthy" in disease_id.lower(),
        "condition":   lang_info.get("name", disease_id),
        "symptoms":    lang_info.get("symptoms", ""),
        "treatment":   lang_info.get("treatment", ""),
        "prevention":  lang_info.get("prevention", ""),
        "top3": [
            {
                "disease":    CLASS_NAMES[i],
                "confidence": round(float(c) * 100, 1)
            }
            for i, c in zip(top3_idx, top3_conf)
        ]
    }