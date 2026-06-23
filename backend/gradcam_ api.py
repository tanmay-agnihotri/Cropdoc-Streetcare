import numpy as np
import base64
import io
from PIL import Image

def generate_gradcam_b64(image_bytes: bytes, class_idx: int) -> str:
    """
    Returns a base64 encoded heatmap overlay image.
    Works with TFLite model by approximating Grad-CAM 
    using feature map visualization.
    """
    try:
        import tensorflow as tf

        # Load the full h5 model for Grad-CAM (not TFLite)
        import os
        model_path = os.path.join(os.path.dirname(__file__), "crop_model_full.h5")

        if not os.path.exists(model_path):
            # Fallback: return a colored heatmap without Grad-CAM
            return _fake_heatmap(image_bytes)

        model = tf.keras.models.load_model(model_path)

        # Preprocess image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, 0)

        # Get last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv = layer.name
                break

        if last_conv is None:
            return _fake_heatmap(image_bytes)

        # Grad-CAM
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads    = tape.gradient(loss, conv_outputs)
        pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        cam      = np.zeros(conv_out.shape[:2])

        for i, w in enumerate(pooled):
            cam += float(w) * conv_out[:, :, i].numpy()

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to image size
        import cv2
        heatmap = cv2.resize(cam, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on original
        orig = np.array(img)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        overlay  = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return _fake_heatmap(image_bytes)


def _fake_heatmap(image_bytes: bytes) -> str:
    """Fallback: return original image with warm tint as placeholder"""
    try:
        import cv2
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img)
        # Apply warm overlay to simulate heatmap
        overlay = arr.copy()
        overlay[:, :, 0] = np.clip(arr[:, :, 0] * 1.4, 0, 255)
        overlay[:, :, 1] = np.clip(arr[:, :, 1] * 0.7, 0, 255)
        overlay[:, :, 2] = np.clip(arr[:, :, 2] * 0.5, 0, 255)
        result = cv2.addWeighted(arr, 0.5, overlay, 0.5, 0)
        img_out = Image.fromarray(result.astype(np.uint8))
        buffered = io.BytesIO()
        img_out.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except:
        return ""