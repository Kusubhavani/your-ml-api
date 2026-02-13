import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

MODEL = None
IMAGE_SIZE = (32, 32)
CLASS_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model(model_path: str = None):
    global MODEL
    if MODEL is None:
        path = model_path or os.getenv("MODEL_PATH", "models/my_classifier_model.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        MODEL = tf.keras.models.load_model(path)
    return MODEL

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

def predict_image(image: np.ndarray):
    model = load_model()
    preds = model.predict(image)
    idx = int(np.argmax(preds))
    return {
        "class_label": CLASS_LABELS[idx],
        "probabilities": preds[0].tolist()
    }
