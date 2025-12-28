import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "image_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "ml", "image_labels.json")

# ---------- LOAD MODEL & LABELS ----------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# ---------- IMAGE PREPROCESS FUNCTION ----------
def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load image from path and preprocess for CNN
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# ---------- PREDICTION FUNCTION ----------
def predict_image(img_path):
    """
    Input  : path to image
    Output : dict {disease, confidence, source}
    """
    img = preprocess_image(img_path)
    preds = model.predict(img, verbose=0)
    
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease": labels[str(class_idx)],
        "confidence": round(confidence * 100, 2),
        "source": "image-model"
    }

# ---------- TEST EXAMPLE ----------
if __name__ == "__main__":
    test_image_path = "training/skin/val/Acne/123.jpg"  # replace with your image
    result = predict_image(test_image_path)
    print(result)
