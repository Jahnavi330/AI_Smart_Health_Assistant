import json
import pickle
import numpy as np
import tensorflow as tf
from fuzzywuzzy import process

# Load model
model = tf.keras.models.load_model("ml/model.h5")

# Load labels
with open("ml/labels.json", "r") as f:
    labels = json.load(f)

# Load symptoms
with open("ml/symptoms.pkl", "rb") as f:
    symptom_cols = pickle.load(f)


def text_to_vector(text):
    symptoms = [s.strip().lower() for s in text.split(",")]
    vector = np.zeros(len(symptom_cols))

    for s in symptoms:
        match, score = process.extractOne(s, symptom_cols)
        if score > 70:
            vector[symptom_cols.index(match)] = 1

    return vector.reshape(1, -1)


def predict_disease(symptoms_text):
    X = text_to_vector(symptoms_text)
    preds = model.predict(X, verbose=0)[0]

    idx = int(np.argmax(preds))
    return {
        "disease": labels[idx],
        "confidence": round(float(preds[idx]) * 100, 2)
    }
