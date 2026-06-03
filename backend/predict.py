import json
import pickle
import re
import numpy as np
import tensorflow as tf
from fuzzywuzzy import fuzz, process

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Load symptoms
with open("symptoms.pkl", "rb") as f:
    symptom_cols = pickle.load(f)
    symptom_cols_lower = [s.lower() for s in symptom_cols]
    symptom_lookup = {symptom: idx for idx, symptom in enumerate(symptom_cols_lower)}


def normalize_symptom_token(token):
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9 ]+", "", token)
    return token


def text_to_vector(text):
    symptoms = re.split(r"[\n;,]+|\band\b", text.lower())
    vector = np.zeros(len(symptom_cols))

    for s in symptoms:
        clean_token = normalize_symptom_token(s)
        if not clean_token:
            continue

        idx = symptom_lookup.get(clean_token)
        if idx is None:
            result = process.extractOne(clean_token, symptom_cols_lower, scorer=fuzz.token_set_ratio)
            if result:
                match, score = result
                if score > 70:
                    idx = symptom_lookup[match]
        if idx is not None:
            vector[idx] = 1

    if vector.sum() == 0:
        fallback = normalize_symptom_token(text)
        result = process.extractOne(fallback, symptom_cols_lower, scorer=fuzz.token_set_ratio)
        if result:
            match, score = result
            if score > 70:
                vector[symptom_lookup[match]] = 1

    return vector.reshape(1, -1)


def predict_disease(symptoms_text):
    try:
        X = text_to_vector(symptoms_text)
        preds = model.predict(X, verbose=0)[0]

        # FIX 1: Explicitly force the raw index into a standard Python integer
        idx = int(np.argmax(preds))
        
        # FIX 2: Safeguard tensor floating metrics explicitly to prevent serialization crashes
        confidence_val = float(preds[idx])

        return {
            "disease": str(labels[idx]),
            "confidence": round(confidence_val * 100, 2)
        }
    except Exception as e:
        print(f"INTERNAL ML ROUTE CRASH LOG: {str(e)}")
        # Safe fallback so your backend doesn't crash the whole server
        return {
            "disease": "Unable to process symptoms text layout format",
            "confidence": 0.0
        }
