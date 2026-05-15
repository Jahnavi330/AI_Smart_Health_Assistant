import json
import pickle
import numpy as np
import tensorflow as tf
from fuzzywuzzy import process

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Load symptoms
with open("symptoms.pkl", "rb") as f:
    symptom_cols = pickle.load(f)


def text_to_vector(text):
    symptoms = [s.strip().lower() for s in text.split(",")]
    vector = np.zeros(len(symptom_cols))

    for s in symptoms:
        if not s:  # Skip empty entries or extra commas
            continue
            
        result = process.extractOne(s, symptom_cols)
        
        # ADD THIS SAFETY CHECK: Only unpack if fuzzywuzzy found a valid match
        if result:
            match, score = result
            if score > 70:
                vector[symptom_cols.index(match)] = 1

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
