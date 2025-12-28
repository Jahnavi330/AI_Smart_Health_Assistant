from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import numpy as np
import tensorflow as tf
from fuzzywuzzy import process
from rules import rule_based_prediction

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "ml", "labels.json")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "..", "ml", "symptoms.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

with open(SYMPTOMS_PATH, "rb") as f:
    symptom_cols = pickle.load(f)

def text_to_vector(symptoms_text):
    """
    Convert user text → binary symptom vector
    """
    symptoms_text = symptoms_text.lower()
    input_symptoms = [s.strip() for s in symptoms_text.split(",")]

    vector = np.zeros(len(symptom_cols))

    for s in input_symptoms:
        if s in symptom_cols:
            idx = symptom_cols.index(s)
            vector[idx] = 1

    return vector.reshape(1, -1)

def ml_predict(symptoms):
    X = text_to_vector(symptoms)
    preds = model.predict(X, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx])
    return {
        "disease": labels[idx],
        "confidence": round(confidence * 100, 2)
    }

@app.route("/")
def home():
    return jsonify({"message": "Health AI Backend Running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data["symptoms"]

    rule_result = rule_based_prediction(symptoms)
    if rule_result:
        rule_result["source"] = "rule-based"
        return jsonify(rule_result)

    result = ml_predict(symptoms)
    result["source"] = "ml-model"
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
