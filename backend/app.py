from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import numpy as np
import tensorflow as tf
from fuzzywuzzy import process
from rules import rule_based_prediction
from predict import predict_disease
import google.generativeai as genai

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True) 
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_LOCAL_FALLBACK_KEY"))

# Strict System Prompt Rule Definition to maintain clinical guardrails
SYSTEM_INSTRUCTION = (
    "You are an expert AI medical advisor widget integrated into a disease prediction site. "
    "Analyze user questions regarding symptoms structurally. Offer clear, markdown-formatted information "
    "including bullet points on helpful lifestyle modifications, tracking factors, and home precautions. "
    "CRITICAL MANDATE: Provide a prominent bold disclaimer statement noting that your advice is strictly "
    "educational, does not represent standard definitive medical diagnoses, and requires professional consulting if severe."
)

chatbot_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

@app.route('/chat', methods=['POST'])
def chatbot_endpoint():
    try:
        
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"reply": "I did not receive any message context. Please try typing again."}), 400
            
        # Call the Google Gemini API directly
        response = chatbot_model.generate_content(user_message)
        return jsonify({"reply": response.text})
        
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        return jsonify({"reply": "I encountered a system routing error. Please consult directly with local clinic resources if symptoms are severe."}), 500
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"
SYMPTOMS_PATH = "symptoms.pkl"

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

    result = predict_disease(symptoms_text=symptoms)
    result["source"] = "ml-model"
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
