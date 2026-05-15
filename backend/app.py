import os
import json
import pickle
import numpy as np

# Optimize TensorFlow memory boundaries to prevent Render Free Tier (512MB RAM) crashes
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Safe runtime fallback helper for rule-based check engine
try:
    from rules import rule_based_prediction
except ImportError:
    def rule_based_prediction(symptoms): return None

# Safe runtime fallback helper for external predict files
try:
    from predict import predict_disease
except ImportError:
    predict_disease = None

app = Flask(__name__)

# Enforce clean global wildcard cross-origin resource access control rules
CORS(app, resources={r"/*": {"origins": "*"}})

# Disable strict trailing slash routing engines globally to eliminate ReqBin 405 blocks
app.url_map.strict_slashes = False

# Initialize Google AI Gemini Engine configurations
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_LOCAL_FALLBACK_KEY"))

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

# --- MACHINE LEARNING ASSETS LOAD CODES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "symptoms.pkl")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    with open(SYMPTOMS_PATH, "rb") as f:
        symptom_cols = pickle.load(f)
except Exception as e:
    print(f"Warning: ML assets could not be loaded on startup: {str(e)}")
    model, labels, symptom_cols = None, [], []

def text_to_vector(symptoms_text):
    symptoms_text = symptoms_text.lower()
    input_symptoms = [s.strip() for s in symptoms_text.split(",")]
    vector = np.zeros(len(symptom_cols)) if symptom_cols else np.zeros(132)
    if symptom_cols:
        for s in input_symptoms:
            if s in symptom_cols:
                idx = symptom_cols.index(s)
                vector[idx] = 1
    return vector.reshape(1, -1)

def ml_predict(symptoms):
    if model is None or not labels:
        return {"disease": "Model Loading...", "confidence": 0.0, "info": "The network is compiling."}
    X = text_to_vector(symptoms)
    preds = model.predict(X, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx])
    return {
        "disease": labels[idx],
        "confidence": round(confidence * 100, 2),
        "info": f"Automated analytical profile generated for symptoms: {symptoms}."
    }

# --- ROUTES DEFINITIONS ---

@app.route("/")
def home():
    return jsonify({"message": "Health AI Backend Running Successfully"}), 200

@app.route('/chat', methods=['POST'])
def chatbot_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "Empty data request context payload received."}), 400
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "I did not receive any message context. Please try typing again."}), 400
            
        response = chatbot_model.generate_content(user_message)
        return jsonify({"reply": response.text}), 200
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        return jsonify({"reply": "I encountered a system routing error. Please try again later."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided inside data packet key schema"}), 400

        symptoms = data["symptoms"]

        # 1. Run through your rule-based check engine first
        try:
            rule_result = rule_based_prediction(symptoms)
            if rule_result and isinstance(rule_result, dict) and "disease" in rule_result:
                rule_result["source"] = "rule-based"
                return jsonify(rule_result), 200
        except Exception as rule_err:
            print(f"Rule evaluation skipped: {str(rule_err)}")

        # 2. Run through explicit imported module or local fallback compilation cleanly
        if predict_disease is not None:
            try:
                result = predict_disease(symptoms)
                result["source"] = "ml-model"
                return jsonify(result), 200
            except Exception as fn_err:
                print(f"Imported prediction module failed, running fallback: {str(fn_err)}")

        result = ml_predict(symptoms)
        result["source"] = "ml-model"
        return jsonify(result), 200
        
    except Exception as e:
        print(f"CRITICAL MODEL ROUTE CRASH: {str(e)}")
        return jsonify({"error": f"Internal predictive pipeline failure: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
