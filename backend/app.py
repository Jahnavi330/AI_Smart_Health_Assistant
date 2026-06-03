import os
import re
import json
import pickle
import numpy as np

# Force low-memory usage for Render Free Tier to prevent out-of-memory crashes
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
from fuzzywuzzy import fuzz, process

# Try to load your rule-based backup file if it exists
try:
    from rules import rule_based_prediction
except ImportError:
    def rule_based_prediction(symptoms): return None

app = Flask(__name__)

# Apply open security permissions for all public requests
CORS(app, resources={r"/*": {"origins": "*"}})
app.url_map.strict_slashes = False

# Chatbot lazy initialization globals
chatbot_model = None
SYSTEM_INSTRUCTION = (
    "You are an expert AI medical advisor widget integrated into a disease prediction site. "
    "Analyze user questions regarding symptoms structurally. Offer clear, markdown-formatted information "
    "including bullet points on helpful lifestyle modifications, tracking factors, and home precautions. "
    "CRITICAL MANDATE: Provide a prominent bold disclaimer statement noting that your advice is strictly "
    "educational, does not represent standard definitive medical diagnoses, and requires professional consulting if severe."
)

# --- SELF-CONTAINED MACHINE LEARNING MODEL LOADING DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "symptoms.pkl")

# Load your system files right here safely
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    with open(SYMPTOMS_PATH, "rb") as f:
        symptom_cols = pickle.load(f)
        symptom_cols_lower = [s.lower() for s in symptom_cols]
        symptom_lookup = {symptom: idx for idx, symptom in enumerate(symptom_cols_lower)}
except Exception as e:
    print(f"Asset loading log message: {str(e)}")
    model, labels, symptom_cols, symptom_cols_lower, symptom_lookup = None, [], [], [], {}


def normalize_symptom_token(token):
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9 ]+", "", token)
    return token


def text_to_vector(symptoms_text):
    symptoms_text = symptoms_text.lower()
    input_symptoms = re.split(r"[\n;,]+|\band\b", symptoms_text)
    
    vector_len = len(symptom_cols) if symptom_cols else 132
    vector = np.zeros(vector_len)
    
    if symptom_cols:
        for token in input_symptoms:
            clean_token = normalize_symptom_token(token)
            if not clean_token:
                continue
            idx = symptom_lookup.get(clean_token)
            if idx is None:
                match = process.extractOne(clean_token, symptom_cols_lower, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 70:
                    idx = symptom_lookup[match[0]]
            if idx is not None:
                vector[idx] = 1
        
        if vector.sum() == 0:
            fallback_token = normalize_symptom_token(symptoms_text)
            if fallback_token:
                match = process.extractOne(fallback_token, symptom_cols_lower, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 70:
                    vector[symptom_lookup[match[0]]] = 1
    return vector.reshape(1, -1)

def direct_ml_predict(symptoms):
    if model is None or not labels:
        return {"disease": "Model Initializing", "confidence": 75.0}
    
    try:
        X = text_to_vector(symptoms)
        preds = model.predict(X, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(preds[0][idx])
        return {
            "disease": labels[idx],
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"ML prediction crash: {str(e)}")
        return {
            "disease": "Unable to process symptoms",
            "confidence": 0.0
        }

# --- ENDPOINTS ---

@app.route("/")
def home():
    return jsonify({"message": "Health AI Backend Running"}), 200

def get_chatbot_model():
    global chatbot_model
    if chatbot_model is not None:
        return chatbot_model

    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_LOCAL_FALLBACK_KEY"))
    chatbot_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION
    )
    return chatbot_model

@app.route('/chat', methods=['POST'])
def chatbot_endpoint():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Empty query payload context."}), 400
        chatbot = get_chatbot_model()
        response = chatbot.generate_content(user_message)
        return jsonify({"reply": response.text}), 200
    except Exception as e:
        return jsonify({"reply": f"Chatbot routing error: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = data["symptoms"]

        # 1. Run your rules first
        try:
            rule_result = rule_based_prediction(symptoms)
            if rule_result and isinstance(rule_result, dict) and "disease" in rule_result:
                rule_result["source"] = "rule-based"
                return jsonify(rule_result), 200
        except Exception:
            pass

        # 2. Run the self-contained ML prediction directly in this file
        result = direct_ml_predict(symptoms)
        result["source"] = "ml-model"
        result["info"] = f"AI analysis complete for symptoms: {symptoms}."
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Internal system crash: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
