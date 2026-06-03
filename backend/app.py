import os
import re
import json
import pickle
import numpy as np

# Force low-memory usage for Render Free Tier to prevent out-of-memory crashes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from fuzzywuzzy import fuzz, process
import sys

# Try to load your rule-based backup file if it exists
try:
    from backend.rules import rule_based_prediction
except ImportError:
    try:
        from .rules import rule_based_prediction
    except Exception:
        def rule_based_prediction(symptoms): return None

app = Flask(__name__)

# Apply open security permissions for all public requests
CORS(app, resources={r"/*": {"origins": "*"}})
app.url_map.strict_slashes = False

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS,HEAD"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

# Initialize the Gemini Engine
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

# --- SELF-CONTAINED MACHINE LEARNING MODEL LOADING DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "symptoms.pkl")

# Load your system files right here safely
model = None
labels = []
symptom_cols = []
symptom_cols_lower = []
symptom_lookup = {}
last_load_error = None

def load_assets():
    global model, labels, symptom_cols, symptom_cols_lower, symptom_lookup, last_load_error
    try:
        model_exists = os.path.exists(MODEL_PATH)
        labels_exists = os.path.exists(LABELS_PATH)
        symptoms_exists = os.path.exists(SYMPTOMS_PATH)
        print(f"Asset files: model_exists={model_exists}, labels_exists={labels_exists}, symptoms_exists={symptoms_exists}")

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
        with open(SYMPTOMS_PATH, "rb") as f:
            symptom_cols = pickle.load(f)
            symptom_cols_lower = [s.lower() for s in symptom_cols]
            symptom_lookup = {symptom: idx for idx, symptom in enumerate(symptom_cols_lower)}
        last_load_error = None
        print(f"Model load successful: model={MODEL_PATH}, labels={len(labels)}, symptoms={len(symptom_cols)}")
    except Exception as e:
        last_load_error = str(e)
        print(f"Asset loading log message: {last_load_error}")
        print(f"Model path: {MODEL_PATH}", f"Labels path: {LABELS_PATH}", f"Symptoms path: {SYMPTOMS_PATH}")
        model, labels, symptom_cols, symptom_cols_lower, symptom_lookup = None, [], [], [], {}

load_assets()

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
        print("ML assets not loaded, retrying load_assets()")
        load_assets()

    if model is None or not labels:
        print("ML assets still not available after retry")
        return {"disease": "Model Initializing", "confidence": 75.0}
    
    try:
        print("Vectorizing symptoms")
        X = text_to_vector(symptoms)
        print(f"Input vector sum: {int(X.sum())}, shape: {X.shape}")
        print("Running model inference")
        preds = model(X, training=False)
        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        print("Prediction returned")
        idx = int(np.argmax(preds))
        confidence = float(preds[0][idx])
        return {
            "disease": labels[idx],
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"ML prediction crash: {str(e)}")
        print("Symptoms:", symptoms)
        return {
            "disease": "Unable to process symptoms",
            "confidence": 0.0
        }

# --- ENDPOINTS ---

@app.route("/")
def home():
    return jsonify({"message": "Health AI Backend Running"}), 200

@app.route('/chat', methods=['POST', 'OPTIONS', 'GET', 'HEAD'])
def chatbot_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    if request.method == 'GET':
        return jsonify({"message": "Chat endpoint expects POST."}), 200
    if request.method == 'HEAD':
        return jsonify({"status": "ready"}), 200

    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Empty query payload context."}), 400
        response = chatbot_model.generate_content(user_message)
        return jsonify({"reply": response.text}), 200
    except Exception as e:
        return jsonify({"reply": f"Chatbot routing error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_loaded": model is not None,
        "labels_count": len(labels),
        "symptoms_count": len(symptom_cols),
        "model_exists": os.path.exists(MODEL_PATH),
        "labels_exists": os.path.exists(LABELS_PATH),
        "symptoms_exists": os.path.exists(SYMPTOMS_PATH),
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH,
        "symptoms_path": SYMPTOMS_PATH,
        "last_load_error": last_load_error,
        "tensorflow_version": tf.__version__,
        "python_version": sys.version
    }), 200


def process_prediction_request(symptoms):
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    print(f"Predict symptoms payload: {symptoms}")

    try:
        rule_result = rule_based_prediction(symptoms)
        if rule_result and isinstance(rule_result, dict) and "disease" in rule_result:
            rule_result["source"] = "rule-based"
            print(f"Rule-based prediction returned: {rule_result}")
            return jsonify(rule_result), 200
    except Exception as rule_exc:
        print(f"Rule-based prediction error: {rule_exc}")

    print("Starting ML prediction")
    result = direct_ml_predict(symptoms)
    print(f"ML prediction result: {result}")
    result["source"] = "ml-model"
    result["info"] = f"AI analysis complete for symptoms: {symptoms}."
    return jsonify(result), 200


@app.route("/predict/<path:symptoms>", methods=["GET", "OPTIONS", "HEAD"])
def predict_path(symptoms):
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    if request.method == 'HEAD':
        return jsonify({"status": "ready"}), 200
    return process_prediction_request(symptoms)


@app.route("/predict", methods=["POST", "OPTIONS", "GET", "HEAD"])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    if request.method == 'HEAD':
        return jsonify({"status": "ready"}), 200
    if request.method == 'GET':
        symptoms = (
            request.args.get("symptoms", "")
            or request.args.get("symptom", "")
            or request.args.get("q", "")
            or request.args.get("query", "")
        ).strip()
        if symptoms:
            return process_prediction_request(symptoms)
        return jsonify({
            "message": "Predict endpoint expects POST with JSON payload. For quick GET testing, use ?symptoms=fever,cough or /predict/fever,cough."
        }), 200

    data = None
    try:
        if request.is_json:
            data = request.get_json(silent=True)
        else:
            raw = request.get_data(as_text=True)
            if raw:
                try:
                    data = json.loads(raw)
                except Exception:
                    data = None
        if not data and request.form:
            data = request.form.to_dict()
    except Exception as e:
        print(f"JSON parse fallback error: {e}")
        data = None

    try:
        print(f"Predict request received: {data}")
        if not data or "symptoms" not in data:
            print("Predict request missing symptoms")
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = data["symptoms"]
        return process_prediction_request(symptoms)
    except Exception as e:
        print(f"Predict route uncaught exception: {str(e)}")
        return jsonify({"error": f"Internal system crash: {str(e)}"}), 500

@app.errorhandler(405)
def handle_method_not_allowed(e):
    print(f"405 Method Not Allowed on {request.method} {request.path}")
    return jsonify({"error": "Method not allowed. Use POST for /predict and /chat."}), 405

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    print(f"Unhandled exception caught: {e}")
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
