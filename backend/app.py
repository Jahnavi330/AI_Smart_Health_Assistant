import os
import json
import pickle
import numpy as np

# Prevent Render free tier memory threshold limits from cutting off computation threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Establish explicit absolute directory path tracking definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Force Python to look inside the active folder for your external dependencies
import sys
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import your custom engine files safely
from rules import rule_based_prediction
from predict import predict_disease  # Restored your native prediction pipeline logic

app = Flask(__name__)

# Complete global open-access configuration rule definition
CORS(app, resources={r"/*": {"origins": "*"}})
app.url_map.strict_slashes = False

# Configure the Gemini Engine safely
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

@app.route("/")
def home():
    return jsonify({"message": "Health AI Backend Running Successfully"}), 200

@app.route('/chat', methods=['POST'])
def chatbot_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "Empty data request payload."}), 400
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "I did not receive any message context."}), 400
            
        response = chatbot_model.generate_content(user_message)
        return jsonify({"reply": response.text}), 200
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        return jsonify({"reply": "I encountered a system routing error."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided inside payload"}), 400

        symptoms = data["symptoms"]

        # 1. Run through your rule-based check engine first
        try:
            rule_result = rule_based_prediction(symptoms)
            if rule_result and isinstance(rule_result, dict) and "disease" in rule_result:
                rule_result["source"] = "rule-based"
                return jsonify(rule_result), 200
        except Exception as rule_err:
            print(f"Rule evaluation skipped: {str(rule_err)}")

        # 2. Run through your custom native matrix mapping function from predict.py
        print(f"Passing symptoms data directly to predict_disease(): {symptoms}")
        result = predict_disease(symptoms)
        
        # Verify result dictionary structure payload parameters exist cleanly
        if not result or not isinstance(result, dict):
            raise ValueError("The predict_disease function returned an invalid empty data dictionary schema.")
            
        if "source" not in result:
            result["source"] = "ml-model"
            
        return jsonify(result), 200
        
    except Exception as e:
        # This will write out the exact line number that crashes directly onto your Render console logs!
        print(f"CRITICAL ML ROUTING ENGINE FAILURE: {str(e)}")
        return jsonify({"error": f"Internal predictive pipeline failure: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
