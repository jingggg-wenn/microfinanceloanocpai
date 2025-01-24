from flask import Flask, render_template, request, Response, session
from flask_session import Session
import pandas as pd
import numpy as np
import joblib
import requests
import json
import time
from lime.lime_tabular import LimeTabularExplainer
from transformers import AutoTokenizer
import torch

app = Flask(__name__)

# Session configuration
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize tokenizer
model_path = "./flan"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define inference URL for the hosted LLM
llm_infer_url = "http://modelmesh-serving.llm.svc.cluster.local:8008/v2/models/t5-pytorch/infer"

# Load necessary files
training_columns = joblib.load("training_columns.pkl")
best_model = joblib.load("best_model.joblib")
preprocessor = best_model.named_steps['preprocessor']

# Load training data for LIME initialization
X_train = pd.read_csv("X_train.csv")
X_train_processed = preprocessor.transform(X_train)

#ENDPOINTS
# Hosted Model API Endpoint
infer_url = "http://modelmesh-serving.demo.svc.cluster.local:8008/v2/models/xgb/infer"
headers = {"Content-Type": "application/json"}

# LIME Explainer Initialization
explainer = LimeTabularExplainer(
    training_data=X_train_processed,
    feature_names=training_columns,
    class_names=["Non-default", "Default"],
    mode="classification"
)

def call_llm_with_iterative_decoding(input_text):
    # Tokenize the input
    tokenized_input = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )

    # Extract tokenized tensors
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Initialize decoder input IDs with <pad> token
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)

    # Iterative decoding
    max_length = 80  # Maximum generation length
    generated_ids = []

    for _ in range(max_length):
        # Prepare payload for the current step
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                },
                {
                    "name": "decoder_input_ids",
                    "shape": list(decoder_input_ids.shape),
                    "datatype": "INT64",
                    "data": decoder_input_ids.flatten().tolist()
                }
            ]
        }

        # Send POST request
        response = requests.post(llm_infer_url, json=payload, headers=headers, verify=False)

        if response.status_code == 200:
            response_data = response.json()
            logits = response_data["outputs"][0]["data"]
            logits_tensor = torch.tensor(logits, device=device).view(1, -1, 32128)

            # Get the next token
            next_token_id = torch.argmax(logits_tensor[:, -1, :], dim=-1)
            generated_ids.append(next_token_id.item())

            # Break if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Update decoder_input_ids for the next iteration
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(0)], dim=-1)

        else:
            return f"Error {response.status_code}: {response.text}"

    # Decode the generated IDs to text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

@app.route('/explain', methods=['GET'])
def explain():
    processed_data = session.get("processed_data")
    probabilities = session.get("probabilities")

    if not processed_data or not probabilities:
        return Response("No data available for explanation.", content_type='text/plain')
    
    labeled_features = session.get("labeled_features")
    if not labeled_features:
        return Response("No labeled features available for explanation.", content_type='text/plain')

    # Collect explanations for each labeled feature
    explanations = []
    for feature in labeled_features:
        input_text = f"Instruction: Explain the feature.\nInput: {feature}"
        explanation = call_llm_with_iterative_decoding(input_text)
        explanations.append((feature, explanation))

    # Combine all explanations into a single string for streaming
    combined_explanation = "\n\n".join([explanation for _, explanation in explanations])

    # Define a generator for word-by-word streaming
    def generate_explanation():
        for word in combined_explanation.split():
            yield word + " "
            time.sleep(0.1)  # Simulate word-by-word streaming

    return Response(generate_explanation(), content_type='text/plain')


# @app.route('/explain', methods=['GET'])
# def explain():
#     processed_data = session.get("processed_data")
#     probabilities = session.get("probabilities")

#     if not processed_data or not probabilities:
#         return Response("No data available for explanation.", content_type='text/plain')
    
#     labeled_features = session.get("labeled_features")
#     if not labeled_features:
#         return Response("No labeled features available for explanation.", content_type='text/plain')

#     # Define a generator to stream explanations as they are generated
#     def generate_explanations():
#         for feature in labeled_features:
#             input_text = f"Instruction: Explain the feature in a user-friendly way.\nInput: {feature}"
#             explanation = call_llm_with_iterative_decoding(input_text)  # Generate explanation
            
#             # Stream the feature name first
#             # yield f"Feature: {feature}\n"
            
#             # Stream the explanation word by word
#             for word in explanation.split():
#                 yield word + " "
#                 time.sleep(0.1)  # Simulate word-by-word streaming
            
#             # Add spacing between explanations for readability
#             # yield "\n\n"

#     return Response(generate_explanations(), content_type='text/plain')

@app.route("/clear", methods=["GET"])
def clear():
    # Clear session data
    session.clear()
    # Redirect to the home page
    return render_template(
        "index.html",
        prediction=None,
        credit_score_category=None,
        probability_non_default=None,
        probability_default=None,
        lime_explanation=None,
        llm_explanation=None,
        top_features=None
    )

#POSTING

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    credit_score_category = None
    probability_non_default = None
    probability_default = None
    lime_explanation = None
    llm_explanation = None
    top_features = None
    labeled_features = []  # Store labeled features for "Default" or "Non-default"

    if request.method == "POST":
        try:
            # Input processing logic
            input_data = {key: value for key, value in request.form.items()}

            # Encode categorical features
            if input_data["NAME_EDUCATION_LEVEL"] == "Tertiary_qualification":
                input_data["NAME_EDUCATION_TYPE_Higher_education"] = 1
                input_data["NAME_EDUCATION_TYPE_Secondary_education"] = 0
            elif input_data["NAME_EDUCATION_LEVEL"] == "Secondary_education":
                input_data["NAME_EDUCATION_TYPE_Higher_education"] = 0
                input_data["NAME_EDUCATION_TYPE_Secondary_education"] = 1
            elif input_data["NAME_EDUCATION_LEVEL"] == "No_secondary_or_higher_education":
                input_data["NAME_EDUCATION_TYPE_Higher_education"] = 0
                input_data["NAME_EDUCATION_TYPE_Secondary_education"] = 0
            del input_data["NAME_EDUCATION_LEVEL"]

            if input_data["NAME_FAMILY_STATUS"] == "Married":
                input_data["NAME_FAMILY_STATUS_Married"] = 1
                input_data["NAME_FAMILY_STATUS_Single"] = 0
            elif input_data["NAME_FAMILY_STATUS"] == "Single":
                input_data["NAME_FAMILY_STATUS_Married"] = 0
                input_data["NAME_FAMILY_STATUS_Single"] = 1
            del input_data["NAME_FAMILY_STATUS"]

            if input_data["NAME_HOUSING_TYPE"] == "House_apartment":
                input_data["NAME_HOUSING_TYPE_House_apartment"] = 1
                input_data["NAME_HOUSING_TYPE_With_parents"] = 0
            elif input_data["NAME_HOUSING_TYPE"] == "With_parents":
                input_data["NAME_HOUSING_TYPE_House_apartment"] = 0
                input_data["NAME_HOUSING_TYPE_With_parents"] = 1
            del input_data["NAME_HOUSING_TYPE"]

            if input_data["OCCUPATION_TYPE"] == "Laborers":
                input_data["OCCUPATION_TYPE_Laborers"] = 1
                input_data["OCCUPATION_TYPE_Sales_staff"] = 0
            elif input_data["OCCUPATION_TYPE"] == "Sales_staff":
                input_data["OCCUPATION_TYPE_Laborers"] = 0
                input_data["OCCUPATION_TYPE_Sales_staff"] = 1
            del input_data["OCCUPATION_TYPE"]

            # Process binary fields
            input_data["FLAG_MOBIL"] = 1 if input_data["FLAG_MOBIL"] == "Yes" else 0
            input_data["FLAG_EMAIL"] = 1 if input_data["FLAG_EMAIL"] == "Yes" else 0
            input_data["FLAG_WORK_PHONE"] = 1 if input_data["FLAG_WORK_PHONE"] == "Yes" else 0

            # Convert years to days
            input_data["DAYS_BIRTH"] = int(float(input_data["DAYS_BIRTH"]) * -365.25)
            input_data["DAYS_EMPLOYED"] = int(float(input_data["DAYS_EMPLOYED"]) * -365.25)

            # Convert all remaining inputs to floats
            input_data = {key: float(value) for key, value in input_data.items()}

            # Preprocess input data
            input_data_df = pd.DataFrame([input_data]).reindex(columns=training_columns, fill_value=0)
            processed_data = preprocessor.transform(input_data_df).astype(np.float32)

            # Prepare payload for hosted API
            payload = {
                "inputs": [
                    {
                        "name": "float_input",
                        "shape": [1, len(processed_data.flatten())],
                        "datatype": "FP32",
                        "data": processed_data.flatten().tolist()
                    }
                ]
            }

            # Call Hosted Model API
            response = requests.post(infer_url, headers=headers, data=json.dumps(payload), verify=False)
            response_json = response.json()

            # Extract probabilities
            probabilities = None
            for output in response_json.get("outputs", []):
                if output["name"] == "probabilities":
                    probabilities = output["data"]
                    break

            if probabilities and len(probabilities) >= 2:
                probability_non_default = probabilities[0] * 100
                probability_default = probabilities[1] * 100
                prediction = "Default" if probabilities[1] > 0.5 else "Non-default"
                credit_score_category = classify_credit_score(probability_non_default)

            else:
                raise ValueError("API response is missing probabilities or has insufficient data.")

            if probabilities and len(probabilities) >= 2:
                probability_non_default = probabilities[0] * 100
                probability_default = probabilities[1] * 100
                prediction = "Default" if probabilities[1] > 0.5 else "Non-default"
                credit_score_category = classify_credit_score(probability_non_default)

                # Save processed data and probabilities for explanation
                session["processed_data"] = processed_data.tolist()
                session["probabilities"] = probabilities

                # Generate LIME Explanation
                def lime_predict_fn(data):
                    """LIME-compatible prediction function."""
                    repeated_probs = np.tile([probabilities[0], probabilities[1]], (data.shape[0], 1))
                    return repeated_probs

                exp = explainer.explain_instance(
                    data_row=processed_data[0],
                    predict_fn=lime_predict_fn,
                    num_features=3
                )
                
                lime_explanation = exp.as_list()

                # Extract raw feature names (remove conditions like '<= -0.21')
                top_features = [feature.split(" ")[0] for feature, _ in lime_explanation[:3]]
                print("Top 3 Features (Clean):", top_features)  # Debug print for clean feature names

                # Label the top 3 features based on the prediction
                if prediction == "Non-default":
                    labeled_features = [f"{feature}_NEGATIVE" for feature in top_features]
                    print("Labeled Features (Non-default):", labeled_features)  # Debug print for Non-default
                elif prediction == "Default":
                    labeled_features = [f"{feature}_POSITIVE" for feature in top_features]
                    print("Labeled Features (Default):", labeled_features)  # Debug print for Default

                # Store labeled features in the session
                session["labeled_features"] = labeled_features
                    
            else:
                raise ValueError("API response is missing probabilities or has insufficient data.")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    else:
        # Handle GET requests and initialize default values
        prediction = None
        credit_score_category = None
        probability_non_default = None
        probability_default = None
        lime_explanation = None
        top_features = None
        labeled_features = []
        # llm_explanation = None

    return render_template(
        "index.html",
        prediction=prediction,
        credit_score_category=credit_score_category,
        probability_non_default=round(probability_non_default, 2) if probability_non_default else None,
        probability_default=round(probability_default, 2) if probability_default else None,
        lime_explanation=lime_explanation,
        top_features=top_features,
        labeled_features = []
    )



def classify_credit_score(probability_non_default):
    if 98.89 <= probability_non_default <= 100.0:
        return "Excellent"
    elif 98.11 <= probability_non_default < 98.89:
        return "Excellent"
    elif 97.29 <= probability_non_default < 98.11:
        return "Good"
    elif 96.33 <= probability_non_default < 97.29:
        return "Good"
    elif 95.13 <= probability_non_default < 96.33:
        return "Good"
    elif 93.64 <= probability_non_default < 95.13:
        return "Fair"
    elif 91.54 <= probability_non_default < 93.64:
        return "Fair"
    elif 88.18 <= probability_non_default < 91.54:
        return "Poor"
    elif 81.58 <= probability_non_default < 88.18:
        return "Very Poor"
    elif 14.35 <= probability_non_default < 81.58:
        return "Very Poor"
    else:
        return "Uncategorized"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)