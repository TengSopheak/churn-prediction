from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import pickle
import json
from sqlalchemy.orm import Session
from .database import get_db
from .models import Prediction
# import boto3, uuid
# from config import S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

router = APIRouter()

# Request model
class CustomerData(BaseModel):
    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Global artifacts (loaded at module level)
model = None
scaler = None
mappings = None
class_map = {}

def load_artifacts():
    """Load model artifacts at startup"""
    global model, scaler, mappings, class_map
    
    # Load model on startup
    try:
        model_path = Path("./models/churn_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model not found. Predictions will fail until model file is provided.")
    except Exception as e:
        print(f"ERROR loading model: {e}")

    # Load scaler (for numerical features)
    try:
        scaler_path = Path("./models/scaler.pkl")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")

    # Load encoding mappings
    try:
        mappings_path = Path("../json/encoding_mappings.json")
        with open(mappings_path, "r") as f:
            mappings = json.load(f)
        print("Encoding mappings loaded successfully.")
        
        # Set up class mapping - derive class info for human-readable output
        if model:
            classes = list(getattr(model, "classes_", []))
            if set(classes) == {0, 1}:
                class_map = {0: "Not Churn", 1: "Churn"}
            else:
                class_map = {c: str(c) for c in classes}
    except Exception as e:
        print(f"ERROR loading encoding mappings: {e}")

    ''' Debug info: print loaded feature names '''
    # if hasattr(model, "feature_names_in_"):
    #     print(f"\n[DEBUG] Model features: {list(model.feature_names_in_)}")

# Load artifacts when module is imported
load_artifacts()

def encode_feature(value: str, feature_name: str):
    """Convert value using mapping rules from JSON file"""          
    # Handle other categorical features
    feature_map = mappings.get(feature_name, {})
    if value in feature_map:
        return feature_map[value]
    
    # Fallback for unseen categories
    print(f"Warning: Unseen category '{value}' for {feature_name}. Using default 0.")
    return 0

# s3 = boto3.client("s3", region_name=AWS_REGION,
#                   aws_access_key_id=AWS_ACCESS_KEY_ID,
#                   aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#                   aws_session_token=AWS_SESSION_TOKEN
# )

# def upload_json_to_s3(obj: dict, key_prefix="logs/"):
#     key = f"{key_prefix}{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex}.json"
#     s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj).encode("utf-8"))
#     return key

@router.post("/predict")
def predict(customer: CustomerData, db: Session = Depends(get_db)):
    # Validate all artifacts
    if not all([model, scaler, mappings]):
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not fully loaded.")
    
    ''' Below is for validating each artifact individually (debugging purposes)'''
    # # Check if model is loaded
    # if model is None:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Prediction model not loaded.")
    
    # # Check if scaler is loaded
    # if scaler is None:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Scaler not loaded.")
    
    # # Check if json mappings are loaded
    # if mappings is None:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Encoding mappings (json) not loaded.")

    try:
        # Encode all categorical inputs
        encoded = {
            "SeniorCitizen": int(customer.SeniorCitizen),  # Already integer
            "Partner": encode_feature(customer.Partner, "Partner"),
            "Dependents": encode_feature(customer.Dependents, "Dependents"),
            "tenure": float(customer.tenure),  # Will scale later
            "InternetService": encode_feature(customer.InternetService, "InternetService"),
            "OnlineBackup": encode_feature(customer.OnlineBackup, "OnlineBackup"),
            "DeviceProtection": encode_feature(customer.DeviceProtection, "DeviceProtection"),
            "StreamingTV": encode_feature(customer.StreamingTV, "StreamingTV"),
            "StreamingMovies": encode_feature(customer.StreamingMovies, "StreamingMovies"),
            "Contract": encode_feature(customer.Contract, "Contract"),
            "PaperlessBilling": encode_feature(customer.PaperlessBilling, "PaperlessBilling"),
            "PaymentMethod": encode_feature(customer.PaymentMethod, "PaymentMethod"),
            "MonthlyCharges": float(customer.MonthlyCharges),  # Will scale later
            "TotalCharges": float(customer.TotalCharges)     # Will scale later
        }

        scaler_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        # Prepare data for scaling in EXACT order used during training
        numerical_df = pd.DataFrame([[
            encoded["tenure"],
            encoded["MonthlyCharges"],
            encoded["TotalCharges"]
        ]], columns=scaler_cols)

        # Apply scaling
        scaled_nums = scaler.transform(numerical_df)[0]  # Get first (only) row

        # Update with scaled values
        encoded.update({
            "tenure": scaled_nums[0],
            "MonthlyCharges": scaled_nums[1],
            "TotalCharges": scaled_nums[2]
        })

        # Prepare feature vector in training order
        feature_order = [
            "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
            "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    
        X_df = pd.DataFrame([[encoded[feat] for feat in feature_order]], columns=feature_order)

        # Make prediction
        pred = model.predict(X_df)[0]
        prob = model.predict_proba(X_df)[0][1] if hasattr(model, "predict_proba") else None # Just class 1 probability
        label_name = class_map.get(pred, str(pred))

        # Probability logic
        if prob is not None:
            churn_prob = round(prob * 100, 2)
            probability = float(churn_prob if label_name == "Churn" else 100 - churn_prob)
        else:
            probability = None

        # Save prediction to database
        db_prediction = Prediction(
            gender=customer.Gender,
            senior_citizen=customer.SeniorCitizen,
            partner=customer.Partner,
            dependents=customer.Dependents,
            tenure=customer.tenure,
            phone_service=customer.PhoneService,
            multiple_lines=customer.MultipleLines,
            internet_service=customer.InternetService,
            online_security=customer.OnlineSecurity,
            online_backup=customer.OnlineBackup,
            device_protection=customer.DeviceProtection,
            tech_support=customer.TechSupport,
            streaming_tv=customer.StreamingTV,
            streaming_movies=customer.StreamingMovies,
            contract=customer.Contract,
            paperless_billing=customer.PaperlessBilling,
            payment_method=customer.PaymentMethod,
            monthly_charges=customer.MonthlyCharges,
            total_charges=customer.TotalCharges,
            prediction_label=label_name,
            probability = probability
        )

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        # for uploading prediction record to S3 (if needed)
        # record = {
        #     "input": customer.model_dump(),
        #     "prediction": int(pred),
        #     "label": label_name,
        #     "probability": f"{round(prob * 100, 2)}%"
        # }
        
        # upload_key = upload_json_to_s3(record)

        return {
            "result": int(pred),
            "label": label_name,
            "probability": f"{round(prob * 100, 2)}%",
            "prediction_id": db_prediction.id
            # "s3_key": upload_key
            }
    
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Failed to load prediction model. Model may be incompatible."
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )