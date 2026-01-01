from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .prediction import router as prediction_router, model, scaler, mappings

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(prediction_router)

# Root endpoint
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Churn prediction API"
    }

# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "mappings_loaded": mappings is not None
    }
