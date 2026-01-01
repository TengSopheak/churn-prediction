from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    gender = Column(String)
    senior_citizen = Column(Integer)
    partner = Column(String)
    dependents = Column(String)
    tenure = Column(Integer)
    phone_service = Column(String)
    multiple_lines = Column(String)
    internet_service = Column(String)
    online_security = Column(String)
    online_backup = Column(String)
    device_protection = Column(String)
    tech_support = Column(String)
    streaming_tv = Column(String)
    streaming_movies = Column(String)
    contract = Column(String)
    paperless_billing = Column(String)
    payment_method = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    prediction_label = Column(String)  # "Not Churn" or "Churn"
    probability = Column(Float)  # Probability of churn
