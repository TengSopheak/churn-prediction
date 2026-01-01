-- Stores customer churn prediction results
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When prediction was made
    gender VARCHAR(10),
    senior_citizen INTEGER,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INTEGER,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(20),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),
    prediction_label VARCHAR(20), -- Predicted outcome (e.g., "Churn")
    probability DECIMAL(5,2) -- Confidence score (0.00 to 1.00)
);