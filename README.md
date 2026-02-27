# 📡 Full Stack Customer Churn Prediction System

This is a complete, full-stack machine learning project (Classification) designed to help a fictional telecom company retain their customers.

The goal is simple: **Predict whether a customer is likely to leave (churn) and provide a confidence score for that prediction.**

This repository covers everything from the raw data analysis and machine learning model training to a deployed backend API and a user-friendly frontend dashboard.

**Website link:** https://churn-prediction-frontend.onrender.com/

---

## 📂 Dataset Overview

The data used in this project comes from the **IBM Sample Datasets**, hosted on Hugging Face.

* **Dataset Link:** [Scikit-learn Churn Prediction Dataset](https://huggingface.co/datasets/scikit-learn/churn-prediction)

### What’s inside?

The dataset contains information about telecom customers, including:

* **Demographics:** Gender, senior citizen status, partners, dependents.
* **Services:** Phone service, internet type, streaming TV/movies, security.
* **Billing:** Contract type, payment method, monthly charges, total charges.
* **Target Variable (`Churn`):** The column we want to predict (Yes = Churn, No = Stay).

---

## 🖥️ Frontend Dashboard

The frontend is designed to be lightweight and fast. It doesn't rely on heavy frameworks like React or Angular. Instead, it uses:

* **HTML5** for structure.
* **TailwindCSS** for styling.
* **Vanilla JavaScript** for logic and API communication.

**How it works:**

1. The user enters customer details into a form.
2. The JavaScript sends this data to the backend API.
3. The dashboard displays the **Churn Prediction** (Yes/No) and a **Confidence Score** (e.g., "85% confidence").

*The frontend is currently deployed on **Render**.*

---

## ⚙️ Backend & API

The backend is the engine of the application, built with **FastAPI**.

**Key Responsibilities:**

* **API Endpoints:** Receives user inputs from the frontend.
* **Model Inference:** Loads the trained machine learning model to generate predictions in real-time.
* **Database Management:** Connects to a **Neon Postgres** database to store every user input and the resulting prediction.

**Reusable Pipelines:**
To ensure the code is production-ready, the logic from the Jupyter Notebooks (EDA, Feature Engineering, Modeling) was converted into reusable Python scripts (`feature_pipeline.py`, `model_experiment.py`) located in the `backend/ml/` folder.

---

## 📓 Notebooks & Pipeline Structure

The data science workflow is broken down into three logical steps:

1. **`01_eda.ipynb`**: Exploratory Data Analysis. We look at the data, visualize trends, and understand the problem.
2. **`02_feature_engineering.ipynb`** *(Converted to `feature_pipeline.py`)*: Cleaning and preparing the data.
3. **`03_model_experiment.ipynb`** *(Converted to `model_experiment.py`)*: Training and testing different algorithms.

**Why convert notebooks to scripts?**
Notebooks are great for experimentation, but scripts are better for production. Converting them allows the backend to import and use the exact same logic used during training, ensuring consistency.

---

## 🛠️ Feature Engineering

This is how we transform raw data into something the model can understand.

### 1. Preparation (Before Engineering)

* **Load Data:** Read the dataset from CSV.
* **Encode Target:** Convert the `Churn` column to 0 and 1.
* **Split Data:** We split the data into **Train** and **Test** sets *before* doing any processing. This prevents "data leakage" (cheating by seeing the test answers).

### 2. Engineering Steps

1. **Fix Data Types:** Convert `TotalCharges` from an object (string) to a number (`float64`).
2. **Handle Missing Values:** Fill missing `TotalCharges` with the **median** value.
3. **Outliers:** Checked, but no extreme treatment was needed.
4. **Encoding:** Convert categorical text (like "Yes/No", "Fiber Optic") into numbers. Numerical features are left alone.
5. **Scaling:** Apply **StandardScaler** to numerical features so they are on the same scale.
6. **Feature Selection:** We kept only features with an absolute correlation > 0.05 to the target.
7. **Balancing:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, so the model learns both "Churn" and "Stay" cases equally well.

---

## 🧪 Model Experimentation

We didn't just pick one model; we audited **9 different algorithms** to find the best one.

**Process:**

* Loaded the preprocessed train/test data.
* Used **5-Fold Cross-Validation** to test stability.

### Cross-Validation Results (F1 Score)

The **LightGBM** model came out on top with the highest Mean F1 score and low variance.

| Model | Mean F1 | Std Dev |
| --- | --- | --- |
| **LightGBM** | **0.853180** | **0.010274** |
| Random Forest | 0.847186 | 0.012595 |
| XGBoost | 0.845340 | 0.010613 |
| Gradient Boosting | 0.834098 | 0.011656 |
| KNN | 0.814083 | 0.007182 |
| Decision Tree | 0.782882 | 0.010607 |
| SVM | 0.780575 | 0.002204 |
| Logistic Regression | 0.777873 | 0.008128 |
| Naive Bayes | 0.763919 | 0.008477 |

---

## 🏆 Final Model & Evaluation

After selecting **LightGBM**, we performed **Hyperparameter Tuning** to squeeze out more performance. We then retrained the final model on the full balanced dataset and evaluated it against the held-out Test set.

### Performance Metrics

We evaluated models using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.7289 | 0.7969 | 0.7289 | 0.7439 |
| Decision Tree | 0.7140 | 0.7338 | 0.7140 | 0.7218 |
| Random Forest | 0.7573 | 0.7684 | 0.7573 | 0.7619 |
| Gradient Boosting | 0.7665 | 0.8010 | 0.7665 | 0.7764 |
| SVM | 0.7395 | 0.7951 | 0.7395 | 0.7530 |
| KNN | 0.7062 | 0.7565 | 0.7062 | 0.7206 |
| Naive Bayes | 0.7040 | 0.7923 | 0.7040 | 0.7212 |
| **XGBoost** | **0.7729** | **0.7840** | **0.7729** | **0.7773** |
| **LightGBM** | **0.7708** | **0.7803** | **0.7708** | **0.7747** |

* **Best Model:** LightGBM and XGBoost performed very similarly, with strong balance across precision and recall.
* **Weakest Models:** Naive Bayes and KNN struggled to capture the complexity of the data compared to the tree-based models.

### Deployment

The final, tuned **XGBoost** model was saved as a `.pkl` file. This file is loaded by the FastAPI backend to make live predictions.

---

## 📂 Repository Structure

The project is organized to keep the data science, backend, and frontend logic clean and separate.

```text
churn_prediction/
├─ backend/
│  ├─ app/
│  │  ├─ config.py          # Database and app configuration
│  │  ├─ database.py        # Database connection logic
│  │  ├─ main.py            # Main FastAPI application entry point
│  │  ├─ models.py          # SQL Alchemy models for database tables
│  │  ├─ prediction.py      # Logic to load model and predict
│  ├─ ml/
│  │  ├─ feature_pipeline.py # Script for processing raw data
│  │  ├─ model_experiment.py # Script for training models
│  ├─ models/
│  │  ├─ churn_model.pkl    # The saved final model
│  │  ├─ scaler.pkl         # Saved scaler for normalizing input
│  ├─ requirements.txt      # Python dependencies
│  ├─ start.sh
├─ database/
│  ├─ db.sql                # SQL script to setup Postgres tables
├─ dataset/
│  ├─ original/             # Raw CSV data
│  ├─ preprocessed_v1/      # Intermediate processed data
│  ├─ preprocessed_v2/      # Final processed data for training
├─ frontend/
│  ├─ index.html            # Dashboard UI structure
│  ├─ main.css              # TailwindCSS styling
│  ├─ script.js             # API calls and interaction logic
├─ json/
│  ├─ encoding_mappings.json # Mappings for categorical features
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_model_experiment.ipynb

```

* **`backend/`**: Holds the API, ML pipeline scripts, trained models, and startup files.
* **`backend/app/`**: The core API logic (configuration, database, routing).
* **`backend/ml/`**: The bridge between notebooks and production—reusable scripts.
* **`backend/models/`**: Where the "brain" (model) and "tools" (scaler) live.
* **`database/`**: Contains the SQL commands to create the necessary tables.
* **`dataset/`**: Stores data at every stage (Raw -> Processed -> Final).
* **`frontend/`**: The user interface.
* **`json/`**: Stores the rules for translating text to numbers (e.g., "Yes" = 1).
* **`notebooks/`**: The laboratory where we experiment before building the app.

---

## 🚀 Running the Project Locally

Follow these steps to get the system running on your machine.

**1. Clone the Repository**

```bash
git clone https://github.com/TengSopheak/churn-prediction
cd churn-prediction

```

**2. Backend Setup**
Navigate to the backend folder and install dependencies:

```bash
cd backend
pip install -r requirements.txt

```

**3. Database Setup**
Ensure you have a Postgres database running. The setup script is located at `database/db.sql`. Run the SQL commands inside that file to create your tables.

**4. Run the API**
In the `backend` root directory, run:

```bash
uvicorn app.main:app --reload --port 8000

```

The API is now live at `http://localhost:8000`.

**5. Frontend Setup**
Open `frontend/index.html` in your browser. It will connect to your local backend API.

---

## Conclusion

This project demonstrates a complete lifecycle of a data science project. We started with raw customer data, analyzed it, engineered features, and rigorously tested multiple models. We didn't stop at the notebook—we built a scalable FastAPI backend and a clean frontend dashboard to make the predictions accessible.

**Tech Stack:** Python, Scikit-Learn, XGBoost, LightGBM, FastAPI, PostgreSQL, HTML/JS/Tailwind.
