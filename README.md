# 🚗 Car Insurance Claim Prediction App

A Machine Learning web application that predicts whether a policyholder is likely to file an insurance claim.

This app is built using:
- Python
- XGBoost
- SMOTE (Imbalanced Data Handling)
- Streamlit
- Scikit-learn

---

## 📌 Problem Statement

Insurance companies face significant losses due to unexpected claim filings.  
This project predicts whether a policyholder is likely to file a claim based on vehicle and policyholder information.

Target Variable:
- **is_claim**
  - 0 → No Claim
  - 1 → Claim

---

## 🧠 Model Pipeline

1. Data Cleaning
2. Feature Engineering
3. Manual Encoding
4. Feature Scaling
5. SMOTE for class imbalance
6. Feature Selection (Top 20)
7. XGBoost Classifier
8. Custom Probability Threshold (0.45)

---

## 📊 Key Features Used (Top 20)

- age_of_car  
- age_of_policyholder  
- displacement  
- max_torque_nm  
- policy_tenure  
- length  
- cylinder  
- height  
- model  
- gross_weight  
- max_power_rpm  
- transmission_type  
- max_power_bhp  
- width  
- area_cluster  
- max_torque_rpm  
- population_density  
- gear_box  
- segment  
- rear_brakes_type  

---

## 📈 Exploratory Data Analysis Includes

- Class imbalance visualization
- Feature count summaries
- Binary feature claim profiling
- KPI statistics (min, max, mean)
- Area cluster claim distribution
- Claim patterns by:
  - Make
  - Model
  - Engine type
  - Age groups

---

## 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

### Live App:
👉 https://car-insurance-app-7tylfzvkozh4qljrd9ljhf.streamlit.app/

---

## 📂 Project Structure

```
car-insurance-app/
│
├── app.py
├── requirements.txt
├── models/
│   ├── xgb_sm1_model.pkl
│   ├── scaler.pkl
│   └── model_features.pkl
└── car_encoded.csv
```

---

## ⚙ Installation (Local Run)

Clone the repository:

```
git clone https://github.com/avanthi-uc/car-insurance-app.git
```

Navigate to folder:

```
cd car-insurance-app
```

Install dependencies:

```
pip install -r requirements.txt
```

Run app:

```
streamlit run app.py
```

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Git & GitHub

---

## 👩‍💻 Author

Avanthi  

---

## ⭐ Future Improvements

- SHAP Explainability
- Model Monitoring
- Login Authentication
- API Version of Model
---
