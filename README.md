# 📉 Bank Customer Churn Predictor

A Streamlit-based web application for predicting **bank customer churn** using a trained machine learning pipeline. This project helps banks identify customers at risk of leaving and supports strategic decision-making through insights, threshold tuning, and business impact simulations.

---

## 🚀 Features
- **Single Prediction**: Input customer details manually to predict churn probability.
- **Batch Prediction / Evaluation**: Upload a CSV to generate predictions for multiple customers, with evaluation if labels are included.
- **Model Info**: View pipeline details and model parameters.
- **Business Impact Simulation**: Estimate financial impact of churn and retention strategies.
- **Threshold Tuning**: Explore trade-offs between precision, recall, and decision threshold.

---

## 🧠 Model
- **Pipeline**: RobustScaler (numeric features) + OneHotEncoder (categorical features)
- **Algorithm**: Gradient Boosting (tuned)
- **Target Variable**: `Exited` (1 = churn, 0 = stay)

**Features used:**
1. `CreditScore`
2. `Geography`
3. `Gender`
4. `Age`
5. `Tenure`
6. `Balance`
7. `NumOfProducts`
8. `HasCrCard`
9. `IsActiveMember`
10. `EstimatedSalary`

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Make sure the trained model file `best_model.joblib` is placed in the project directory.

---

## 📊 Example Outputs
- **Prediction probability** for a single customer.
- **Batch prediction results** downloadable as CSV.
- **Confusion matrix** and key metrics (F1, Precision, Recall, ROC-AUC).
- **Business impact bar chart** showing intervention vs no intervention.
- **Precision-Recall curve** for threshold analysis.

---

## 🛠️ Requirements
See [requirements.txt](./requirements.txt) for full dependencies:
- streamlit==1.48.0
- pandas==2.3.0
- numpy==2.2.0
- joblib==1.5.1
- scikit-learn==1.6.1
- matplotlib==3.10.3
- seaborn==0.13.2

---

## 📌 Project Structure
```
├── streamlit_app.py   # Main Streamlit application
├── best_model.joblib  # Trained ML pipeline
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

---

## 💡 Author
Developed by Patrick Jonathan as part of a machine learning project on predicting bank customer churn.
