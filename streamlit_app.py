import os
import io
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
TARGET = "Exited"  # 1 = churn, 0 = stay
MODEL_PATH = "best_model.joblib"

DEFAULTS = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 60000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0,
}

GEOGRAPHY_OPTIONS = ["France", "Spain", "Germany"]
GENDER_OPTIONS = ["Male", "Female"]

# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Place your trained pipeline (joblib) next to app.py."
        )
    return joblib.load(path)


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURES]


def evaluate_with_labels(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    metrics = {
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "Precision (positive=1)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (positive=1)": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        pass
    return metrics

# ----------------------------
# Sidebar — global controls
# ----------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    threshold = st.slider(
        "Decision Threshold (positive = churn)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Predicted probability ≥ threshold → labeled as churn (1).",
    )
    st.markdown(
        """
        **Tip**: If your business wants to *minimize false negatives* (i.e., avoid missing actual churners),
        consider a slightly lower threshold (e.g., 0.35–0.45). If you want *fewer false positives*, raise it.
        """
    )

# ----------------------------
# Load model once
# ----------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.title("📉 Bank Customer Churn Predictor")
st.caption("Pipeline: RobustScaler (num) + OneHot (Geography, Gender) → Gradient Boosting (tuned)")

# Tabs for UX
TAB_HOME, TAB_SINGLE, TAB_BATCH, TAB_MODEL, TAB_EXPLAIN, TAB_THRESHOLD = st.tabs([
    "Overview",
    "Single Prediction",
    "Batch Predict / Evaluate",
    "Model Info",
    "Business Impact",
    "Threshold Tuning"
])

# ----------------------------
# Tab: Overview
# ----------------------------
with TAB_HOME:
    st.subheader("What this app does")
    st.markdown(
        """
        - Predicts the probability that a customer will **churn** (1 = exit) based on 10 inputs.  
        - **Single Prediction** via manual input form for one customer.  
        - **Batch Predict / Evaluate** with CSV upload (supports evaluation if you include the true label column `Exited`).  
        - **Model Info** to display training metrics and feature details.  
        - **Business Impact** simulation to estimate potential losses and net benefits from retention strategies.  
        - **Threshold Tuning** to adjust decision threshold and optimize metrics such as F2-score.  
                
        **Expected feature columns** in order:
        1. `CreditScore` (int) — typically 300–900
        2. `Geography` (France/Spain/Germany)
        3. `Gender` (Male/Female)
        4. `Age` (int)
        5. `Tenure` (int years with bank)
        6. `Balance` (float)
        7. `NumOfProducts` (int, 1–4)
        8. `HasCrCard` (0/1)
        9. `IsActiveMember` (0/1)
        10. `EstimatedSalary` (float)
        """
    )

# ----------------------------
# Tab: Single Prediction
# ----------------------------
with TAB_SINGLE:
    st.subheader("🔮 Predict a single customer")

    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cs = st.number_input("CreditScore", min_value=300, max_value=1000, value=DEFAULTS["CreditScore"], step=1)
            age = st.number_input("Age", min_value=18, max_value=100, value=DEFAULTS["Age"], step=1)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, value=DEFAULTS["Tenure"], step=1)
            nprod = st.number_input("NumOfProducts", min_value=1, max_value=6, value=DEFAULTS["NumOfProducts"], step=1)
        with col2:
            geo = st.selectbox("Geography", GEOGRAPHY_OPTIONS, index=GEOGRAPHY_OPTIONS.index(DEFAULTS["Geography"]))
            gender = st.selectbox("Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(DEFAULTS["Gender"]))
            hcc = st.selectbox("HasCrCard", [0, 1], index=DEFAULTS["HasCrCard"])
            iam = st.selectbox("IsActiveMember", [0, 1], index=DEFAULTS["IsActiveMember"])
        with col3:
            bal = st.number_input("Balance", min_value=0.0, max_value=1_000_000.0, value=float(DEFAULTS["Balance"]), step=100.0)
            sal = st.number_input("EstimatedSalary", min_value=0.0, max_value=2_000_000.0, value=float(DEFAULTS["EstimatedSalary"]), step=100.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([
            {
                "CreditScore": int(cs),
                "Geography": str(geo),
                "Gender": str(gender),
                "Age": int(age),
                "Tenure": int(tenure),
                "Balance": float(bal),
                "NumOfProducts": int(nprod),
                "HasCrCard": int(hcc),
                "IsActiveMember": int(iam),
                "EstimatedSalary": float(sal),
            }
        ])
        try:
            input_df = ensure_feature_order(input_df)
            proba = float(model.predict_proba(input_df)[0, 1])
            pred = int(proba >= threshold)

            st.metric(
                label="Churn Probability (P[y=1])",
                value=f"{proba:.3f}",
                delta="Churn" if pred == 1 else "Stay",
                delta_color="inverse" if pred == 1 else "normal",
            )

            st.code(input_df.to_string(index=False), language="text")
        except Exception as ex:
            st.error(f"Prediction failed: {ex}")

# ----------------------------
# Tab: Batch Predict / Evaluate
# ----------------------------
with TAB_BATCH:
    st.subheader("📦 Batch predict from CSV")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(io.StringIO(up.getvalue().decode("utf-8")))

        st.write("Preview:")
        st.dataframe(df.head(10))

        has_label = TARGET in df.columns
        X_df = ensure_feature_order(df)

        with st.spinner("Scoring..."):
            prob = model.predict_proba(X_df)[:, 1]
            pred = (prob >= threshold).astype(int)

        out = df.copy()
        out["churn_proba"] = prob
        out["churn_pred"] = pred

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

        if has_label:
            st.subheader("Evaluation (using uploaded labels)")
            metrics = evaluate_with_labels(df[TARGET].values, prob, threshold)
            colm1, colm2, colm3, colm4 = st.columns(4)
            colm1.metric("F1 (macro)", f"{metrics.get('F1 (macro)', np.nan):.3f}")
            colm2.metric("Precision (pos=1)", f"{metrics.get('Precision (positive=1)', np.nan):.3f}")
            colm3.metric("Recall (pos=1)", f"{metrics.get('Recall (positive=1)', np.nan):.3f}")
            if "ROC-AUC" in metrics:
                colm4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

            cm = confusion_matrix(df[TARGET].values, pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha='center', va='center')
            st.pyplot(fig)

# ----------------------------
# Tab: Model Info
# ----------------------------
with TAB_MODEL:
    st.subheader("🧠 Pipeline & Parameters")
    try:
        params = model.get_params()
        keys_of_interest = [k for k in params.keys() if k.startswith("model__")]
        if not keys_of_interest:
            keys_of_interest = sorted(params.keys())
        info_df = pd.DataFrame({"param": keys_of_interest, "value": [params[k] for k in keys_of_interest]})
        st.dataframe(info_df, use_container_width=True)
    except Exception as ex:
        st.error(f"Could not read model parameters: {ex}")


# ----------------------------
# Tab: Business Impact
# ----------------------------
with TAB_EXPLAIN:
    st.subheader("💼 Business Impact Simulation")
    st.write("""This simulation helps you understand the potential profit or loss from customer retention strategies. You can adjust the business assumptions to see their impact.""")

    # --- Input Asumsi Bisnis ---
    with st.expander("📊 Business Assumptions", expanded=True):
        total_customers = st.number_input("Total Customers", min_value=100, value=1000, step=100)
        predicted_churn_rate = st.slider("Predicted Churn Rate (%)", 0, 100, 20, 1)
        retention_cost = st.number_input("Retention Cost per Customer (Rp)", min_value=0, value=100000)
        customer_value = st.number_input("Customer Lifetime Value (CLV)", min_value=0, value=1000000)

    # --- Perhitungan ---
    predicted_churn_customers = total_customers * (predicted_churn_rate / 100)

    # Strategy 1: Tidak ada intervensi
    loss_no_action = predicted_churn_customers * customer_value

    # Strategy 2: Intervensi (anggap semua prediksi churn diintervensi)
    intervention_cost = predicted_churn_customers * retention_cost
    saved_revenue = predicted_churn_customers * customer_value
    net_gain = saved_revenue - intervention_cost

    # --- Output ---
    st.write("### 📈 Simulation Results")
    st.markdown(f"""
    - **Without intervention** → Potential loss: **Rp {loss_no_action:,.0f}**
    - **With intervention (using the model)**  
    - Intervention cost: Rp {intervention_cost:,.0f}  
    - Revenue retained: Rp {saved_revenue:,.0f}  
    - **Net Impact: Rp {net_gain:,.0f}**
    """)

    # --- Visualisasi ---
    impact_data = pd.DataFrame({
        "Strategi": ["Tanpa Intervensi", "Dengan Model"],
        "Net Impact": [-loss_no_action, net_gain]
    })

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=impact_data, x="Strategi", y="Net Impact", ax=ax)
    ax.set_ylabel("Rp")
    ax.set_title("Perbandingan Dampak Bisnis")
    st.pyplot(fig)

    st.info("Ubah input di atas untuk melihat simulasi dengan skenario berbeda.")


# ----------------------------
# Tab: Threshold Tuning
# ----------------------------
with TAB_THRESHOLD:
    st.subheader("⚖️ Threshold Tuning")
    st.write("Explore trade-offs between Precision and Recall by adjusting the threshold.")

    uploaded_eval = st.file_uploader("Upload CSV with labels for PR Curve", type=["csv"], key="thr")
    if uploaded_eval is not None:
        df_thr = pd.read_csv(uploaded_eval)
        if TARGET not in df_thr.columns:
            st.error("CSV must include 'Exited' column for threshold tuning.")
        else:
            X_thr = ensure_feature_order(df_thr)
            prob_thr = model.predict_proba(X_thr)[:, 1]
            y_true_thr = df_thr[TARGET].values

            prec, rec, thrsh = precision_recall_curve(y_true_thr, prob_thr)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(thrsh, prec[:-1], label="Precision")
            ax.plot(thrsh, rec[:-1], label="Recall")
            ax.axvline(threshold, color='red', linestyle='--', label=f"Current thr={threshold:.2f}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Precision-Recall vs Threshold")
            ax.legend()
            st.pyplot(fig)
