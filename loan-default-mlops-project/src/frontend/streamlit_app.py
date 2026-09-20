import streamlit as st
import requests

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Default Prediction")

st.write("Enter customer details below")

# ==========================================
# INPUTS
# ==========================================

income = st.number_input(
    "Income",
    value=200000.0
)

credit = st.number_input(
    "Credit Amount",
    value=500000.0
)

annuity = st.number_input(
    "Annuity",
    value=25000.0
)

birth = st.number_input(
    "Days Birth",
    value=-12000
)

employed = st.number_input(
    "Days Employed",
    value=-2000
)

# ==========================================
# BUTTON
# ==========================================

if st.button("Predict"):

    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": birth,
        "DAYS_EMPLOYED": employed
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        st.write(result)

        prediction = result["prediction"]
        probability = result["default_probability"]

        st.subheader("Prediction Result")

        st.write(
            f"Prediction: {prediction}"
        )

        st.write(
            f"Default Probability: {probability:.2%}"
        )

        if prediction == "Defaulter":
            st.error("⚠ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )