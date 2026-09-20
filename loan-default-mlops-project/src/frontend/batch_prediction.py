import streamlit as st
import pandas as pd
import requests


# =========================
# PAGE TITLE
# =========================

st.title("Batch Loan Default Prediction")


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# =========================
# IF FILE IS UPLOADED
# =========================

if uploaded_file is not None:

    # Read CSV file
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")

    st.dataframe(df.head())


    # Store predictions
    predictions = []

    probabilities = []


    # Loading spinner
    with st.spinner("Generating predictions..."):


        # Loop through each row
        for _, row in df.iterrows():

            # Convert row to dictionary
            data = row.to_dict()


            # Send request to FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data
            )


            # Check response
            if response.status_code == 200:

                result = response.json()

                predictions.append(
                    result["prediction"]
                )

                probabilities.append(
                    result["default_probability"]
                )

            else:

                predictions.append("Error")

                probabilities.append(None)


    # =========================
    # ADD RESULTS TO DATAFRAME
    # =========================

    df["Prediction"] = predictions

    df["Default_Probability"] = probabilities


    # Success message
    st.success("Batch prediction completed successfully!")


    # Show results
    st.write("Prediction Results")

    st.dataframe(df.head())


    # =========================
    # DOWNLOAD BUTTON
    # =========================

    csv = df.to_csv(index=False).encode("utf-8")


    st.download_button(
        "Download Predictions CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )