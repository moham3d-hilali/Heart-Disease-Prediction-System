import streamlit as st
import pandas as pd
import joblib

# Load the trained model

model = joblib.load("heart_model.pkl")

# Page Configuration

st.set_page_config(
page_title="Heart Disease Prediction System",
page_icon="❤️"
)

# Main Title

st.title("Heart Disease Prediction System")

st.write(
"Smart Clinical Decision Support Tool "
"for Heart Disease Prediction"
)

# System Overview

st.sidebar.subheader("System Overview")
st.sidebar.write("Algorithm: Decision Tree")
st.sidebar.write("Libraries: Streamlit, Pandas, Scikit-learn")

# Patient Information

st.subheader("Patient Information")

age = st.number_input(
"Patient Age:",
min_value=1,
max_value=120,
value=45
)

blood_pressure = st.number_input(
"Resting Blood Pressure (mm Hg):",
min_value=50,
max_value=250,
value=120
)

cholesterol = st.number_input(
"Serum Cholesterol (mg/dL):",
min_value=100,
max_value=500,
value=200
)

chest_pain = st.selectbox(
"Chest Pain Type:",
options=[0, 1, 2, 3]
)

# Prediction

if st.button("Analyze & Predict Risk"):

    patient_data = pd.DataFrame({
        "age": [age],
        "trestbps": [blood_pressure],
        "chol": [cholesterol],
         "cp": [chest_pain]
    })

    prediction = model.predict(patient_data)[0]

    if prediction == 1:
        st.error(
            "The model predicts a higher likelihood of heart disease."
        )
    else:
        st.success(
            "The model predicts a lower likelihood of heart disease."
        )


# Disclaimer

st.info(
"Educational project only. This tool does not replace "
"professional medical diagnosis."
)

