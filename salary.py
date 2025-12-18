import streamlit as st
import numpy as np
import joblib

# ✅ LOAD MODEL + SCALER
model, scaler = joblib.load("knn_bundle.pkl")

st.title("🛒 Purchase Prediction App")

age = st.number_input("Age", 1, 100, 31)
salary = st.number_input("Salary", 0, 200000, 35000)

if st.button("Predict"):
    new_data = np.array([[age, salary]])
    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)[0]

    if prediction == 1:
        st.success("✅ Customer WILL Purchase")
    else:
        st.error("❌ Customer will NOT Purchase")
