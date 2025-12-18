import streamlit as st
import numpy as np
import joblib

# Load model + scaler
model= joblib.load("knn_bundle.pkl")

st.set_page_config(page_title="Purchase Prediction App", layout="centered")

st.title("🛒 Purchase Prediction App")
st.write("Enter customer details to predict purchase")

age = st.number_input("Age", min_value=1, max_value=100, value=31)
salary = st.number_input("Salary", min_value=0, value=35000)

if st.button("Predict"):
    new_data = np.array([[age, salary]])


    prediction = model.predict(new_data)[0]

    if prediction == 1:
        st.success("✅ Customer WILL Purchase")
    else:
        st.warning("❌ Customer will NOT Purchase")
