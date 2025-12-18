import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="KNN Purchase Predictor",
    page_icon="🛒",
    layout="centered"
)

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🧠 KNN Purchase Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict whether a user will purchase based on Age & Salary</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------ LOAD DATA ------------------
df = pd.read_csv("Social_Network_Ads.csv")

X = df[['Age', 'EstimatedSalary']].values[:10]
y = df['Purchased'].values[:10]

# ------------------ SCALING ------------------
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# ------------------ MODEL ------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# ------------------ USER INPUT ------------------
st.subheader("👤 Enter Customer Details")

age = st.slider("🎂 Age", min_value=18, max_value=60, value=30)
salary = st.slider("💰 Estimated Salary", min_value=15000, max_value=150000, value=60000, step=5000)

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Purchase"):
    new_data = np.array([[age, salary]])
    new_data_scaled = sc.transform(new_data)
    prediction = model.predict(new_data_scaled)

    st.divider()

    if prediction[0] == 1:
        st.success("✅ **Prediction: Purchased** 🛍️")
    else:
        st.error("❌ **Prediction: Not Purchased**")

# ------------------ VISUALIZATION ------------------
st.divider()
st.subheader("📊 Visualization")

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(len(X)):
    if y[i] == 1:
        ax.scatter(X[i][0], X[i][1], color='green', label='Purchased' if i == 0 else "")
    else:
        ax.scatter(X[i][0], X[i][1], color='red', label='Not Purchased' if i == 1 else "")

# New customer point
ax.scatter(age, salary, color='blue', marker='x', s=150, label='New Customer')

ax.set_xlabel("Age")
ax.set_ylabel("Estimated Salary")
ax.set_title("KNN Decision Space")
ax.legend()
ax.grid(True)

st.pyplot(fig)

