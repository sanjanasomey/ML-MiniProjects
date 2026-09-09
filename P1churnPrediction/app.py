import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction System")

credit_score = st.number_input("Credit Score", value=600)

age = st.number_input("Age", value=30)

tenure = st.number_input("Tenure", value=5)

balance = st.number_input("Balance", value=50000.0)

num_products = st.number_input("Number of Products", value=2)

has_card = st.selectbox("Has Credit Card", [0, 1])

active_member = st.selectbox("Active Member", [0, 1])

salary = st.number_input("Estimated Salary", value=50000.0)

country = st.selectbox("Country", ["France", "Germany", "Spain"]
)

gender = st.selectbox("Gender",  ["Male", "Female"]
)

germany = 1 if country == "Germany" else 0
spain = 1 if country == "Spain" else 0
male = 1 if gender == "Male" else 0

if st.button("Predict"):
    data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        active_member,
        salary,
        germany,
        spain,
        male
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer Will Exit")
    else:
        st.success("Customer Will Stay")