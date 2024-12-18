import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Load model and scaler
model, scaler = load_model()

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🔮 Customer Churn Prediction")
st.write("""
    This application uses a machine learning model to predict 
    whether a customer is likely to churn (cancel their subscription).
""")

# Sidebar for user inputs
st.sidebar.header("📝 Enter Customer Features")

def user_input_features():
    input_method = st.sidebar.radio(
        "Input Method:",
        options=["Sliders", "Manual Input"]
    )

    if input_method == "Sliders":
        # Slider inputs
        account_length = st.sidebar.slider("Account Length (in months)", 1, 120, 24)
        intl_plan = st.sidebar.selectbox("International Plan", options=["No", "Yes"], index=0)
        vm_plan = st.sidebar.selectbox("Voicemail Plan", options=["No", "Yes"], index=0)
        total_day_minutes = st.sidebar.slider("Daytime Call Minutes", 0, 500, 200)
        total_day_calls = st.sidebar.slider("Daytime Calls", 0, 200, 100)
        total_eve_minutes = st.sidebar.slider("Evening Call Minutes", 0, 500, 150)
        total_eve_calls = st.sidebar.slider("Evening Calls", 0, 200, 100)
        total_night_minutes = st.sidebar.slider("Nighttime Call Minutes", 0, 500, 100)
        total_night_calls = st.sidebar.slider("Nighttime Calls", 0, 200, 50)
        total_intl_minutes = st.sidebar.slider("International Call Minutes", 0, 50, 10)
        total_intl_calls = st.sidebar.slider("International Calls", 0, 20, 5)
        customer_service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 2)
        total_monthly_charges = st.sidebar.slider("Total Monthly Charges (in $)", 0, 200, 70)

    else:
        # Manual inputs
        account_length = st.sidebar.number_input("Account Length (in months)", value=24)
        intl_plan = st.sidebar.selectbox("International Plan", options=["No", "Yes"], index=0)
        vm_plan = st.sidebar.selectbox("Voicemail Plan", options=["No", "Yes"], index=0)
        total_day_minutes = st.sidebar.number_input("Daytime Call Minutes", value=200.0)
        total_day_calls = st.sidebar.number_input("Daytime Calls", value=100)
        total_eve_minutes = st.sidebar.number_input("Evening Call Minutes", value=150.0)
        total_eve_calls = st.sidebar.number_input("Evening Calls", value=100)
        total_night_minutes = st.sidebar.number_input("Nighttime Call Minutes", value=100.0)
        total_night_calls = st.sidebar.number_input("Nighttime Calls", value=50)
        total_intl_minutes = st.sidebar.number_input("International Call Minutes", value=10.0)
        total_intl_calls = st.sidebar.number_input("International Calls", value=5)
        customer_service_calls = st.sidebar.number_input("Customer Service Calls", value=2)
        total_monthly_charges = st.sidebar.number_input("Total Monthly Charges (in $)", value=70.0)

    data = {
        'Account length': account_length,
        'International plan': 1 if intl_plan == "Yes" else 0,
        'Voice mail plan': 1 if vm_plan == "Yes" else 0,
        'Total day minutes': total_day_minutes,
        'Total day calls': total_day_calls,
        'Total eve minutes': total_eve_minutes,
        'Total eve calls': total_eve_calls,
        'Total night minutes': total_night_minutes,
        'Total night calls': total_night_calls,
        'Total intl minutes': total_intl_minutes,
        'Total intl calls': total_intl_calls,
        'Customer service calls': customer_service_calls,
        'Total Monthly Charges': total_monthly_charges
    }
    return pd.DataFrame([data])

# Input data from user
input_df = user_input_features()

# Ensure input_df column names match the training data
input_df = input_df[['Account length', 'International plan', 'Voice mail plan', 'Total day minutes',
                     'Total day calls', 'Total eve minutes', 'Total eve calls', 'Total night minutes',
                     'Total night calls', 'Total intl minutes', 'Total intl calls', 'Customer service calls',
                     'Total Monthly Charges']]

# Show input data in an expandable section
with st.expander("📊 User Input Data"):
    st.write(input_df)

# Preprocess the data
@st.cache_data
def preprocess_data(_scaler, input_data):
    # Select only numeric columns
    numeric_columns = ['Account length', 'Total day minutes', 'Total day calls', 
                       'Total eve minutes', 'Total eve calls', 'Total night minutes',
                       'Total night calls', 'Total intl minutes', 'Total intl calls',
                       'Customer service calls', 'Total Monthly Charges']
    input_data[numeric_columns] = _scaler.transform(input_data[numeric_columns])
    return input_data

processed_input = preprocess_data(scaler, input_df)

# Make prediction
if st.button("🔍 Predict"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    st.subheader("📈 Prediction Results")
    churn_label = "⚠️ High Risk of Churn" if prediction[0] == 1 else "✅ Low Risk of Churn"
    st.write(f"**Prediction** : {churn_label}")
    st.progress(int(prediction_proba[0] * 100))  # Add a visual progress bar
    st.metric(label="Churn Probability", value=f"{prediction_proba[0]:.2%}")
