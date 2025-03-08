import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the trained models
with open("D:\ML Folders\ml_env\GitHub\Hack-it-out\Hack-It-Out\models\wind_model.pkl", "rb") as wind_model_file:
    wind_model = pickle.load(wind_model_file)

with open("D:\ML Folders\ml_env\GitHub\Hack-it-out\Hack-It-Out\models\solar_model.pkl", "rb") as solar_model_file:
    solar_model = pickle.load(solar_model_file)

# Load sample datasets for scaling
wind_data = pd.read_csv("D:\ML Folders\ml_env\GitHub\Hack-it-out\Hack-It-Out\models\Datasets\Cleaned Data.csv")  # Ensure this file exists
solar_data = pd.read_csv("D:\ML Folders\ml_env\GitHub\Hack-it-out\Hack-It-Out\models\Datasets\complete_solar_data.csv")  # Ensure this file exists

# 🔥 Fix: Extract Hour, Day, Month from 'Date/Time' in Wind Data
wind_data["Date/Time"] = pd.to_datetime(wind_data["Date/Time"], format="%m/%d/%Y %H:%M")
wind_data["Hour"] = wind_data["Date/Time"].dt.hour
wind_data["Day"] = wind_data["Date/Time"].dt.day
wind_data["Month"] = wind_data["Date/Time"].dt.month

# Define feature columns
wind_features = ["Wind Speed (m/s)", "Wind Direction", "Hour", "Day", "Month"]
solar_features = [col for col in solar_data.columns if col not in ["AC_POWER", "DATE_TIME"]]

# Fit scalers
wind_scaler = StandardScaler()
wind_scaler.fit(wind_data[wind_features])

solar_scaler = StandardScaler()
solar_scaler.fit(solar_data[solar_features])

# Streamlit UI
st.title("Renewable Energy Prediction System 🌞💨")

st.sidebar.header("Select Prediction Model")
option = st.sidebar.radio("Choose a model:", ("Wind Power Prediction", "Solar Power Prediction"))

# Function for Wind Power Prediction
def predict_wind_power():
    st.header("Wind Power Prediction 🌬️")

    # Input fields
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)
    wind_direction = st.number_input("Wind Direction (°)", min_value=0, max_value=360, step=1)
    date_time_str = st.text_input("Enter Date/Time (YYYY-MM-DD HH:MM:SS)", "2025-03-01 15:30:00")

    if st.button("Predict Wind Power"):
        try:
            date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            hour, day, month = date_time.hour, date_time.day, date_time.month

            # Prepare input data
            wind_input = np.array([[wind_speed, wind_direction, hour, day, month]])
            wind_input_scaled = wind_scaler.transform(wind_input)

            # Prediction
            wind_prediction = wind_model.predict(wind_input_scaled)
            st.success(f"Predicted Wind LV ActivePower: {wind_prediction[0]:.2f} kW")

        except Exception as e:
            st.error(f"Error: {e}")


# Function for Solar Power Prediction
def predict_solar_power():
    st.header("Solar Power Prediction ☀️")

    # Input fields for all solar features
    solar_inputs = {}
    for col in solar_features:
        solar_inputs[col] = st.number_input(f"Enter {col}", step=0.1)

    if st.button("Predict Solar Power"):
        try:
            # Prepare input data
            solar_input_array = np.array(list(solar_inputs.values())).reshape(1, -1)
            solar_input_scaled = solar_scaler.transform(solar_input_array)
            solar_input_reshaped = np.reshape(solar_input_scaled, (1, 1, solar_input_scaled.shape[1]))  # For LSTM input

            # 🔥 FIX: Use `solar_model.predict()`, not `solar_scaler.predict()`
            solar_prediction = solar_model.predict(solar_input_reshaped)
            st.success(f"Predicted Solar AC Power: {solar_prediction[0][0]:.2f} kW")

        except Exception as e:
            st.error(f"Error: {e}")

# Execute the selected model
if option == "Wind Power Prediction":
    predict_wind_power()
elif option == "Solar Power Prediction":
    predict_solar_power()
