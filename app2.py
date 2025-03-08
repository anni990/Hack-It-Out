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

# Define features used in training
wind_features = ["Wind Speed (m/s)", "Wind Direction (°)", "Day", "Month", "Year"]
solar_features = ["Month", "Day"]  # Only Month & Day for solar model

# Fit scalers on the sample dataset
wind_scaler = StandardScaler()
wind_scaler.fit(wind_data[wind_features])

solar_scaler = StandardScaler()
solar_scaler.fit(solar_data[solar_features])

# Streamlit UI
st.title("Renewable Energy Prediction System 🌞💨")

st.sidebar.header("Select Prediction Model")
option = st.sidebar.radio("Choose a model:", ("Wind Power Prediction", "Solar Power Prediction"))

# Wind Power Prediction
if option == "Wind Power Prediction":
    st.header("Wind Power Prediction 🌬️")

    # Input fields
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)
    wind_direction = st.number_input("Wind Direction (°)", min_value=0, max_value=360, step=1)
    date_time_str = st.text_input("Enter Date/Time (YYYY-MM-DD HH:MM:SS)", "2025-03-01 15:30:00")

    if st.button("Predict Wind Power"):
        try:
            date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            day = date_time.day
            month = date_time.month
            year = date_time.year

            # Prepare input data
            wind_input = np.array([[wind_speed, wind_direction, day, month, year]])
            wind_input_scaled = wind_scaler.transform(wind_input)

            # Prediction
            wind_prediction = wind_model.predict(wind_input_scaled)
            st.success(f"Predicted Wind Power: {wind_prediction[0]:.2f} kW")

        except Exception as e:
            st.error(f"Error: {e}")

# Solar Power Prediction
elif option == "Solar Power Prediction":
    st.header("Solar Power Prediction ☀️")

    # Date input
    date_time_str = st.text_input("Enter Date (YYYY-MM-DD)", "2025-03-01")

    if st.button("Predict Solar Power"):
        try:
            date_time = datetime.strptime(date_time_str, "%Y-%m-%d")
            month = date_time.month
            day = date_time.day

            # Prepare input data
            solar_input = np.array([[month, day]])
            solar_input_scaled = solar_scaler.transform(solar_input)

            # Prediction
            solar_prediction = solar_model.predict(solar_input_scaled)
            st.success(f"Predicted Solar Power: {solar_prediction[0]:.2f} kW")

        except Exception as e:
            st.error(f"Error: {e}")
