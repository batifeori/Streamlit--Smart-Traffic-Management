import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# ----------------------------
# Load Model & Scaler
# ----------------------------
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.markdown("""
# 🚦 Smart Traffic Management Dashboard
Predict traffic congestion risk using Machine Learning
""")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🚗 Traffic Inputs")

traffic_volume = st.sidebar.slider("Traffic Volume", 0, 200, 50)
avg_speed = st.sidebar.slider("Avg Vehicle Speed", 0, 120, 40)

cars = st.sidebar.number_input("Cars", 0, 500, 50)
trucks = st.sidebar.number_input("Trucks", 0, 200, 10)
bikes = st.sidebar.number_input("Bikes", 0, 200, 20)

temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)

accident = st.sidebar.selectbox("Accident Reported", [0, 1])

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Cloudy", "Foggy", "Rainy", "Sunny", "Windy"]
)

signal = st.sidebar.selectbox(
    "Signal Status",
    ["Green", "Yellow", "Red"]
)

# ----------------------------
# Encoding (MATCH TRAINING)
# ----------------------------
weather_map = {
    "Cloudy": 0,
    "Foggy": 1,
    "Rainy": 2,
    "Sunny": 3,
    "Windy": 4
}

signal_map = {
    "Green": 0,
    "Red": 1,
    "Yellow": 2
}

weather_encoded = weather_map[weather]
signal_encoded = signal_map[signal]

# ----------------------------
# Time Features (IMPORTANT)
# ----------------------------
now = datetime.now()
hour = now.hour
day = now.day
month = now.month

# ----------------------------
# Feature Array (EXACT MATCH)
# ----------------------------
features = np.array([[
    traffic_volume,
    avg_speed,
    cars,
    trucks,
    bikes,
    temperature,
    humidity,
    accident,
    weather_encoded,
    signal_encoded,
    hour,
    day,
    month
]])

# ----------------------------
# Scaling
# ----------------------------
features_scaled = scaler.transform(features)

# ----------------------------
# Prediction
# ----------------------------
prob = model.predict_proba(features_scaled)[0][1]

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Traffic Congestion Probability",
        value=f"{prob*100:.2f}%"
    )

    # Progress Bar
    st.progress(float(prob))

    # Status Message
    if prob < 0.3:
        st.success("✅ Low Traffic - Smooth Flow")
    elif prob < 0.7:
        st.warning("⚠️ Moderate Traffic - Be Alert")
    else:
        st.error("🚨 High Traffic Congestion!")

with col2:
    st.subheader("📌 Current Conditions")

    st.write(f"🌤 Weather: **{weather}**")
    st.write(f"🚦 Signal: **{signal}**")
    st.write(f"🌡 Temp: **{temperature}°C**")
    st.write(f"💧 Humidity: **{humidity}%**")
    st.write(f"🚗 Vehicles: **{cars + trucks + bikes}**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")