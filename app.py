import streamlit as st
import numpy as np
import pickle

# Load model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Smart Traffic System", layout="wide")

# Title
st.title("🚦 Smart Traffic Management Dashboard")
st.markdown("Predict traffic congestion risk using ML")

# Sidebar
st.sidebar.header("Traffic Inputs")

traffic_volume = st.sidebar.number_input("Traffic Volume", 0, 1000, 200)
avg_speed = st.sidebar.number_input("Avg Vehicle Speed", 0.0, 120.0, 40.0)

cars = st.sidebar.number_input("Cars", 0, 500, 100)
trucks = st.sidebar.number_input("Trucks", 0, 200, 20)
bikes = st.sidebar.number_input("Bikes", 0, 300, 50)

temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)

accident = st.sidebar.selectbox("Accident Reported", [0, 1])

# Weather selection
weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Cloudy", "Foggy", "Rainy", "Sunny", "Windy"]
)

# Signal status
signal = st.sidebar.selectbox(
    "Signal Status",
    ["Green", "Red", "Yellow"]
)

# -----------------------------
# Encoding (MATCHES TRAINING)
# -----------------------------

# Weather encoding (Cloudy dropped)
weather_foggy = 1 if weather == "Foggy" else 0
weather_rainy = 1 if weather == "Rainy" else 0
weather_sunny = 1 if weather == "Sunny" else 0
weather_windy = 1 if weather == "Windy" else 0

# Signal encoding (Green dropped)
signal_red = 1 if signal == "Red" else 0
signal_yellow = 1 if signal == "Yellow" else 0

# Final feature array (ORDER IS CRITICAL)
features = np.array([[
    traffic_volume,
    avg_speed,
    cars,
    trucks,
    bikes,
    temperature,
    humidity,
    accident,
    weather_foggy,
    weather_rainy,
    weather_sunny,
    weather_windy,
    signal_red,
    signal_yellow
]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]
    else:
        prob = 0.0

    # -----------------------------
    # UI Output (Dashboard Style)
    # -----------------------------
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{prob*100:.2f}%")

    with col2:
        if prediction == 1:
            st.error("⚠️ High Traffic Risk")
        else:
            st.success("✅ Normal Traffic")

    with col3:
        if prob > 0.7:
            st.error("High Risk")
        elif prob > 0.4:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")

    st.markdown("---")

    st.subheader("Risk Gauge")
    st.progress(float(prob))