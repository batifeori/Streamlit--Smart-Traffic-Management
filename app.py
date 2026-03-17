import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Traffic ML Dashboard",
    layout="wide"
)

st.title("🚦 Smart Traffic Management Prediction System")

st.markdown("Predict accident probability using trained Logistic Regression model")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------

st.sidebar.header("Traffic Data Input")

location_id = st.sidebar.number_input("Location ID", 1, 10, 1)

traffic_volume = st.sidebar.number_input("Traffic Volume", 0, 5000, 500)

avg_vehicle_speed = st.sidebar.number_input("Average Vehicle Speed", 0.0, 120.0, 40.0)

vehicle_count_cars = st.sidebar.number_input("Car Count", 0, 2000, 200)

vehicle_count_trucks = st.sidebar.number_input("Truck Count", 0, 1000, 50)

vehicle_count_bikes = st.sidebar.number_input("Bike Count", 0, 1000, 60)

temperature = st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0)

humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Sunny", "Cloudy", "Foggy", "Rainy", "Windy"]
)

signal_status = st.sidebar.selectbox(
    "Signal Status",
    ["Red", "Yellow", "Green"]
)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def preprocess():

    # numeric features
    features = [
        location_id,
        traffic_volume,
        avg_vehicle_speed,
        vehicle_count_cars,
        vehicle_count_trucks,
        vehicle_count_bikes,
        temperature,
        humidity,
        hour
    ]

    # weather one-hot (drop Sunny baseline)

    weather_cloudy = 1 if weather == "Cloudy" else 0
    weather_foggy = 1 if weather == "Foggy" else 0
    weather_rainy = 1 if weather == "Rainy" else 0
    weather_windy = 1 if weather == "Windy" else 0

    features.extend([
        weather_cloudy,
        weather_foggy,
        weather_rainy,
        weather_windy
    ])

    # signal encoding

    signal_map = {
        "Red":0,
        "Yellow":1,
        "Green":2
    }

    signal_encoded = signal_map[signal_status]

    features.append(signal_encoded)

    return np.array(features).reshape(1,-1)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.sidebar.button("Predict Accident Risk"):

    X = preprocess()

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")

    col1,col2 = st.columns(2)

    with col1:

        if prediction == 1:
            st.error("⚠️ High Accident Risk Detected")
        else:
            st.success("✅ Low Accident Risk")

        st.metric(
            label="Accident Probability",
            value=f"{probability:.2%}"
        )

    with col2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={"text":"Accident Risk Level"},
            gauge={
                "axis":{"range":[0,1]},
                "bar":{"color":"red"},
                "steps":[
                    {"range":[0,0.3],"color":"lightgreen"},
                    {"range":[0.3,0.6],"color":"yellow"},
                    {"range":[0.6,1],"color":"salmon"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("Model Info")

st.sidebar.write("Model: Logistic Regression")

st.sidebar.write("Input Features: 14")

st.sidebar.write("Target: Accident Reported")