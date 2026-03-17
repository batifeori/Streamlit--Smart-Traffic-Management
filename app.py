import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Management Dashboard")

# -----------------------------------
# FILE PATHS (FIXES YOUR ERROR)
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    try:
        if not MODEL_PATH.exists():
            st.error("❌ logistic_model.pkl not found")
            st.stop()

        if not SCALER_PATH.exists():
            st.error("❌ scaler.pkl not found")
            st.stop()

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error("❌ Error loading model files")
        st.exception(e)
        st.stop()


model, scaler = load_model()

# -----------------------------------
# SIDEBAR INPUT
# -----------------------------------
st.sidebar.header("🔧 Input Parameters")

vehicle_count = st.sidebar.slider("Vehicle Count", 0, 200, 50)
avg_speed = st.sidebar.slider("Average Speed (km/h)", 0, 120, 40)
signal_time = st.sidebar.slider("Signal Time (sec)", 0, 180, 60)

# Feature array (MATCH YOUR TRAINING ORDER)
features = np.array([[vehicle_count, avg_speed, signal_time]])

# -----------------------------------
# PREDICTION
# -----------------------------------
prediction = None

if st.sidebar.button("🚀 Run Prediction"):
    try:
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

# -----------------------------------
# MAIN DASHBOARD
# -----------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🚗 Vehicles", vehicle_count)
col2.metric("⚡ Speed", avg_speed)
col3.metric("⏱ Signal Time", signal_time)

st.divider()

# -----------------------------------
# RESULT DISPLAY
# -----------------------------------
if prediction is not None:
    st.subheader("📊 Traffic Status Prediction")

    if prediction == 0:
        st.success("🟢 Low Traffic")
    elif prediction == 1:
        st.warning("🟡 Moderate Traffic")
    else:
        st.error("🔴 Heavy Traffic")

# -----------------------------------
# VISUALIZATION
# -----------------------------------
st.subheader("📈 Traffic Overview")

data = pd.DataFrame({
    "Feature": ["Vehicle Count", "Average Speed", "Signal Time"],
    "Value": [vehicle_count, avg_speed, signal_time]
})

st.bar_chart(data.set_index("Feature"))

# -----------------------------------
# FOOTER
# -----------------------------------
st.caption("Smart Traffic Management System • Streamlit Dashboard")