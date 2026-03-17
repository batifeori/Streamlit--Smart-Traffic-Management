import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------------
# HEADER
# -----------------------------------
st.title("🚦 Smart Traffic Management System")
st.write("Predict traffic conditions using a trained Machine Learning model.")

# -----------------------------------
# LOAD MODEL ONLY (NO SCALER)
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------
st.sidebar.header("Input Parameters")

vehicle_count = st.sidebar.slider("Vehicle Count", 0, 200, 50)
avg_speed = st.sidebar.slider("Average Speed (km/h)", 0, 120, 40)
signal_time = st.sidebar.slider("Signal Time (seconds)", 0, 180, 60)

# Prepare input
input_data = np.array([[vehicle_count, avg_speed, signal_time]])

# -----------------------------------
# DISPLAY METRICS
# -----------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Vehicle Count", vehicle_count)
col2.metric("Average Speed", avg_speed)
col3.metric("Signal Time", signal_time)

st.divider()

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.sidebar.button("Predict Traffic"):
    prediction = model.predict(input_data)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(input_data).max()

    # -----------------------------------
    # RESULT OUTPUT
    # -----------------------------------
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🟢 Low Traffic")
    elif prediction == 1:
        st.warning("🟡 Moderate Traffic")
    elif prediction == 2:
        st.error("🔴 Heavy Traffic")
    else:
        st.info("Unknown result")

    if confidence is not None:
        st.write(f"Confidence: {confidence * 100:.2f}%")

# -----------------------------------
# VISUALIZATION
# -----------------------------------
st.subheader("Input Overview")

data = pd.DataFrame({
    "Feature": ["Vehicle Count", "Average Speed", "Signal Time"],
    "Value": [vehicle_count, avg_speed, signal_time]
})

st.bar_chart(data.set_index("Feature"))

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption("Smart Traffic Management System | Machine Learning Application")