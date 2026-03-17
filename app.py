import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------
# LOAD FILES (SAFE PATH)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "smart_traffic_management_dataset.csv")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# ----------------------------
# PREPROCESSING TEMPLATE
# ----------------------------
X = df.drop(columns=["accident_reported", "timestamp"])
X_encoded = pd.get_dummies(X, drop_first=True)

# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.header("🚗 Traffic Input")

def slider(col):
    return st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

input_data = {
    "location_id": st.sidebar.selectbox("Location ID", sorted(df["location_id"].unique())),
    "traffic_volume": slider("traffic_volume"),
    "avg_vehicle_speed": slider("avg_vehicle_speed"),
    "vehicle_count_cars": slider("vehicle_count_cars"),
    "vehicle_count_trucks": slider("vehicle_count_trucks"),
    "vehicle_count_bikes": slider("vehicle_count_bikes"),
    "temperature": slider("temperature"),
    "humidity": slider("humidity"),
    "weather_condition": st.sidebar.selectbox("Weather", df["weather_condition"].unique()),
    "signal_status": st.sidebar.selectbox("Signal", df["signal_status"].unique())
}

input_df = pd.DataFrame([input_data])

# ----------------------------
# ENCODING (CRITICAL)
# ----------------------------
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# ----------------------------
# PREDICTION
# ----------------------------
prediction = model.predict(input_encoded)[0]
probability = model.predict_proba(input_encoded)[0][1]

# ----------------------------
# HEADER
# ----------------------------
st.title("🚦 Smart Traffic Accident Predictor")
st.markdown("### Predicts likelihood of traffic accidents using ML")

# ----------------------------
# KPI SECTION (LIKE YOUR IMAGE)
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accident Probability", f"{probability:.1%}")

with col2:
    st.metric(
        "Prediction",
        "🚨 Accident Likely" if prediction == 1 else "✅ Safe"
    )

with col3:
    risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    st.metric("Risk Level", risk)

# ----------------------------
# GAUGE CHART (LIKE CHURN UI)
# ----------------------------
st.subheader("🚨 Risk Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={"text": "Accident Risk (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red"},
        "steps": [
            {"range": [0, 40], "color": "green"},
            {"range": [40, 70], "color": "orange"},
            {"range": [70, 100], "color": "red"},
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# INPUT SUMMARY TABLE
# ----------------------------
st.subheader("📋 Input Summary")

summary_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Value": input_df.iloc[0].values
})

st.dataframe(summary_df, use_container_width=True)

# ----------------------------
# INSIGHTS SECTION
# ----------------------------
st.subheader("📊 Traffic Insights")

col4, col5 = st.columns(2)

with col4:
    st.bar_chart(df["traffic_volume"])

with col5:
    st.bar_chart(df["avg_vehicle_speed"])

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("✅ MSc Data Science Project | Smart Traffic Prediction System")