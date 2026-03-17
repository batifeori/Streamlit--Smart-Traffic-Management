import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart Traffic Accident Predictor",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------
# FILE PATHS (IMPORTANT FOR CLOUD)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "smart_traffic_management_dataset.csv")

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------
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
# PREPARE TRAINING STRUCTURE
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

input_dict = {
    "location_id": st.sidebar.selectbox("Location ID", sorted(df["location_id"].unique())),
    "traffic_volume": slider("traffic_volume"),
    "avg_vehicle_speed": slider("avg_vehicle_speed"),
    "vehicle_count_cars": slider("vehicle_count_cars"),
    "vehicle_count_trucks": slider("vehicle_count_trucks"),
    "vehicle_count_bikes": slider("vehicle_count_bikes"),
    "temperature": slider("temperature"),
    "humidity": slider("humidity"),
    "weather_condition": st.sidebar.selectbox("Weather Condition", sorted(df["weather_condition"].unique())),
    "signal_status": st.sidebar.selectbox("Signal Status", sorted(df["signal_status"].unique()))
}

input_df = pd.DataFrame([input_dict])

# ----------------------------
# SAFE PREDICTION FUNCTION
# ----------------------------
def make_prediction(input_df, model, X_encoded):
    # Encode input
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Ensure numeric format
    input_encoded = input_encoded.astype(float)

    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    return prediction, probability

prediction, probability = make_prediction(input_df, model, X_encoded)

# ----------------------------
# MAIN HEADER
# ----------------------------
st.title("🚦 Smart Traffic Accident Predictor")
st.markdown("### Machine Learning Dashboard for Accident Risk Prediction")

# ----------------------------
# KPI CARDS (PROFESSIONAL UI)
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accident Probability", f"{probability:.2%}")

with col2:
    st.metric(
        "Prediction",
        "🚨 Accident Likely" if prediction == 1 else "✅ No Accident"
    )

with col3:
    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    st.metric("Risk Level", risk_level)

# ----------------------------
# GAUGE CHART
# ----------------------------
st.subheader("📊 Risk Gauge")

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
# INPUT SUMMARY
# ----------------------------
st.subheader("📋 Input Summary")

summary_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Value": input_df.iloc[0].values
})

st.dataframe(summary_df, use_container_width=True)

# ----------------------------
# DATA INSIGHTS
# ----------------------------
st.subheader("📈 Dataset Insights")

col4, col5 = st.columns(2)

with col4:
    st.bar_chart(df["traffic_volume"])

with col5:
    st.bar_chart(df["avg_vehicle_speed"])

# ----------------------------
# DEBUG (REMOVE AFTER TESTING)
# ----------------------------
with st.expander("⚙️ Debug Info"):
    st.write("Model expects:", model.n_features_in_)
    st.write("Input shape:", pd.get_dummies(input_df).shape)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("✅ MSc Data Science Project | Smart Traffic Prediction System")