import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("smart_traffic_management_dataset.csv")

model = load_model()
df = load_data()

# ----------------------------
# PREPARE TRAINING FEATURES
# ----------------------------
X = df.drop(columns=["accident_reported", "timestamp"])
X_encoded = pd.get_dummies(X, drop_first=True)

# ----------------------------
# TITLE
# ----------------------------
st.title("🚦 Smart Traffic Accident Prediction System")
st.markdown("### Interactive Machine Learning Dashboard")

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("🔧 Input Features")

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
# ENCODING (CRITICAL PART)
# ----------------------------
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Align EXACTLY with training columns
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# ----------------------------
# PREDICTION
# ----------------------------
prediction = model.predict(input_encoded)[0]
probability = model.predict_proba(input_encoded)[0][1]

# ----------------------------
# KPI DISPLAY
# ----------------------------
st.subheader("🎯 Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Prediction", "Accident 🚨" if prediction == 1 else "No Accident ✅")

with col2:
    st.metric("Probability", f"{probability:.2%}")

with col3:
    st.metric("Speed", f"{input_dict['avg_vehicle_speed']} km/h")

# ----------------------------
# PROBABILITY VISUAL
# ----------------------------
st.subheader("📊 Prediction Confidence")

prob_df = pd.DataFrame({
    "Outcome": ["No Accident", "Accident"],
    "Probability": model.predict_proba(input_encoded)[0]
})

fig_prob = px.pie(prob_df, names="Outcome", values="Probability", hole=0.4)
st.plotly_chart(fig_prob, use_container_width=True)

# ----------------------------
# DASHBOARD VISUALS
# ----------------------------
st.subheader("📈 Traffic Insights")

col4, col5 = st.columns(2)

with col4:
    fig1 = px.histogram(df, x="traffic_volume", title="Traffic Volume Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    fig2 = px.histogram(df, x="avg_vehicle_speed", title="Speed Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# ACCIDENT ANALYSIS
# ----------------------------
st.subheader("🚨 Accident Overview")

accident_df = df["accident_reported"].value_counts().reset_index()
accident_df.columns = ["Category", "Count"]

fig3 = px.bar(accident_df, x="Category", y="Count", text="Count")
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# WEATHER IMPACT
# ----------------------------
st.subheader("🌦️ Weather Impact")

fig4 = px.box(df, x="weather_condition", y="traffic_volume")
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# DATA PREVIEW
# ----------------------------
with st.expander("📄 View Dataset"):
    st.dataframe(df.head(50))

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("✅ MSc Data Science Project | Smart Traffic Prediction System")