import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("smart_traffic_management_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    return df

df = load_data()

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("traffic_volume_model.pkl")
    return model

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🚦 Smart Traffic Dashboard")

page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard Overview", "Traffic Prediction"]
)

# ---------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------

if page == "Dashboard Overview":

    st.title("🚦 Smart Traffic Management Dashboard")

    st.markdown("Real-time insights from traffic monitoring data.")

    # KPI METRICS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Traffic Volume",
        int(df["traffic_volume"].sum())
    )

    col2.metric(
        "Average Speed",
        round(df["avg_vehicle_speed"].mean(), 2)
    )

    col3.metric(
        "Total Cars",
        int(df["vehicle_count_cars"].sum())
    )

    col4.metric(
        "Accidents Reported",
        int(df["accident_reported"].sum())
    )

    st.divider()

    # ---------------------------------------------------
    # TRAFFIC BY HOUR
    # ---------------------------------------------------

    st.subheader("Traffic Volume by Hour")

    hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()

    fig_hour = px.line(
        hourly,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    st.plotly_chart(fig_hour, use_container_width=True)

    # ---------------------------------------------------
    # VEHICLE DISTRIBUTION
    # ---------------------------------------------------

    st.subheader("Vehicle Type Distribution")

    vehicle_data = pd.DataFrame({
        "Vehicle Type": ["Cars", "Trucks", "Bikes"],
        "Count": [
            df["vehicle_count_cars"].sum(),
            df["vehicle_count_trucks"].sum(),
            df["vehicle_count_bikes"].sum()
        ]
    })

    fig_vehicle = px.pie(
        vehicle_data,
        values="Count",
        names="Vehicle Type",
        hole=0.4
    )

    st.plotly_chart(fig_vehicle, use_container_width=True)

    # ---------------------------------------------------
    # WEATHER IMPACT
    # ---------------------------------------------------

    st.subheader("Traffic Volume by Weather Condition")

    weather_data = df.groupby("weather_condition")["traffic_volume"].mean().reset_index()

    fig_weather = px.bar(
        weather_data,
        x="weather_condition",
        y="traffic_volume",
        color="weather_condition"
    )

    st.plotly_chart(fig_weather, use_container_width=True)

    # ---------------------------------------------------
    # TRAFFIC VS SPEED
    # ---------------------------------------------------

    st.subheader("Traffic Volume vs Vehicle Speed")

    fig_scatter = px.scatter(
        df,
        x="avg_vehicle_speed",
        y="traffic_volume",
        color="weather_condition",
        opacity=0.6
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------
# TRAFFIC PREDICTION PAGE
# ---------------------------------------------------

elif page == "Traffic Prediction":

    st.title("🚗 Traffic Volume Prediction")

    st.markdown("Enter traffic conditions to predict traffic volume.")

    col1, col2, col3 = st.columns(3)

    location_id = col1.number_input("Location ID", 1, 50, 1)

    avg_vehicle_speed = col1.slider(
        "Average Vehicle Speed",
        10.0, 120.0, 60.0
    )

    vehicle_count_cars = col1.number_input(
        "Cars Count",
        0, 500, 100
    )

    vehicle_count_trucks = col2.number_input(
        "Trucks Count",
        0, 200, 20
    )

    vehicle_count_bikes = col2.number_input(
        "Bikes Count",
        0, 200, 10
    )

    temperature = col2.slider(
        "Temperature",
        -10.0, 45.0, 20.0
    )

    humidity = col3.slider(
        "Humidity",
        0.0, 100.0, 50.0
    )

    weather_condition = col3.selectbox(
        "Weather Condition",
        df["weather_condition"].unique()
    )

    signal_status = col3.selectbox(
        "Signal Status",
        df["signal_status"].unique()
    )

    hour = st.slider("Hour of Day", 0, 23, 12)

    day = st.slider("Day", 1, 31, 15)

    month = st.slider("Month", 1, 12, 6)

    if st.button("Predict Traffic Volume"):

        input_data = pd.DataFrame({
            "location_id":[location_id],
            "avg_vehicle_speed":[avg_vehicle_speed],
            "vehicle_count_cars":[vehicle_count_cars],
            "vehicle_count_trucks":[vehicle_count_trucks],
            "vehicle_count_bikes":[vehicle_count_bikes],
            "temperature":[temperature],
            "humidity":[humidity],
            "weather_condition":[weather_condition],
            "signal_status":[signal_status],
            "hour":[hour],
            "day":[day],
            "month":[month]
        })

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Traffic Volume: {int(prediction[0])}"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(prediction[0]),
            title={"text": "Predicted Traffic Volume"},
            gauge={
                "axis": {"range": [0, 1000]},
                "bar": {"color": "red"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)