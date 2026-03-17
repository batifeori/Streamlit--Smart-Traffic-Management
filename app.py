import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

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

    file_path = "smart_traffic_management_dataset.csv"

    if not os.path.exists(file_path):
        st.error("Dataset file not found")
        st.stop()

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    return df


df = load_data()

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "traffic_volume_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found")
        st.stop()

    return joblib.load(model_path)


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
# DASHBOARD
# ---------------------------------------------------

if page == "Dashboard Overview":

    st.title("🚦 Smart Traffic Management Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Traffic Volume",
        int(df["traffic_volume"].sum())
    )

    col2.metric(
        "Average Speed",
        round(df["avg_vehicle_speed"].mean(),2)
    )

    col3.metric(
        "Total Cars",
        int(df["vehicle_count_cars"].sum())
    )

    col4.metric(
        "Accidents",
        int(df["accident_reported"].sum())
    )

    st.divider()

    # Traffic by hour
    st.subheader("Traffic Volume by Hour")

    hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()

    fig_hour = px.line(hourly, x="hour", y="traffic_volume", markers=True)

    st.plotly_chart(fig_hour, use_container_width=True)

    # Vehicle distribution
    st.subheader("Vehicle Distribution")

    vehicle_data = pd.DataFrame({

        "Vehicle":["Cars","Trucks","Bikes"],

        "Count":[
            df["vehicle_count_cars"].sum(),
            df["vehicle_count_trucks"].sum(),
            df["vehicle_count_bikes"].sum()
        ]

    })

    fig_vehicle = px.pie(vehicle_data, names="Vehicle", values="Count", hole=0.4)

    st.plotly_chart(fig_vehicle, use_container_width=True)

    # Weather impact
    st.subheader("Traffic by Weather Condition")

    weather = df.groupby("weather_condition")["traffic_volume"].mean().reset_index()

    fig_weather = px.bar(weather, x="weather_condition", y="traffic_volume", color="weather_condition")

    st.plotly_chart(fig_weather, use_container_width=True)

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------

elif page == "Traffic Prediction":

    st.title("🚗 Traffic Volume Prediction")

    col1, col2, col3 = st.columns(3)

    location_id = col1.number_input("Location ID",1,100,1)

    avg_vehicle_speed = col1.slider("Average Vehicle Speed",10.0,120.0,60.0)

    vehicle_count_cars = col1.number_input("Cars",0,1000,100)

    vehicle_count_trucks = col2.number_input("Trucks",0,500,20)

    vehicle_count_bikes = col2.number_input("Bikes",0,500,10)

    temperature = col2.slider("Temperature", -10.0,45.0,20.0)

    humidity = col3.slider("Humidity",0.0,100.0,50.0)

    accident_reported = col3.selectbox("Accident Reported",[0,1])

    weather_condition = col3.selectbox(
        "Weather Condition",
        df["weather_condition"].unique()
    )

    signal_status = col3.selectbox(
        "Signal Status",
        df["signal_status"].unique()
    )

    hour = st.slider("Hour",0,23,12)

    day = st.slider("Day",1,31,15)

    month = st.slider("Month",1,12,6)

    if st.button("Predict Traffic Volume"):

        try:

            # Create dataframe
            input_data = pd.DataFrame({

                "location_id":[location_id],
                "avg_vehicle_speed":[avg_vehicle_speed],
                "vehicle_count_cars":[vehicle_count_cars],
                "vehicle_count_trucks":[vehicle_count_trucks],
                "vehicle_count_bikes":[vehicle_count_bikes],
                "temperature":[temperature],
                "humidity":[humidity],
                "accident_reported":[accident_reported],
                "weather_condition":[weather_condition],
                "signal_status":[signal_status],
                "hour":[hour],
                "day":[day],
                "month":[month]

            })

            # Ensure categorical types
            input_data["weather_condition"] = input_data["weather_condition"].astype(str)
            input_data["signal_status"] = input_data["signal_status"].astype(str)

            # Predict
            prediction = model.predict(input_data)

            predicted_volume = int(prediction[0])

            st.success(f"Predicted Traffic Volume: {predicted_volume}")

            fig = go.Figure(go.Indicator(

                mode="gauge+number",

                value=predicted_volume,

                title={"text":"Predicted Traffic Volume"},

                gauge={
                    "axis":{"range":[0,1000]},
                    "bar":{"color":"red"}
                }

            ))
