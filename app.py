import streamlit as st
import pandas as pd

# Utils imports
from utils.data_loader import load_energy_data
from utils.weather import get_weather
from utils.preprocessing import create_features
from utils.decomposition import decompose
from utils.models import (
    prophet_model,
    random_forest_model,
    xgboost_model,
    arima_model,
    sarima_model
)
from utils.anomaly import autoencoder_anomaly

st.set_page_config(page_title="Energy Forecast Intelligence", layout="wide")

st.title("⚡ Energy Forecast Intelligence")

# -----------------------------
# 🔽 City Selection
# -----------------------------
city = st.selectbox(
    "Select City",
    ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"]
)

# -----------------------------
# 🔽 Model Type Selection
# -----------------------------
model_type = st.selectbox(
    "Select Model Type",
    ["Statistical", "Machine Learning"]
)

# -----------------------------
# 🔽 Model Selection (Dynamic)
# -----------------------------
if model_type == "Statistical":
    model_name = st.selectbox(
        "Select Model",
        ["Prophet", "ARIMA", "SARIMA"]
    )
else:
    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost"]
    )

# -----------------------------
# 🚀 Run Button
# -----------------------------
if st.button("Run Forecast"):

    st.subheader(f"📍 Selected City: {city}")

    # -----------------------------
    # 📡 Load Data
    # -----------------------------
    energy_df = load_energy_data()
    weather_df = get_weather(city)

    df = energy_df.copy()

    # -----------------------------
    # ⚙️ Feature Engineering
    # -----------------------------
    df = create_features(df)

    # -----------------------------
    # 📊 Decomposition
    # -----------------------------
    df = decompose(df)

    # -----------------------------
    # 🚨 Anomaly Detection (AUTOENCODER ONLY)
    # -----------------------------
    df = autoencoder_anomaly(df)

    # -----------------------------
    # 📈 Energy Consumption Plot
    # -----------------------------
    st.subheader("📈 Energy Consumption")
    st.line_chart(df["energy"])

    # -----------------------------
    # 📉 Decomposition Plot
    # -----------------------------
    st.subheader("📉 Decomposition")
    st.line_chart(df[["trend", "seasonal", "residual"]])

    # -----------------------------
    # 🚨 Anomalies Display
    # -----------------------------
    st.subheader("🚨 Detected Anomalies (Autoencoder)")
    anomalies = df[df["anomaly"] == True]
    st.write(anomalies.tail(20))

    # -----------------------------
    # 🔮 Forecast Section
    # -----------------------------
    st.subheader("🔮 Forecast Results")

    if model_name == "Prophet":
        forecast = prophet_model(df)
        forecast_df = forecast.set_index("ds")
        st.line_chart(forecast_df["yhat"])

    elif model_name == "ARIMA":
        forecast = arima_model(df)
        forecast_df = pd.DataFrame(forecast, columns=["Forecast"])
        st.line_chart(forecast_df)

    elif model_name == "SARIMA":
        forecast = sarima_model(df)
        forecast_df = pd.DataFrame(forecast, columns=["Forecast"])
        st.line_chart(forecast_df)

    elif model_name == "Random Forest":
        preds = random_forest_model(df)
        pred_df = pd.DataFrame(preds, index=df.index, columns=["Prediction"])
        st.line_chart(pred_df)

    elif model_name == "XGBoost":
        preds = xgboost_model(df)
        pred_df = pd.DataFrame(preds, index=df.index, columns=["Prediction"])
        st.line_chart(pred_df)

    # -----------------------------
    # 🧠 Autoencoder Insight
    # -----------------------------
    st.subheader("🧠 Reconstruction Error")
    st.line_chart(df["reconstruction_error"])

    # -----------------------------
    # 📊 Summary Insights
    # -----------------------------
    st.subheader("📊 Summary Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Energy", round(df["energy"].mean(), 2))
    col2.metric("Max Energy", round(df["energy"].max(), 2))
    col3.metric("Min Energy", round(df["energy"].min(), 2))