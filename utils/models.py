from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


# -----------------------------
# 🔮 PROPHET MODEL
# -----------------------------
def prophet_model(df):
    data = df.reset_index()[["datetime", "energy"]]
    data.columns = ["ds", "y"]

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=24, freq="H")
    forecast = model.predict(future)

    return forecast


# -----------------------------
# 🌲 RANDOM FOREST MODEL
# -----------------------------
def random_forest_model(df):
    features = ["hour", "dayofweek", "lag1", "lag2"]
    
    X = df[features]
    y = df["energy"]

    model = RandomForestRegressor()
    model.fit(X, y)

    preds = model.predict(X)
    return preds


# -----------------------------
# ⚡ XGBOOST MODEL
# -----------------------------
def xgboost_model(df):
    features = ["hour", "dayofweek", "lag1", "lag2"]
    
    X = df[features]
    y = df["energy"]

    model = XGBRegressor()
    model.fit(X, y)

    preds = model.predict(X)
    return preds


# -----------------------------
# 📉 ARIMA MODEL
# -----------------------------
def arima_model(df):
    series = df["energy"]

    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=24)

    forecast_df = pd.DataFrame({
        "Forecast": forecast
    })

    return forecast_df


# -----------------------------
# 📊 SARIMA MODEL
# -----------------------------
def sarima_model(df):
    series = df["energy"]

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24)
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=24)

    forecast_df = pd.DataFrame({
        "Forecast": forecast
    })

    return forecast_df