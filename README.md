# ⚡ Energy Forecast Intelligence

🔗 Live Demo: https://energy-analytics-platform.streamlit.app/
**A production-grade **time series forecasting and anomaly detection system** for energy consumption using **weather-driven signals** and multiple forecasting models.

---

## 🚀 Overview

**Energy Forecast Intelligence** predicts energy consumption for major Indian cities using:

* 🌦 Live weather data (Open-Meteo API)
* 📊 Time series decomposition (trend, seasonality, residual)
* 🔮 Multiple forecasting models (Statistical + Machine Learning)
* 🚨 Deep learning-based anomaly detection (Autoencoder)
* 🎛 Interactive dashboard using Streamlit

---

## 🌆 Supported Cities

* Mumbai
* Delhi
* Bangalore
* Kolkata
* Chennai

---

## 🧠 Key Features

* 📡 Real-time weather data integration
* ⚡ Synthetic yet realistic energy consumption simulation
* 📊 STL decomposition for time series analysis
* 🔮 Forecasting models:

  * Prophet
  * ARIMA
  * SARIMA
  * Random Forest
  * XGBoost
* 🚨 Anomaly detection:

  * Autoencoder (Deep Learning)
* 🎛 Dynamic UI:

  * City selection
  * Model type selection
  * Model selection dropdown
* 📊 Insight metrics (min, max, average energy)

---

## 🏗 Project Structure

```bash
energy_forecasting_app/
│
├── app.py
├── requirements.txt
│
└── utils/
    ├── data_loader.py
    ├── weather.py
    ├── preprocessing.py
    ├── decomposition.py
    ├── models.py
    ├── anomaly.py
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd energy_forecasting_app
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Select an Indian city
2. Choose model type:

   * Statistical (Prophet, ARIMA, SARIMA)
   * Machine Learning (Random Forest, XGBoost)
3. Run forecasting
4. Visualize:

   * Energy consumption trends
   * Time series decomposition
   * Forecast results
   * Detected anomalies

---

## 🔮 Models Used

| Type             | Models                 |
| ---------------- | ---------------------- |
| Statistical      | Prophet, ARIMA, SARIMA |
| Machine Learning | Random Forest, XGBoost |
| Deep Learning    | Autoencoder            |

---

## 🚨 Anomaly Detection (Autoencoder)

The Autoencoder detects anomalies by:

* Learning normal energy usage patterns
* Reconstructing input signals
* Identifying anomalies using **reconstruction error thresholds**

---

## 📈 Future Enhancements

* LSTM / Transformer-based forecasting
* Real energy dataset integration (India grid data)
* Model performance comparison (RMSE, MAE dashboard)
* FastAPI backend for production deployment
* Real-time streaming pipeline

---

## 🏆 Use Cases

* Smart Grid Optimization
* Energy Demand Forecasting
* Load Balancing
* Utility Analytics

---

## 📜 License

MIT License
