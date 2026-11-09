import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Advanced Big Market Forecast Dashboard", layout="wide")
st.title("📊 Advanced Big Market Forecast Dashboard")

# =========================
# USER INPUTS
# =========================
API_KEY = st.sidebar.text_input("6MBRUX470ELHFOXU", "")
FORECAST_DAYS = st.sidebar.slider("Forecast Days", 1, 30, 7)
SELECT_PRODUCTS = st.sidebar.multiselect(
    "Select Products / Markets",
    ["Apple", "Microsoft", "Google", "Amazon", "Tesla"],
    default=["Apple", "Microsoft"]
)
SELECT_MODELS = st.sidebar.multiselect(
    "Select Models to Compare",
    ["Random Forest", "XGBoost", "ARIMA", "LSTM"],
    default=["Random Forest", "XGBoost", "ARIMA", "LSTM"]
)

PRODUCT_SYMBOLS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA"
}

if not API_KEY:
    st.warning("Please enter your Alpha Vantage API Key.")
    st.stop()

# =========================
# FUNCTION: FETCH LIVE DATA
# =========================
@st.cache_data(ttl=60)
def fetch_live_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series (Daily)" not in data:
        return None, data
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df[["4. close"]]
    df.rename(columns={"4. close": "close"}, inplace=True)
    return df, None

# =========================
# PROCESS MULTIPLE PRODUCTS
# =========================
for product in SELECT_PRODUCTS:
    st.subheader(f"📈 Historical Data & Forecast — {product}")
    df, error = fetch_live_data(PRODUCT_SYMBOLS[product])
    if df is None:
        st.error(f"Failed to fetch data for {product}")
        st.write(error)
        continue

    # =========================
    # FEATURE ENGINEERING
    # =========================
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["rolling_3"] = df["close"].rolling(window=3).mean()
    df["rolling_7"] = df["close"].rolling(window=7).mean()
    df["lag_1"] = df["close"].shift(1)
    df = df.fillna(method='bfill')

    feature_cols = ["day", "month", "year", "rolling_3", "rolling_7", "lag_1"]
    X = df[feature_cols]
    y = df["close"]

    # Future dates
    future_dates = pd.date_range(df.index[-1], periods=FORECAST_DAYS+1, freq="D")[1:]
    future_df = pd.DataFrame({
        "day": future_dates.day,
        "month": future_dates.month,
        "year": future_dates.year
    })
    # Use last rolling & lag for future features
    last_values = df[["rolling_3","rolling_7","lag_1"]].iloc[-1].values
    future_df["rolling_3"] = last_values[0]
    future_df["rolling_7"] = last_values[1]
    future_df["lag_1"] = last_values[2]

    forecast_results = {}
    metrics_results = {}

    # =========================
    # RANDOM FOREST
    # =========================
    if "Random Forest" in SELECT_MODELS:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        preds = rf.predict(future_df[feature_cols])
        forecast_results["Random Forest"] = preds
        metrics_results["Random Forest"] = {
            "RMSE": np.sqrt(mean_squared_error(y, rf.predict(X))),
            "MAE": mean_absolute_error(y, rf.predict(X))
        }

    # =========================
    # XGBOOST
    # =========================
    if "XGBoost" in SELECT_MODELS:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(X, y)
        preds = xgb.predict(future_df[feature_cols])
        forecast_results["XGBoost"] = preds
        metrics_results["XGBoost"] = {
            "RMSE": np.sqrt(mean_squared_error(y, xgb.predict(X))),
            "MAE": mean_absolute_error(y, xgb.predict(X))
        }

    # =========================
    # ARIMA
    # =========================
    if "ARIMA" in SELECT_MODELS:
        ts = df['close']
        try:
            arima_model = ARIMA(ts, order=(1,1,1))
            arima_fit = arima_model.fit()
            preds = arima_fit.forecast(steps=FORECAST_DAYS)
            forecast_results["ARIMA"] = preds
            metrics_results["ARIMA"] = {
                "RMSE": np.sqrt(mean_squared_error(y, arima_fit.fittedvalues)),
                "MAE": mean_absolute_error(y, arima_fit.fittedvalues)
            }
        except:
            st.warning("ARIMA failed: data may be too short")

    # =========================
    # LSTM
    # =========================
    if "LSTM" in SELECT_MODELS:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df[['close']].values)
        X_train, y_train = [], []
        seq_len = 5
        for i in range(seq_len, len(scaled_data)):
            X_train.append(scaled_data[i-seq_len:i,0])
            y_train.append(scaled_data[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        # Robust LSTM forecast loop
        inputs = scaled_data[-seq_len:].reshape(1, seq_len, 1)
        lstm_forecast = []

        for _ in range(FORECAST_DAYS):
            pred = lstm_model.predict(inputs, verbose=0)   # shape (1,1)
            lstm_forecast.append(pred[0,0])
            pred_reshaped = np.expand_dims(pred, axis=2)   # shape (1,1,1)
            inputs = np.concatenate((inputs[:, 1:, :], pred_reshaped), axis=1)

        lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1,1)).flatten()
        forecast_results["LSTM"] = lstm_forecast

        # Metrics for LSTM
        lstm_pred_full = lstm_model.predict(X_train, verbose=0)
        lstm_pred_full = scaler.inverse_transform(lstm_pred_full)
        metrics_results["LSTM"] = {
            "RMSE": np.sqrt(mean_squared_error(df['close'][seq_len:], lstm_pred_full.flatten())),
            "MAE": mean_absolute_error(df['close'][seq_len:], lstm_pred_full.flatten())
        }

    # =========================
    # PLOT RESULTS
    # =========================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Historical", line=dict(width=2)))
    for name, preds in forecast_results.items():
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=name))
    fig.update_layout(title=f"{product} Forecast Comparison",
                      xaxis_title="Date", yaxis_title="Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # FORECAST TABLE
    # =========================
    st.subheader(f"🔮 Forecast Table — {product}")
    forecast_table = pd.DataFrame(forecast_results, index=future_dates)
    st.dataframe(forecast_table)

    # =========================
    # METRICS TABLE
    # =========================
    st.subheader(f"📊 Model Metrics — {product}")
    metrics_df = pd.DataFrame(metrics_results).T
    st.dataframe(metrics_df)
