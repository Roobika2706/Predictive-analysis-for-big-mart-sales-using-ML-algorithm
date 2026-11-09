from flask import Flask, render_template, request
from flask_caching import Cache
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.io import to_html
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

app = Flask(__name__)

# ── Caching setup (very important for free tier 25/day limit) ─────────────────
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',          # In-memory cache (good for dev)
    'CACHE_DEFAULT_TIMEOUT': 3600         # 1 hour
})

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ALPHA_VANTAGE_KEY")

PRODUCT_SYMBOLS = {
    "Apple":    "AAPL",
    "Microsoft": "MSFT",
    "Google":    "GOOGL",
    "Amazon":    "AMZN",
    "Tesla":     "TSLA"
}

ALL_PRODUCTS = list(PRODUCT_SYMBOLS.keys())
ALL_MODELS   = ["Random Forest", "XGBoost", "ARIMA", "LSTM"]


@cache.memoize(timeout=3600)  # Cache per symbol for 1 hour → saves API calls
def fetch_live_data(symbol):
    if not API_KEY:
        return None, "Alpha Vantage API key not set. Please set ALPHA_VANTAGE_KEY in .env or environment."

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if "Time Series (Daily)" not in data:
            msg = data.get("Note", data.get("Information", "No time series data returned"))
            if "rate limit" in msg.lower() or "premium" in msg.lower() or "25 requests" in msg:
                return None, (
                    "Alpha Vantage free tier daily limit (25 requests/day) reached. "
                    "Wait until tomorrow (resets ~midnight UTC) or get a new free key at "
                    "https://www.alphavantage.co/support/#api-key"
                )
            return None, msg

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[["4. close"]].rename(columns={"4. close": "close"})
        return df, None

    except Exception as e:
        return None, f"Fetch error: {str(e)}"


def generate_forecast(product, days, selected_models):
    symbol = PRODUCT_SYMBOLS.get(product)
    if not symbol:
        return None, None, None, f"Unknown product: {product}"

    df, err = fetch_live_data(symbol)
    if df is None:
        return None, None, None, err

    # Feature Engineering
    df["day"]   = df.index.day
    df["month"] = df.index.month
    df["year"]  = df.index.year
    df["rolling_3"] = df["close"].rolling(3).mean()
    df["rolling_7"] = df["close"].rolling(7).mean()
    df["lag_1"]     = df["close"].shift(1)
    df = df.fillna(method="bfill")

    features = ["day", "month", "year", "rolling_3", "rolling_7", "lag_1"]
    X = df[features]
    y = df["close"]

    # Future dates — business days
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="B")
    future = pd.DataFrame(index=future_dates)
    future["day"]   = future.index.day
    future["month"] = future.index.month
    future["year"]  = future.index.year
    future["rolling_3"] = df["rolling_3"].iloc[-1]
    future["rolling_7"] = df["rolling_7"].iloc[-1]
    future["lag_1"]     = df["close"].iloc[-1]

    forecasts = {}
    metrics   = {}

    # Random Forest
    if "Random Forest" in selected_models:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        p = rf.predict(future[features])
        forecasts["Random Forest"] = p
        metrics["Random Forest"] = {
            "RMSE": float(np.sqrt(mean_squared_error(y, rf.predict(X)))),
            "MAE":  float(mean_absolute_error(y, rf.predict(X)))
        }

    # XGBoost
    if "XGBoost" in selected_models:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb.fit(X, y)
        p = xgb.predict(future[features])
        forecasts["XGBoost"] = p
        metrics["XGBoost"] = {
            "RMSE": float(np.sqrt(mean_squared_error(y, xgb.predict(X)))),
            "MAE":  float(mean_absolute_error(y, xgb.predict(X)))
        }

    # ARIMA — with frequency fix
    if "ARIMA" in selected_models:
        try:
            ts = df["close"].copy()
            ts.index = pd.DatetimeIndex(ts.index)
            ts = ts.asfreq("B").interpolate(method="time")

            model = ARIMA(ts, order=(1,1,1))
            fit = model.fit()
            fc = fit.forecast(steps=days)
            forecasts["ARIMA"] = fc.values

            fitted = fit.fittedvalues.reindex(df.index, method="nearest")
            metrics["ARIMA"] = {
                "RMSE": float(np.sqrt(mean_squared_error(df["close"], fitted))),
                "MAE":  float(mean_absolute_error(df["close"], fitted))
            }
        except Exception as e:
            metrics["ARIMA"] = {"RMSE": np.nan, "MAE": np.nan, "note": str(e)}

    # LSTM — modern input
    if "LSTM" in selected_models:
        scaler = MinMaxScaler((0,1))
        scaled = scaler.fit_transform(df[["close"]])

        seq_len = 5
        X_lstm, y_lstm = [], []
        for i in range(seq_len, len(scaled)):
            X_lstm.append(scaled[i-seq_len:i, 0])
            y_lstm.append(scaled[i, 0])

        if len(X_lstm) < 10:
            return None, None, None, "Not enough data for LSTM"

        X_lstm = np.array(X_lstm).reshape(-1, seq_len, 1)
        y_lstm = np.array(y_lstm)

        model = Sequential()
        model.add(Input(shape=(seq_len, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0)

        inputs = scaled[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        for _ in range(days):
            p = model.predict(inputs, verbose=0)[0,0]
            preds.append(p)
            inputs = np.roll(inputs, -1, axis=1)
            inputs[0, -1, 0] = p

        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        forecasts["LSTM"] = preds

        train_pred = model.predict(X_lstm, verbose=0)
        train_pred = scaler.inverse_transform(train_pred).flatten()
        metrics["LSTM"] = {
            "RMSE": float(np.sqrt(mean_squared_error(df["close"][seq_len:], train_pred))),
            "MAE":  float(mean_absolute_error(df["close"][seq_len:], train_pred))
        }

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Historical", line=dict(width=2.2, color="#1f77b4")))

    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, values) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(x=future_dates, y=values, name=name, mode="lines",
                                 line=dict(color=colors[i % len(colors)], dash="dash")))

    fig.update_layout(
        title=f"{product} — Price Forecast Comparison",
        xaxis_title="Date", yaxis_title="Close Price (USD)",
        template="plotly_white", height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    plot_div = to_html(fig, full_html=False, include_plotlyjs="cdn")

    # Tables
    if forecasts:
        fc_df = pd.DataFrame(forecasts, index=future_dates.strftime("%Y-%m-%d"))
        forecast_html = fc_df.round(2).to_html(classes="table table-striped table-hover", border=0)
    else:
        forecast_html = "<p class='text-muted'>No forecast models succeeded.</p>"

    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        metrics_html = metrics_df.round(4).to_html(classes="table table-striped table-hover", border=0)
    else:
        metrics_html = "<p class='text-muted'>No metrics available.</p>"

    return plot_div, forecast_html, metrics_html, None


@app.route("/", methods=["GET", "POST"])
def dashboard():
    if not API_KEY:
        return render_template(
            "index.html",
            error="Alpha Vantage API key missing. Set ALPHA_VANTAGE_KEY in .env file.",
            products=ALL_PRODUCTS,
            models=ALL_MODELS
        )

    error = None
    results = []

    if request.method == "POST":
        try:
            days = int(request.form.get("days", 7))
            if not 1 <= days <= 60:
                raise ValueError("Forecast days should be between 1 and 60")

            products = request.form.getlist("products")
            models   = request.form.getlist("models")

            if not products:
                error = "Please select at least one product."
            if not models:
                error = "Please select at least one model."

            if not error:
                for prod in products:
                    plot, ftable, mtable, err = generate_forecast(prod, days, models)
                    if err:
                        error = err
                        break
                    results.append({
                        "product": prod,
                        "plot": plot,
                        "forecast_table": ftable,
                        "metrics_table": mtable
                    })

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template(
        "index.html",
        products=ALL_PRODUCTS,
        models=ALL_MODELS,
        results=results,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)