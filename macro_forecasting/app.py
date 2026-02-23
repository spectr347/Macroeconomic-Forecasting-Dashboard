import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_world_bank_data
from src.transformations import transform_series
from src.stationarity import adf_test
from src.models import select_arima_model, forecast_arima
from src.diagnostics import ljung_box_test
from src.evaluation import train_test_split_ts, calculate_rmse

st.set_page_config(page_title="Macroeconomic Forecasting Tool", layout="wide")

st.title("📈 Macroeconomic Forecasting Dashboard")

# Sidebar
st.sidebar.header("Model Configuration")

country = st.sidebar.text_input("Country Code (ISO-3)", "USA")
indicator = st.sidebar.text_input("Indicator Code", "NY.GDP.MKTP.CD")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years)", 1, 20, 5)
log_transform = st.sidebar.checkbox("Apply Log Transformation")

# Load data
df = load_world_bank_data(country, indicator)

if df is None:
    st.error("Unable to load data. Check country or indicator code.")
else:

    st.subheader("Historical Series")
    st.line_chart(df)

    # Transform
    series = transform_series(df["value"], log_transform)

    # ADF Test
    st.subheader("Stationarity Test (ADF)")
    adf_result = adf_test(series)
    st.write(adf_result)

    # Train/Test Split
    train, test = train_test_split_ts(series, forecast_horizon)

    # Model Selection
    st.subheader("ARIMA Model Selection (AIC)")
    model = select_arima_model(train)

    st.write(f"Selected ARIMA Order: {model.model.order}")
    st.write(f"AIC: {model.aic:.2f}")

    # Forecast
    forecast_df = forecast_arima(model, train, forecast_horizon)

    # Plot Forecast
    st.subheader("Forecast")
    fig, ax = plt.subplots()
    ax.plot(series.index, series, label="Historical")
    ax.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")
    ax.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        alpha=0.3
    )
    ax.legend()
    st.pyplot(fig)

    # Evaluation
    st.subheader("Out-of-Sample Evaluation")
    rmse = calculate_rmse(test, forecast_df["forecast"])
    st.write(f"RMSE: {rmse:.4f}")

    # Residual Diagnostics
    st.subheader("Residual Diagnostics (Ljung-Box)")
    lb_result = ljung_box_test(model.resid)
    st.write(lb_result)