import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def select_arima_model(series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_model = None

    for p in range(max_p):
        for d in range(max_d):
            for q in range(max_q):
                try:
                    model = ARIMA(series, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                except:
                    continue

    return best_model


def forecast_arima(model, train_series, horizon):
    forecast = model.get_forecast(steps=horizon)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    last_date = train_series.index[-1]

    forecast_index = pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq="YS"
    )[1:]

    return pd.DataFrame({
        "forecast": forecast_mean.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values
    }, index=forecast_index)