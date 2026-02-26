# 📈 Macroeconomic Forecasting Tool  
### Applied Time-Series Econometrics with ARIMA Modelling

---

## 1. Project Overview

This project implements a fully modular macroeconomic forecasting system using ARIMA-based time-series modelling applied to real-world data retrieved programmatically from the World Bank API.

The objective is not simply to produce forecasts, but to replicate a professional econometric workflow:

- Data acquisition  
- Stationarity diagnostics  
- Transformation and differencing  
- Information-criterion-based model selection  
- Out-of-sample evaluation  
- Residual independence testing  
- Forecast uncertainty quantification  

The system is delivered through an interactive Streamlit dashboard to demonstrate practical deployment of econometric models.

---

## 2. Data Source

Macroeconomic indicators are retrieved dynamically from the World Bank Open Data API.

Example indicator:

- **NY.GDP.MKTP.CD** — GDP (current US dollars)

The dataset consists of annual observations beginning in 1960.

The time series is indexed, cleaned, and frequency-aligned prior to modelling.

---

## 3. Econometric Methodology

---

### 3.1 Stationarity and Unit Root Testing

Time-series modelling requires determining whether a process is stationary.

We formally test for a unit root using the **Augmented Dickey–Fuller (ADF) test**.

#### Hypotheses

H₀: The series contains a unit root (non-stationary)  
H₁: The series is stationary  

If the null hypothesis cannot be rejected (p-value > 0.05), the series is treated as non-stationary and differencing is applied.

#### Macroeconomic Context

Nominal GDP in levels typically exhibits:

- Deterministic trend  
- Stochastic trend  
- Inflation-driven drift  

Thus, it is commonly integrated of order one, I(1).

An optional log transformation is applied:

\[
y_t = \log(GDP_t)
\]

This stabilizes variance and converts multiplicative growth into additive form.

---

### 3.2 ARIMA Model Specification

The general ARIMA model is defined as:

\[
\phi(L)(1 - L)^d y_t = \theta(L)\varepsilon_t
\]

Where:

- \( L \) = lag operator  
- \( d \) = order of differencing  
- \( \phi(L) \) = autoregressive polynomial  
- \( \theta(L) \) = moving average polynomial  
- \( \varepsilon_t \sim WN(0, \sigma^2) \)

An ARIMA(p,d,q) model consists of:

- \( p \) autoregressive terms  
- \( d \) differences  
- \( q \) moving average terms  

---

### 3.3 Model Selection via AIC

Model selection is performed through grid search over candidate (p,d,q) combinations.

The selection criterion is the Akaike Information Criterion (AIC):

\[
AIC = 2k - 2\ln(L)
\]

Where:

- \( k \) = number of parameters  
- \( L \) = maximized likelihood  

The model minimizing AIC balances goodness-of-fit with model complexity.

---

### 3.4 Forecasting

Forecasts are generated recursively using the fitted ARIMA model.

Outputs include:

- Point forecasts  
- 95% confidence intervals  

Confidence intervals account for forecast error variance accumulation over the forecast horizon.

---

### 3.5 Out-of-Sample Evaluation

A rolling train/test split is implemented:

- Training sample: all observations except final \( h \)  
- Test sample: final \( h \) observations  

Predictive accuracy is evaluated using Root Mean Squared Error (RMSE):

\[
RMSE = \sqrt{\frac{1}{n} \sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
\]

When modelling log-transformed GDP, RMSE approximates percentage error.

---

### 3.6 Residual Diagnostics

Residual independence is tested using the Ljung–Box test:

\[
H_0: \text{Residuals are independently distributed}
\]

A high p-value indicates:

- No remaining autocorrelation  
- Model is well specified  

This ensures the ARIMA structure captures serial dependence adequately.

---

## 4. Empirical Findings (Example: US GDP)

For nominal US GDP:

- ADF test confirms non-stationarity in levels  
- Log transformation improves model fit  
- ARIMA(1,1,1) or ARIMA(1,1,2) typically selected  
- Log specification produces lower AIC  
- Residuals pass Ljung–Box independence tests  
- RMSE corresponds to economically plausible percentage forecast error  

This aligns with established macroeconomic time-series literature.

---

## 5. Why Log Transformation Matters

Without log transformation:

- Variance grows with level  
- Forecast errors measured in trillions  
- Heteroskedasticity may distort inference  

With log transformation:

- Variance stabilizes  
- Growth interpretation becomes linear  
- Forecast accuracy becomes interpretable in percentage terms  
- Model diagnostics improve  

This mirrors professional macroeconomic modelling standards.

---

## 6. System Architecture

The application follows modular separation of concerns:

```text
macro_forecasting/
├── app.py                # Dashboard interface
├── src/
│   ├── data_loader.py    # API integration
│   ├── transformations.py
│   ├── stationarity.py
│   ├── models.py         # ARIMA estimation & selection
│   ├── diagnostics.py    # Residual testing
│   ├── evaluation.py     # Forecast accuracy
