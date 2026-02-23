from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box_test(residuals, lags=10):
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)

    return {
        "Ljung-Box Statistic": result["lb_stat"].values[0],
        "p-value": result["lb_pvalue"].values[0]
    }