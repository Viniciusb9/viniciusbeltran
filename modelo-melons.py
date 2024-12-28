# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller

# Carregar os dados
df=pd.read_excel("C:\\Users\\mia.visantos\\Downloads\\Projetoprev\\melon.xlsx")



df["DATE"]=pd.to_datetime(df['DATE']).dt.date
df.set_index('DATE', inplace=True)
df.index = pd.to_datetime(df.index)

out_mean = df.resample('MS').mean()
out_sum= df.resample('MS').sum()

tipo=int(input("Digte 1 para media e 2 para soma: "))
if tipo==1:
   out=out_mean
else:
   out=out_sum 
out["month"] = [d.strftime("%b") for d in out.index] # %b = abbreviated month name (Jan, Fev, ..., Dec)
plt.figure(figsize=(15,6))
sns.boxplot(x = "month", y = "TEUS", data = out).set_title("Multi Month-wise Box Plot")
plt.show()

# performing Seasonal Decompose
from statsmodels.tsa.seasonal import seasonal_decompose

result_mul = seasonal_decompose(out["TEUS"], model = "additive", extrapolate_trend = "freq")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10, 6), sharex = True, sharey = False, dpi = 100)
ax1.plot(out["TEUS"])
ax1.set_ylabel("TEUS")
ax2.plot(result_mul.trend)
ax2.set_ylabel("TEUS_trend")
ax3.plot(result_mul.seasonal)
ax3.set_ylabel("TEUS_seasonal")
ax4.plot(result_mul.resid)
ax4.set_ylabel("TEUS_residual")
fig.suptitle("TEUS Time Series Decomposition")
plt.show()

# performing Augmented Dickey-Fuller (ADF) Test

print("Results of the Augmented Dickey-Fuller Test\n")
adf_test = adfuller(out["TEUS"], autolag = "AIC") # AIC is the "Akaike Information Criterion"
adf_output = pd.Series(adf_test[0:4], index = ["test statistic:", "p-value:", "number of lags used:", "number of observations used:"])
for key, value in adf_test[4].items():
    adf_output[f"critical value for the test statistic at the {key} level:"] = value
print(adf_output)
if adf_test[1] > 0.05:
    print("\nConclusion: Fail to reject the null hypothesis. Data is non-stationary.")
else:
    print("\nConclusion: Reject the null hypothesis. Data is stationary.")

X = out[["TEUS"]]
train, test = X.loc[:'2024-07-31'], X['2023-07-31':]

# performing the training
from pmdarima import auto_arima

stepwise_model = auto_arima(train, start_p = 1, max_p = 7, d = None, start_q = 1, max_q = 7, seasonal = True, start_P = 1, max_P = 7, D = None, max_D = 7, start_Q = 1, max_Q = 7, m = 12, trace = True, error_action = "ignore", suppress_warnings = True, stepwise = True)
stepwise_model.summary()

# performing forecast for 36 months (3 years) + additional 24 months for 2024 and 2025
forecast_periods = 36 + 24  # Update this to extend the forecast period
forecast, conf_int = stepwise_model.predict(n_periods = forecast_periods, return_conf_int = True)

# forecast
forecast = pd.DataFrame(forecast, columns = ["TEUS"])
forecast["new_index"] = pd.date_range(start=test.index[0], periods=forecast_periods, freq='MS')
forecast = forecast.set_index("new_index")

# confidence bound
df_conf = pd.DataFrame(conf_int, columns = ["Lower_bound", "Upper_bound"])
df_conf["new_index"] = pd.date_range(start=test.index[0], periods=forecast_periods, freq='MS')
df_conf = df_conf.set_index("new_index")

# Plotting the forecast along with the train and test data
plt.figure(figsize = (15, 6))
plt.plot(train, label = "Train")
plt.plot(test, label = "Test")
plt.plot(forecast, label = "Predicted")
plt.fill_between(df_conf.index, df_conf["Lower_bound"], df_conf["Upper_bound"], color='k', alpha=0.1, label="Confidence Interval")
plt.xlabel("Date")
plt.ylabel("TEUS")
plt.legend(loc = "best")
plt.title("TEUS Time Series Forecast")
plt.show()

# Plotting a zoomed-in view for the forecast period
plt.figure(figsize = (15, 6))
plt.plot(test, label = "Test", color = "orange")
plt.plot(forecast, label = "Predicted", color = "green")
plt.fill_between(df_conf.index, df_conf["Lower_bound"], df_conf["Upper_bound"], color='k', alpha=0.1, label="Confidence Interval")
plt.xlabel("Date")
plt.ylabel("TEUS")
plt.legend(loc = "best")
plt.title("TEUS Time Series Forecast (Zoomed-In)")
plt.show()

# Plotting diagnostic plots
stepwise_model.plot_diagnostics(figsize = (15, 6))
plt.show()

# performing evaluation of some metrics (test versus forecast)
import numpy as np
from sklearn import metrics

def mean_absolute_percentage_error(X_true, X_pred):
    X_true, X_pred = np.array(X_true), np.array(X_pred)
    epsilon = np.finfo(np.float64).eps # epsilon = 2.220446049250313e-16
    mape = np.mean(np.abs(X_true - X_pred) / np.maximum(np.abs(X_true), epsilon)) * 100
    return mape


print("Evaluation Metric Results\n")
print(f"MSE = {metrics.mean_squared_error(test, forecast.loc[test.index])}")
print(f"MAE = {metrics.mean_absolute_error(test, forecast.loc[test.index])}")
print(f"RMSE = {np.sqrt(metrics.mean_squared_error(test, forecast.loc[test.index]))}")
print(f"MAPE = {mean_absolute_percentage_error(test, forecast.loc[test.index])}")
print(f"R2 = {metrics.r2_score(test, forecast.loc[test.index])}")

# Save forecast and confidence interval to Excel files
forecast.to_excel('forecast_pred.xlsx', index=True)
df_conf.to_excel('forecast_intconf.xlsx', index=True)

# Save forecasted values to another Excel file
tab_for_melons = pd.DataFrame({
    'Date': forecast.index,
    'Forecast TEUs': forecast['TEUS']  # Corrected to extract the TEUS column as a series
})

tab_for_melons.to_excel('tab_for_melons.xlsx', index=False)
