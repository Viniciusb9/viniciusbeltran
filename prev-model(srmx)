import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Carregando os dados
df = pd.read_excel(r'C:\\Users\\mia.visantos\\Downloads\\Projetoprev\\CLEXP - 1820.xlsx')
df2 = pd.read_excel(r'C:\\Users\\mia.visantos\\Downloads\\Projetoprev\\CLEXP - 2123.xlsx')  
df3 = pd.concat([df, df2], ignore_index=True)

# Convertendo a coluna de data para datetime
df3['DATE'] = pd.to_datetime(df3['DATE'])

# Verificando duplicatas no índice
duplicates = df3[df3.duplicated(subset='DATE', keep=False)]
print(f"Duplicated dates:\n{duplicates}")

# Removendo ou agregando duplicatas
# Aqui, somamos os valores de TEUS para datas duplicadas
df3 = df3.groupby('DATE').agg({'TEUS': 'sum'}).reset_index()

# Configurando a coluna 'DATE' como índice e configurando a frequência
df3.set_index('DATE', inplace=True)
df3 = df3.asfreq('ME')

# Preenchendo valores faltantes se houver
df3['TEUS'] = df3['TEUS'].fillna(method='ffill')

# Plotando os dados ao longo do tempo
plt.figure(figsize=(14,10))
plt.plot(df3.index, df3['TEUS'])
plt.title('TEUS over time')
plt.ylabel('TEUs')
plt.xlabel('Date')
plt.show()

# Decomposição sazonal para verificar tendências
result = seasonal_decompose(df3['TEUS'], model='additive')
result.plot()
plt.show()

# Definindo os conjuntos de treino (até 2021) e teste (2022 em diante)
train_mod = df3[df3.index < '2022-01-01']
test_mod = df3[df3.index >= '2022-01-01']

# Ajustando o modelo SARIMAX
modelf = SARIMAX(train_mod['TEUS'],
                 order=(1, 1, 1),
                 seasonal_order=(1, 1, 1, 12),
                 enforce_stationarity=False,
                 enforce_invertibility=False)

# Ajustando o modelo
sr_result = modelf.fit(disp=False)

# Realizando previsões para o conjunto de teste
Prev = sr_result.get_forecast(steps=len(test_mod))
Prev_values = Prev.predicted_mean
Prev_conf = Prev.conf_int()

# Métricas de erro
mse = mean_squared_error(test_mod['TEUS'], Prev_values)
mae = mean_absolute_error(test_mod['TEUS'], Prev_values)

print(f'Mean squared Error: {mse:.2f}')
print(f'Mean absolute Error: {mae:.2f}')

# Plotando previsões vs valores reais
plt.figure(figsize=(14,10))
plt.plot(train_mod.index, train_mod['TEUS'], label='Train')
plt.plot(test_mod.index, test_mod['TEUS'], label='Test')
plt.plot(test_mod.index, Prev_values, label='Forecast')
plt.fill_between(test_mod.index, 
                 Prev_conf.iloc[:, 0], 
                 Prev_conf.iloc[:, 1], color='k', alpha=0.2)
plt.title('Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('TEUS')
plt.legend()
plt.show()

# Realizando previsões para períodos futuros até o final de 2025
future_steps = (2025 - 2021) * 12  # Número de meses de 2022 até o final de 2025
future_forecast = sr_result.get_forecast(steps=future_steps)
future_values = future_forecast.predicted_mean
future_conf = future_forecast.conf_int()

# Plotando previsões futuras
plt.figure(figsize=(14,10))
plt.plot(df3.index, df3['TEUS'], label='Historical Data')
plt.plot(future_values.index, future_values, label='Forecast')
plt.fill_between(future_values.index, 
                 future_conf.iloc[:, 0], 
                 future_conf.iloc[:, 1], color='k', alpha=0.2)
plt.title('TEUS Forecast')
plt.xlabel('Date')
plt.ylabel('TEUS')
plt.legend()
plt.show()


# Exportando previsões para Excel
forecast_df = pd.DataFrame({
    'Date': future_values.index,
    'Forecasted TEUs': future_values,
    'Lower_CI': future_conf.iloc[:,0],
    'Upper_CI': future_conf.iloc[:,1]
})

forecast_df.to_excel('Forecasted_TEUS_CLEXP.xlsx', index=False)
