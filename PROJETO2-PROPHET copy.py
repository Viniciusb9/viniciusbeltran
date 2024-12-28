from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar os dados
df = pd.read_excel(r'C:\\Users\\mia.visantos\\Downloads\\Projetoprev\\melon.xlsx')

# Converter a coluna de data para o formato datetime e renomear para ds e y (formato exigido pelo Prophet)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.rename(columns={'DATE': 'ds', 'TEUS': 'y'})

# Dividir os dados em treino e teste
train_size = int(len(df) * 0.8)  # Usar 80% para treino e 20% para teste
train_df = df[:train_size]
test_df = df[train_size:]

# Inicializar e ajustar o modelo Prophet com os dados de treino
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(train_df)

# Criar um DataFrame para previsões futuras (inclui datas do teste)
df_future = model.make_future_dataframe(periods=len(test_df), freq='M')

# Fazer previsões
df_forecast = model.predict(df_future)

# Plotar os componentes (tendência, sazonalidade)
model.plot_components(df_forecast)
plt.show()

# Plotar as previsões com os dados históricos
model.plot(df_forecast)
plt.show()

# Extraindo as previsões para o período de teste
forecast_test = df_forecast.set_index('ds').loc[test_df['ds']]

# Avaliar o modelo comparando previsões com valores reais no conjunto de teste
y_true = test_df['y'].values
y_pred = forecast_test['yhat'].values

# Cálculo das métricas de erro
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = mse ** 0.5
mape = (abs(y_true - y_pred) / y_true).mean() * 100

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")

# Plotar os valores reais vs preditos
plt.figure(figsize=(10, 6))
plt.plot(test_df['ds'], y_true, label='Valor Real', marker='o')
plt.plot(test_df['ds'], y_pred, label='Valor Predito Prophet', marker='x')
plt.title('Predição Prophet vs Valor Real (Conjunto de Teste)')
plt.xlabel('Data')
plt.ylabel('TEUS')
plt.legend()
plt.grid(True)
plt.show()
