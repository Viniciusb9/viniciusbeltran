import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_excel(r'C:\\Users\\mia.visantos\\Downloads\\Projetoprev\\melon.xlsx')

# Converter a coluna de data para datetime e definir como índice
df["DATE"] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Resample para dados mensais
df_mensal = df.resample('ME').sum()

# Função para criar features
def criar_features(df):
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    for lag in range(1, 13):
        df[f'lag_{lag}'] = df['TEUS'].shift(lag)
    
    ## excluir valores nulos
    df.dropna(inplace=True)
    return df


# Aplicar a função de criação de features
df_mensal = criar_features(df_mensal)

# Definir X e y
x = df_mensal.drop(columns=['TEUS'])
y = df_mensal['TEUS']

# Dividir os dados de treino e teste
train = df_mensal[df_mensal.index < '2023-09-30']
test = df_mensal[df_mensal.index >= '2023-09-30']

x_train = train.drop('TEUS', axis=1)
y_train = train['TEUS']
x_test = test.drop('TEUS', axis=1)
y_test = test['TEUS']

# Construir o modelo
modelo_rf = RandomForestRegressor(n_estimators=150, bootstrap=True, random_state=42)
modelo_rf.fit(x_train, y_train)

# Fazer previsões
y_pred = modelo_rf.predict(x_test)



# calculating the model perf
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Calculado manualmente
r2 = metrics.r2_score(y_test, y_pred)

print(f"MSE = {mse}")
print(f"MAE = {mae}")
print(f"RMSE = {rmse}")
print(f"MAPE = {mape}")
print(f"R2 = {r2}")


## Defining future preds

# Definir número de passos futuros para prever
future_steps = 24  # 24 meses à frente
last_date = df_mensal.index[-1]  # última data disponível no conjunto de dados
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='ME')

# Últimos valores conhecidos (linha mais recente do dataframe) - servirá como base para previsões futuras
last_values = df_mensal.iloc[-1].copy()

# Lista para armazenar previsões futuras
future_preds = []

# Iterar sobre as futuras datas
for date in future_dates:
    # Atualizar 'Month' e 'Year' para o novo mês da previsão
    last_values['Month'] = date.month
    last_values['Year'] = date.year
    
    # Atualizar as features de defasagem (lags)
    for lag in range(1, 13):
        if lag == 1:
            # 'lag_1' será o último valor de 'TEUS' (seja do histórico ou da última previsão)
            last_values[f'lag_{lag}'] = last_values['TEUS']
        else:
            # Preencher os lags subsequentes com previsões anteriores se disponíveis, ou valores históricos
            if len(future_preds) >= lag - 1:
                last_values[f'lag_{lag}'] = future_preds[-(lag - 1)]
            else:
                last_values[f'lag_{lag}'] = df_mensal.iloc[-lag]['TEUS']
    
    # Criar o conjunto de features (excluindo 'TEUS') para a previsão futura
    future_features = last_values.drop('TEUS')
    
    # Fazer a previsão com o modelo treinado
    predicted_teus = modelo_rf.predict(future_features.values.reshape(1, -1))[0]
    
    # Armazenar a previsão para usar em previsões futuras
    future_preds.append(predicted_teus)
    
    # Atualizar 'TEUS' no last_values com a previsão atual
    last_values['TEUS'] = predicted_teus

# Criar um DataFrame com as previsões futuras
df_future_preds = pd.DataFrame({
    'DATE': future_dates,
    'Predicted_TEUS': future_preds
}).set_index('DATE')

# Exibir previsões futuras
print(df_future_preds)


tab_for_melon = pd.DataFrame({
    'Date': df_future_preds.index,
    'Forecast TEUs': df_future_preds['Predicted_TEUS']  # Corrected to extract the TEUS column as a series
})

tab_for_melon.to_excel('tab_for_melon.xlsx', index=False)

