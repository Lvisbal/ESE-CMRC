import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_excel(r"E:/ESE Maternidad RC/Contratacion/Proyecciones de Consumo PYTHON/Datasets/Asist - Cirugia Minima Invasiva - 20241029.xlsx", header=0)
print(data.head())

# Convertir a tipo datetime y establecer 'Mes' como índice
data['Mes'] = pd.to_datetime(data['Mes'])
data.set_index('Mes', inplace=True)

# Convertir a miles
data['Valor Facturado'] = (data['Valor Facturado'] / 1000).round(0).astype(int)

# Calcular y almacenar los estadísticos descriptivos
estadisticos = data['Valor Facturado'].describe().round(0).astype(int)

# Imprimir los estadísticos descriptivos
print("Estadísticos Descriptivos de 'Valor Facturado':")
print(estadisticos)

# Extraer las estadísticas necesarias de los estadísticos descriptivos
stats = {
    'mean': estadisticos['mean'],
    '25%': estadisticos['25%'],
    '75%': estadisticos['75%'],
    'min': estadisticos['min'],
    'max': estadisticos['max']
}

# Configurar el tamaño de la figura
plt.figure(figsize=(6, 8))

# Subplot 1: Boxplot
plt.subplot(3, 1, 1)
plt.boxplot(data['Valor Facturado'], vert=False)
plt.title("Diagrama de Caja", fontsize=10)
plt.xlabel("Valor Facturado", fontsize=8)

# Subplot 2: Gráfico de Barras
plt.subplot(3, 1, 2)
plt.bar(estadisticos.index, estadisticos, color='skyblue')  # Usar estadísticos calculados
plt.title("Diagrama de Barras", fontsize=10)
plt.ylabel("Valor", fontsize=8)

# Subplot 3: Histograma
plt.subplot(3, 1, 3)
plt.hist(data['Valor Facturado'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histograma", fontsize=10)
plt.xlabel("Valor Facturado", fontsize=8)
plt.ylabel("Frecuencia", fontsize=8)

# Ajustar el espacio entre subgráficos
plt.tight_layout()
plt.show()

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Graficar la serie de tiempo
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Valor Facturado'], marker='o', color='steelblue', label='Valor Facturado')

# Ajustar una línea de tendencia (polinomio de grado 1)
z = np.polyfit(range(len(data)), data['Valor Facturado'], 1)  # Grado 1 (lineal)
p = np.poly1d(z)

# Agregar línea de media y franjas de cuartiles
plt.axhline(y=stats['mean'], color='green', linestyle='--', label='Media')
plt.fill_between(data.index, stats['25%'], stats['75%'], color='lightblue', alpha=0.3, label='IQR (25% - 75%)')
plt.fill_between(data.index, stats['min'], stats['25%'], color='lightcoral', alpha=0.3, label='Min - 25% Percentil')
plt.fill_between(data.index, stats['75%'], stats['max'], color='lightgreen', alpha=0.3, label='75% Percentil - Max')
plt.plot(data.index, p(range(len(data))), color='orange', linestyle='--', label='Línea de Tendencia', linewidth=2)

# Personalizar el gráfico
plt.title('Consumo historico servicio de Cirugia Minimamente Invasiva', fontsize=12)
plt.xlabel('Periodo', fontsize=10)
plt.ylabel('Valor Facturado (en miles)', fontsize=10)
plt.grid(True)
plt.legend(fontsize=10)

# Formatear el eje x para mostrar solo los años
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Mostrar ticks anuales
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato de año
plt.xticks(rotation=0, fontsize=10)

# Función para formatear los valores con separadores de miles
def format_miles(x, pos):
    return f'{int(x):,}'  # Formatear con separadores de miles
# Aplicar el formateador al eje Y
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_miles))
plt.show()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Función principal para entrenar y evaluar tres modelos
def evaluate_time_series_models(data, train_size=0.8):
    # Dividir los datos en entrenamiento y prueba
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]

    results = {}

    # Modelo 1: Regresión Lineal
    def linear_regression_model(train, test):
        X_train = np.arange(len(train)).reshape(-1, 1)
        y_train = train.values
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = sqrt(mean_squared_error(test, predictions))
        results['Linear Regression'] = rmse
        return predictions

    # Modelo 2: ARIMA
    def arima_model(train, test, order=(1,1,1)):
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))

        rmse = sqrt(mean_squared_error(test, predictions))
        results['ARIMA'] = rmse
        return predictions

    # Modelo 3: LSTM
    def lstm_model(train, test, epochs=10, batch_size=1):
        # Preparar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

        # Crear secuencias para el modelo LSTM
        X_train, y_train = [], []
        for i in range(1, len(train_scaled)):
            X_train.append(train_scaled[i-1:i, 0])
            y_train.append(train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))

        # Definir el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar el modelo
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Realizar predicciones
        inputs = scaler.transform(train[-1:].values.reshape(-1, 1))
        X_test = []
        for i in range(len(test)):
            X_test.append(inputs)
            inputs = np.append(inputs[1:], model.predict(inputs.reshape(1, 1, 1))[0][0]).reshape(-1, 1)
        predictions = scaler.inverse_transform(np.array(X_test).reshape(-1, 1))

        rmse = sqrt(mean_squared_error(test, predictions))
        results['LSTM'] = rmse
        return predictions.flatten()

    # Ejecutar cada modelo y almacenar los resultados
    linear_pred = linear_regression_model(train, test)
    arima_pred = arima_model(train, test)
    lstm_pred = lstm_model(train, test)

    # Mostrar los resultados
    print("Root Mean Square Error (RMSE) de cada modelo:")
    for model, error in results.items():
        print(f"{model}: {error:.2f}")

    # Retornar las predicciones para compararlas visualmente
    return linear_pred, arima_pred, lstm_pred, test

# Llamada a la función y visualización de resultados
linear_pred, arima_pred, lstm_pred, test = evaluate_time_series_models(data['Valor Facturado'])

# Gráfico comparativo - Forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual', color='black')
plt.plot(test.index, linear_pred, label='Linear Regression', linestyle='--')
plt.plot(test.index, arima_pred, label='ARIMA', linestyle='--')
plt.plot(test.index, lstm_pred, label='LSTM', linestyle='--')
plt.title("Comparación de Modelos Predictivos")
plt.xlabel("Tiempo")
plt.ylabel("Valor Facturado (en miles)")
plt.legend()
plt.show()

def forecast_next_values(data, linear_rmse, arima_rmse, lstm_rmse, p_best, d_best, q_best):
    # Determinar el modelo con menor RMSE
    best_model = min(('Linear Regression', linear_rmse), ('ARIMA', arima_rmse), ('LSTM', lstm_rmse), key=lambda x: x[1])[0]
    print(f"Mejor modelo basado en RMSE: {best_model}")

    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

        # Configuración para los próximos 6 períodos
    forecast_steps = 6
    last_values = data['Valor Facturado'].values

    if best_model == 'Linear Regression':
        # Modelo de Regresión Lineal
        X = np.arange(len(last_values)).reshape(-1, 1)
        model = LinearRegression().fit(X, last_values)

        # Predicciones de los próximos 6 pasos
        future_X = np.arange(len(last_values), len(last_values) + forecast_steps).reshape(-1, 1)
        forecast = model.predict(future_X)

    elif best_model == 'ARIMA':
        # Modelo ARIMA con los mejores parámetros
        model = ARIMA(last_values, order=(p_best, d_best, q_best))
        model_fit = model.fit()

        # Predicciones de los próximos 6 pasos
        forecast = model_fit.forecast(steps=forecast_steps)

    elif best_model == 'LSTM':
        # Preprocesamiento para el modelo LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(last_values.reshape(-1, 1))

        generator = TimeseriesGenerator(scaled_data, scaled_data, length=12, batch_size=1)

        # Modelo LSTM
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(12, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(generator, epochs=20, verbose=0)

        # Generar predicciones para los próximos 6 pasos
        forecast = []
        current_batch = scaled_data[-12:].reshape(1, 12, 1)
        for _ in range(forecast_steps):
            pred = model.predict(current_batch)[0]
            forecast.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

        # Inversión del escalado para obtener los valores originales
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    # Graficar los datos originales y las predicciones
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Valor Facturado'], label='Datos Originales', color='blue')

    # Crear un índice para las predicciones
    future_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
    plt.plot(future_index, forecast, label='Predicciones', marker='o', color='orange')

    plt.title(f'Predicciones utilizando el modelo: {best_model}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor Facturado (en miles)', fontsize=12)
    plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Inicio de Predicciones')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast

# Llamada a la función con los RMSE y parámetros de los modelos (suponiendo que ya los calculaste)
linear_rmse = 8006.51
arima_rmse = 11719.84
lstm_rmse = 13466.44
p_best, d_best, q_best = 1, 1, 1  # Parámetros óptimos del modelo ARIMA

# Obtener las predicciones de los próximos 6 valores y graficar
forecast_values = forecast_next_values(data, linear_rmse, arima_rmse, lstm_rmse, p_best, d_best, q_best)
print("Predicciones para los próximos 6 períodos:", forecast_values)

