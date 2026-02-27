import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv('biogas_dataset.csv')

#Since we energy grass is being used as input
df = df[['Year','Month', 'Day', 'biogas_production', 'Energy Grass (kg)','Water (L)','Diesel (L)','Electricity Use (kWh)','Temperature (C)','Humidity (%)',
         'Rainfall (mm)','C/N Ratio','Digester Temp (C)']]

#Conveting Year, Month, and Day to one column called date
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])

#Dropping Year, Month, and Day
df = df.drop(columns=['Year','Month','Day'])

#Sorting Date column in ascending order
df.sort_values('Date')

#Making Date column the index
df.set_index('Date',inplace=True)
print(df)

# Fix random starting point for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Train test split
size = int(len(df) * 0.8)
x_train, x_test = df[0:size], df[size:len(df)]

# CHANGE: update column names to match your dataset (first column should always be your target column)
x_train = x_train[['biogas_production', 'Energy Grass (kg)','Water (L)','Diesel (L)','Electricity Use (kWh)','Temperature (C)','Humidity (%)',
         'Rainfall (mm)','C/N Ratio','Digester Temp (C)']]
x_test = x_test[['biogas_production', 'Energy Grass (kg)','Water (L)','Diesel (L)','Electricity Use (kWh)','Temperature (C)','Humidity (%)',
         'Rainfall (mm)','C/N Ratio','Digester Temp (C)']]

# Scale data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

# Create sequences manually
def create_sequences(data, n_input):
    x, y = [], []
    for i in range(len(data) - n_input):
        x.append(data[i: i + n_input])
        y.append(data[i + n_input])
    return np.array(x), np.array(y)

# CHANGE: update n_input (lookback period) to match your data frequency (e.g. 12 for monthly, 7 for daily, 24 for hourly, 144 for 10 minute intervals)
n_input = 7
# CHANGE: update n_features to match the number of columns you selected above
n_features = 10

x_train, y_train = create_sequences(scaled_train, n_input)
x_test, y_test = create_sequences(scaled_test, n_input)

# Define and train model
model = Sequential()

# CHANGE: update neurons (first number) if needed. More neurons = more complex patterns but slower training
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# CHANGE: update epochs if needed. More epochs = better learning but slower training
model.fit(x_train, y_train, epochs=200, verbose=1, shuffle=False)

# Train RMSE
train_predictions = model.predict(x_train)
train_predictions_full = np.zeros((len(train_predictions), n_features))
train_predictions_full[:, 0] = train_predictions[:, 0]
train_predictions = scaler.inverse_transform(train_predictions_full)[:, 0]
actual_train = scaler.inverse_transform(y_train)[:, 0]
train_rmse = np.sqrt(mean_squared_error(actual_train, train_predictions))
train_mape = mean_absolute_percentage_error(actual_train, train_predictions) * 100
print(f'Train RMSE: {train_rmse}')
print(f'Train MAPE: {train_mape}%')

# Test RMSE
test_predictions = model.predict(x_test)
test_predictions_full = np.zeros((len(test_predictions), n_features))
test_predictions_full[:, 0] = test_predictions[:, 0]
test_predictions = scaler.inverse_transform(test_predictions_full)[:, 0]
actual = scaler.inverse_transform(y_test)[:, 0]
test_rmse = np.sqrt(mean_squared_error(actual, test_predictions))
test_mape = mean_absolute_percentage_error(actual, test_predictions) * 100
print(f'Test RMSE: {test_rmse}')
print(f'Test MAPE: {test_mape}%')

# CHANGE: update n_future to however many steps ahead you want to forecast
n_future = 336
last_batch = scaled_train[-n_input:].reshape(1, n_input, n_features)

future_predictions = []
for i in range(n_future):
    future_pred = model.predict(last_batch, verbose=0)

    #in a real world scenario, replace the two new_pred equations with (new_pred = np.array([[[predicted_yield, real_temperature, real_pressure, real_flow_rate]]])
    new_pred = np.zeros((1, 1, n_features))
    new_pred[0, 0, 0] = future_pred[0][0]
    future_predictions.append(future_pred[0])
    last_batch = np.append(last_batch[:, 1:, :], new_pred, axis=1)

future_predictions_full = np.zeros((len(future_predictions), n_features))
future_predictions_full[:, 0] = np.array(future_predictions)[:, 0]
future_predictions = scaler.inverse_transform(future_predictions_full)[:, 0]
test_dates = df.index[size + n_input:]

# CHANGE: update freq to match your data frequency (e.g. 'MS' for monthly, 'D' for daily, 'H' for hourly, '10min' for 10 minute intervals)
future_dates = pd.date_range(df.index[-1], periods=n_future+1, freq='D')[1:]

# CHANGE: update column name to match your target column
plt.plot(df.index, df['biogas_production'], label='Actual', color='red')
plt.plot(test_dates, test_predictions, label='Test Predictions', color='blue')
plt.plot([test_dates[-1]] + list(future_dates), [test_predictions[-1]] + list(future_predictions), label='Future Forecast', color='green')

# CHANGE: update title and axis labels to match your dataset
plt.title('Biogas Yield Forecast (Energy Grass Feedstock)')
plt.xlabel('Date')
plt.ylabel('Biogas yield')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()