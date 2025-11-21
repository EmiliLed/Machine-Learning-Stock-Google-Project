import math

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

# load & prepare
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')
data['Date'] = pd.to_datetime(data['Date'])

df = data.sort_values("Date").set_index("Date")
df = df['Close'].to_frame()

def main(time_step=30):

    #setting up a lstm model
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    #MinMaxScaler and
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    #Prepare the data for LSTM
    def create_dataset(data, time_step=20):
        X, Y = [], []
        for i in range(len(data)-time_step-1):
            a = data[i:(i+time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)



    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    #Split into training and testing sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    #Build the LSTM model

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
    #model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(X_train, Y_train, batch_size=64, epochs=10)
    #Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    #Inverse transform the predictions
    print(train_predict.shape)
    print(test_predict.shape)
    print(Y_test.shape)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train = scaler.inverse_transform([Y_train])
    Y_test = scaler.inverse_transform([Y_test])
    mse_lstm = mean_squared_error(Y_test[0], test_predict[:,0])
    print(f'\nTest MSE (LSTM): {mse_lstm}')
    print(test_predict)

    #Plotting
    plt.figure(figsize=(12,6))
    plt.plot(df.index[train_size + time_step + 1:], Y_test[0], label='Actual', color='blue')
    plt.plot(df.index[train_size + time_step + 1:], test_predict[:,0],label='Predicted (LSTM)', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('LSTM Model - Actual vs Predicted Close Prices with time step '+str(time_step)+' and RMSE: '+str(mse_lstm))
    plt.legend()
    plt.show()


for timestep in [20, 40, 60, 80, 100]:
    main(time_step=timestep)