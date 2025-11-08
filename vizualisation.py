
#read csv file in order to use the data for MAchine Learning
import pandas as pd
from statsmodels.tsa.stl._stl import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')
print(data.head())
# Check for missing values
print(data.isnull().sum())
# Impute missing values

#plot the closing price history
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# Decompose the time series using STL
stl = STL(data['Close'],period=7, seasonal=31)
result = stl.fit()
# Plot the decomposed components
fig = result.plot()
plt.show()
# Plot the closing price with rolling mean and standard deviation
rolling_mean = data['Close'].rolling(window=30).mean()
rolling_std = data['Close'].rolling(window=30).std()
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Close Price')
plt.plot(rolling_mean, label='30-Day Rolling Mean', color='orange')
plt.plot(rolling_std, label='30-Day Rolling Std', color='red')
plt.title('Close Price with Rolling Mean and Standard Deviation')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
plt.show()
data['Close'].fillna(rolling_mean, inplace=True)
# Verify no missing values remain
print(data.isnull().sum())

plot_acf(data['Close'],lags=100)
plt.title('Autocorrelation Function')
plt.show()
