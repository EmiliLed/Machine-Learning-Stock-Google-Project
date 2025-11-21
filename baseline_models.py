# python
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# load & prepare
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')
data['Date'] = pd.to_datetime(data['Date'])

df = data.sort_values("Date").set_index("Date")
df = df['Close'].to_frame()

#Missing values handling
print(df.isna().sum())
missing_rows = df[df.isna().any(axis=1)]
if not missing_rows.empty:
    print("\nRows with missing values:")
    print(missing_rows)


# Consecutive missing values
def consecutive_na_runs(s: pd.Series):
    is_na = s.isna()
    groups = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(groups).transform("sum")
    return run_lengths.where(is_na & (run_lengths > 1))

if not missing_rows.empty:
    print("\nConsecutive missing-value runs:")
    for col in df.columns:
        runs = consecutive_na_runs(df[col])
        if runs.notna().any():
            print(f"\nColumn: {col}")
            tmp = pd.DataFrame({
                col: df[col],
                "consecutive_na_length": runs
            }).dropna()
            print(tmp.head())



#Setting up ARIMA model
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]
history = [x for x in train['Close']]
predictions = []
warnings.filterwarnings("ignore")
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test['Close'][t]
    history.append(obs)
    #print(f'predicted={yhat}, expected={obs}')
mse = mean_squared_error(test['Close'], predictions)
print(f'\nTest MSE: {mse}')
# Plotting
plt.figure(figsize=(12,6))
plt.plot(test.index, test['Close'], label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Model - Actual vs Predicted Close Prices')
plt.legend()
plt.show()

# Setting up AutoReg model
history = [x for x in train['Close']]
predictions_ar = []
for t in range(len(test)):
    model = AutoReg(history, lags=5)
    model_fit = model.fit()
    output = model_fit.predict(start=len(history), end=len(history))
    yhat = output[0]
    predictions_ar.append(yhat)
    obs = test['Close'][t]
    history.append(obs)
    #print(f'predicted={yhat}, expected={obs}')
mse_ar = mean_squared_error(test['Close'], predictions_ar)
print(f'\nTest MSE (AutoReg): {mse_ar}')
# Plotting
plt.figure(figsize=(12,6))
plt.plot(test.index, test['Close'], label='Actual', color='blue')
plt.plot(test.index, predictions_ar, label='Predicted (AutoReg)', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AutoReg Model - Actual vs Predicted Close Prices')
plt.legend()
plt.show()



