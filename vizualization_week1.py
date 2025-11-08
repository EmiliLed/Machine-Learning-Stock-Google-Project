import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

data=pd.read_csv(r"C:\Users\zu154553\Downloads\GOOGL_2006-01-01_to_2018-01-01.csv",)
data["Date"]=pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
print(data.head())
print(data.describe())
print(data.columns)
register_matplotlib_converters()


plt.figure(figsize=(10, 5))
plt.plot( data['Open'], color="b", linestyle='-',label="Open")
plt.plot( data['Open'].rolling(200).mean().shift(-100), color="orange", linestyle='-',label="Rolling mean Open")
plt.plot( data['High'], color="k", linestyle='-',label="High")
plt.plot( data['Low'], color="r", linestyle='-',label="Low")
plt.plot( data['Close'], color="g", linestyle='-',label="Close")
#plt.plot(data.index, data['Volume'], color="y", linestyle='-')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot( data['Volume'], color="b", linestyle='-',label="Volume")
plt.plot( data['Volume'].rolling(200).mean().shift(-100), color="orange", linestyle='-',label="Rolling mean Volume")
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
sns.set_style("darkgrid")
open=pd.read_csv(r"C:\Users\zu154553\Downloads\GOOGL_2006-01-01_to_2018-01-01.csv",parse_dates=True,
    index_col=0,).iloc[:, 0]
open.index = pd.to_datetime(open.index)
open = open.asfreq('B').ffill()
print(open.head())
stl = STL(open,seasonal=5,)
res = stl.fit()
fig = res.plot()
plt.show()
### autocorrelation
plot_acf(open, lags=200)
plt.show()