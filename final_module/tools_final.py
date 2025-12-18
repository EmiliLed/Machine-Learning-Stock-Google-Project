import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# edit this for ur needs
path=r"GOOGL_2006-01-01_to_2018-01-01(2).csv"
def get_sequences_3 (data,dates_data,input_length,output_length,goal_col="Close",trend_col="Trend",seaonal_col="Seaonal",autogression=False):
    # seperate trand incoporation
    X = []
    Y = []
    X_dates=[]
    Y_dates=[]
    X_trends=[]
    Y_trends=[]
    X_seas = []
    Y_seas = []
    help_vectors_list=[]
    index=data.columns.get_loc(goal_col)
    index_trend=data.columns.get_loc(trend_col)
    index_season=data.columns.get_loc(seaonal_col)
    num_cols=len(data.columns)
    if not autogression:
        for i in range(len(data) - (input_length + output_length)):
            """print(data.values[i:i + input_length])
            print(data.values[i:i + input_length].shape)"""
            help=data.values[i+input_length-1][index] # used for reascaling the value to nesure espcially to avoid bi offset
            help_trend=data.values[i+input_length-1][index_trend]
            help_seas =0# data.values[i + input_length - 1][index_season]
            helper_vec=np.zeros(num_cols)
            helper_vec[index]=help
            helper_vec[index_trend]=help_trend

            X.append(data.values[i:i + input_length]-helper_vec)
            Y.append(data[i + input_length:i + input_length+output_length][goal_col]-help)
            X_dates.append(dates_data[i:i + input_length])
            Y_dates.append(dates_data[i + input_length:i + input_length+output_length])
            X_trends.append(data[i:i + input_length][trend_col]-help_trend)
            Y_trends.append(data[i + input_length:i + input_length+output_length][trend_col]-help_trend)
            X_seas.append(data[i:i + input_length][seaonal_col]-help_seas)
            Y_seas.append(data[i + input_length:i + input_length+output_length][seaonal_col]-help_seas)
            help_vectors_list.append(helper_vec)
    else:
        for i in range(len(data) - (input_length + output_length)):
            """print(data.values[i:i + input_length])
            print(data.values[i:i + input_length].shape)"""
            help = data.values[i + input_length - 1][index]
            help_trend = data.values[i + input_length - 1][index_trend]
            help_seas = data.values[i + input_length - 1][index_season]

            helper_vec = np.zeros(num_cols)
            helper_vec[index] = help
            helper_vec[index_trend] = help_trend
            X.append(data[i:i + input_length][goal_col] - help)
            Y.append(data[i + input_length:i + input_length + output_length][goal_col] - help)
            X_dates.append(dates_data[i:i + input_length])
            Y_dates.append(dates_data[i + input_length:i + input_length + output_length])
            X_trends.append(data[i:i + input_length][trend_col] - help_trend)
            Y_trends.append(data[i + input_length:i + input_length + output_length][trend_col] - help_trend)
            X_seas.append(data[i:i + input_length][seaonal_col] - help_seas)
            Y_seas.append(data[i + input_length:i + input_length + output_length][seaonal_col] - help_seas)
            help_vectors_list.append(helper_vec)






    """X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)"""
    X=np.asarray(X)
    Y=np.asarray(Y)
    return X,Y,X_dates,Y_dates,np.asarray(X_trends),np.asarray(Y_trends),np.asarray(X_seas),np.asarray(Y_seas),np.asarray(help_vectors_list)

def data_preprocessing_no_Split():
    df = pd.read_csv( path
        )
    print(df.head())
    df["Date"] = pd.to_datetime(df["Date"])

    # ... [Keep Missing Date Filling unchanged] ...
    missing_dates = ['2007-01-02', '2010-04-01']
    missing_dates = pd.to_datetime(missing_dates)
    for d in missing_dates:
        if d not in df['Date'].values:
            prev_row = df[df['Date'] < d].iloc[-1]
            next_row = df[df['Date'] > d].iloc[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            avg_values = (prev_row[numeric_cols] + next_row[numeric_cols]) / 2
            new_row = prev_row.copy()
            new_row['Date'] = d
            new_row[numeric_cols] = avg_values
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)

    try:
        stl = STL(df['Close'], seasonal=21, trend=121, period=21)
        stl_result = stl.fit()
    except Exception as e:
        from statsmodels.tsa.seasonal import seasonal_decompose
        stl_result = seasonal_decompose(df['Close'], model='additive', period=21, extrapolate='fill_value')

    df['Close_Trend'] = stl_result.trend.values
    df['Close_Seasonal'] = stl_result.seasonal.values
    df['Close_Residual'] = stl_result.resid.values

    df["month"] = np.sin(df["Date"].dt.month / 12 * 2 * np.pi)
    df["day_sin"] = np.sin((df["Date"].dt.day - 1) / (df["Date"].dt.daysinmonth - 1) * 2 * np.pi)
    df["dow_cos"] = np.cos(df["Date"].dt.dayofweek / 5 * 2 * np.pi)
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year

    df['Close_LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Close_LogReturn'].fillna(0, inplace=True)  # Fill first NaN with 0

    df['Close_Derivative'] = df['Close'].diff()
    df['Close_Derivative'].fillna(0, inplace=True)

    # Define feature set

    trend_full = df['Close_Trend'].values
    seasonal_full = df['Close_Seasonal'].values
    features = ["Close_Residual", "High", "Low", "Close", "Volume", "month", "day", "day_sin", "year", "dow_cos",
                "Close_Derivative", "Close_LogReturn"]
    num_features = len(features)  #

    df_features = df[features]
    print(df_features)
    norm_train = True
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    X_train = df_features[:train_size]
    X_val = df_features[train_size:train_size + val_size]
    X_test = df_features[train_size + val_size:]
    dates_train = df['Date'].iloc[:train_size]
    dates_val = df['Date'].iloc[train_size:train_size + val_size]
    dates_test = df['Date'].iloc[train_size + val_size:]
    if norm_train:
        train_mean = X_train.mean()
        train_std = X_train.std()
    else:
        train_mean = df_features.mean()
        train_std = df_features.std()
    std_factor = 2.5


    return  df_features, num_features, train_mean, std_factor * train_std, df['Date']

def data_preprocessing_3(train_ratio = 0.7,
    val_ratio = 0.15,
    test_ratio = 0.15):
    df = pd.read_csv(
        path)
    print(df.head())
    df["Date"] = pd.to_datetime(df["Date"])

    # ... [Keep Missing Date Filling unchanged] ...
    missing_dates = ['2007-01-02', '2010-04-01']
    missing_dates = pd.to_datetime(missing_dates)
    for d in missing_dates:
        if d not in df['Date'].values:
            prev_row = df[df['Date'] < d].iloc[-1]
            next_row = df[df['Date'] > d].iloc[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            avg_values = (prev_row[numeric_cols] + next_row[numeric_cols]) / 2
            new_row = prev_row.copy()
            new_row['Date'] = d
            new_row[numeric_cols] = avg_values
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)


    try:
        stl = STL(df['Close'], seasonal=21, trend=121, period=21)
        stl_result = stl.fit()
    except Exception as e:
        from statsmodels.tsa.seasonal import seasonal_decompose
        stl_result = seasonal_decompose(df['Close'], model='additive', period=21, extrapolate='fill_value')

    df['Close_Trend'] = stl_result.trend.values
    df["Residual_trend_close"]=df["Close"]-df["Close_Trend"]
    df['Close_Seasonal'] = stl_result.seasonal.values
    df['Close_Residual'] = stl_result.resid.values

    df["month"] = np.sin(df["Date"].dt.month / 12 * 2 * np.pi)
    df["day_sin"] = np.sin((df["Date"].dt.day - 1) / (df["Date"].dt.daysinmonth - 1) * 2 * np.pi)
    df["dow_cos"] = np.cos(df["Date"].dt.dayofweek / 5 * 2 * np.pi)
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year

    df['Close_LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Close_LogReturn'].fillna(0, inplace=True)  # Fill first NaN with 0

    df['Close_Derivative'] = df['Close'].diff()
    df['Close_Derivative'].fillna(0, inplace=True)
    df["Close_log_derivitaive"]=np.log(df['Residual_trend_close'] / df['Residual_trend_close'].shift(1))
    df["log_close_res"]=np.log(df['Residual_trend_close'])

    # Define feature set


    df["Residual_Scaled"]=df["Residual_trend_close"]
    trend_full = df['Close_Trend'].values
    seasonal_full = df['Close_Seasonal'].values
    #features=["Residual_trend_close","Residual_Scaled", "Close_Derivative", "Close_LogReturn",'Close_Trend', "High", "Low", "Close", "Volume", "month", "day", "day_sin", "year","dow_cos",]
    features=['Close_Trend','Close_Seasonal','Close_Residual',"Residual_trend_close","Residual_Scaled", "Close_Derivative", "Close_LogReturn", "High", "Low", "Close", "Volume", "month", "day", "day_sin", "year","dow_cos"]
    num_features = len(features)  #

    df_features = df[features]
    df_help=df_features.copy()
    print(df_features)
    norm_train=True

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    X_train = df_features[:train_size]
    X_val = df_features[train_size:train_size + val_size]
    X_test = df_features[train_size + val_size:]
    dates_train = df['Date'].iloc[:train_size]
    dates_val = df['Date'].iloc[train_size:train_size + val_size]
    dates_test = df['Date'].iloc[train_size + val_size:]
    if norm_train:
        train_mean = X_train.mean()
        train_std = X_train.std()
    else:
        train_mean = df_features.mean()
        train_std = df_features.std()
    residual_scaler = RobustScaler()


    std_factor=1
    X_train_norm = (X_train - train_mean) / (std_factor * train_std)
    X_val_norm = (X_val - train_mean) / (std_factor * train_std)
    X_test_norm = (X_test - train_mean) / (std_factor * train_std)
    residual_scaler.fit( df_help["Residual_trend_close"].values[:train_size].reshape(-1, 1))
    #df_features["Residual_Scaled"] = residual_scaler.transform(df_help["Residual_trend_close"].values.reshape(-1, 1)).flatten()
    return X_train_norm, X_val_norm, X_test_norm, num_features, train_mean, std_factor * train_std, dates_train, dates_val, dates_test

def reconstruct_close_price_adapted(residual_pred, train_mean, train_std, helper, index):
    #Reconstruct Close price from predicted +helper and rescale

    residual_denorm = (residual_pred + helper[:, index][:, None]) * train_std[index] + train_mean[index]
    close_reconstructed = residual_denorm
    return close_reconstructed

def evaluate_model_adapted(y_test_pred, y_test_seq,
                           train_mean, train_std, helper, model_name, index):
    #Evaluate model performance
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_test_close_pred = reconstruct_close_price_adapted(
        y_test_pred, train_mean, train_std, helper, index)

    y_test_close_actual = reconstruct_close_price_adapted(
        y_test_seq, train_mean, train_std, helper, index)

    mse_test = mean_squared_error(y_test_close_actual.flatten(),
                                  y_test_close_pred.flatten())
    mae_test = mean_absolute_error(y_test_close_actual.flatten(),
                                   y_test_close_pred.flatten())
    r2_test = r2_score(y_test_close_actual.flatten(),
                       y_test_close_pred.flatten())

    print("=" * 60)
    print(f"MSE: {mse_test:.4f}")
    print(f"RMSE: {np.sqrt(mse_test):.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"R² Score: {r2_test:.4f}")
    print("=" * 60)

    return y_test_close_pred, y_test_close_actual, mse_test, mae_test, r2_test

def plot_sequence_predictions(input_seq, actual_output, predicted_output,
                              title="Sequence Prediction Comparison",
                              figsize=(12, 6)):

    # Convert to numpy arrays if needed
    input_seq = np.array(input_seq)
    actual_output = np.array(actual_output)
    predicted_output = np.array(predicted_output)

    if input_seq.ndim == 2:
        input_len = input_seq.shape[0]
    else:
        input_len = len(input_seq)

    output_len = len(actual_output)

    input_x = np.arange(input_len)
    output_x = np.arange(input_len, input_len + output_len)

    fig, ax = plt.subplots(figsize=figsize)

    if input_seq.ndim == 2:
        for i in range(input_seq.shape[1]):
            ax.plot(input_x, input_seq[:, i], 'o-', alpha=0.6,
                    label=f'Input Feature {i + 1}', linewidth=2)
    else:
        ax.plot(input_x, input_seq, 'bo-', label='Input Sequence',
                linewidth=2, markersize=6)

    ax.plot(output_x, actual_output, 'go-', label='Actual Output',
            linewidth=2, markersize=8, markerfacecolor='lightgreen')

    ax.plot(output_x, predicted_output, 'r^--', label='Predicted Output',
            linewidth=2, markersize=8, alpha=0.7)


    ax.axvline(x=input_len - 0.5, color='gray', linestyle='--',
               alpha=0.5, linewidth=1.5, label='Input/Output Boundary')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ymin=0)
    plt.tight_layout()
    plt.show()
    return fig, ax

def calculate_metrics(actuals_fval, predictions_fval,save=False):
    if save:
        np.save('my_array_ac.npy', actuals_fval)
        np.save('my_array_pred.npy', predictions_fval)
    mse = mean_squared_error(actuals_fval, predictions_fval)
    mae = mean_absolute_error(actuals_fval, predictions_fval)
    r2 = r2_score(actuals_fval, predictions_fval)
    rmse = np.sqrt(mse)
    print("#0"*60)
    print(f"\\nTest Set Metrics_all:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print(f"R2: {r2:.2f}%")

