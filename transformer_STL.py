import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import datetime
import tensorflow as tf
import IPython
import IPython.display

import seaborn as sns
from statsmodels.tsa.seasonal import STL
from tensorflow.keras import layers


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, test_df, batchsize=32,
                 label_columns=None, train_mean=None, train_std=None,
                 dates=None, trend_data=None, seasonal_data=None):
        self.train_df = train_df
        self.batchsize = batchsize
        self.test_df = test_df
        self.dates = dates

        # normalizing variables
        self.train_mean = train_mean
        self.train_std = train_std

        self.trend_data = trend_data
        self.seasonal_data = seasonal_data

        # label indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def inverse_transform(self, normalized_data, column_name):
        if self.train_mean is None or self.train_std is None:
            return normalized_data

        col_mean = self.train_mean[column_name]
        col_std = self.train_std[column_name]

        return (normalized_data * col_std) + col_mean

    def reconstruct_prices(self, model, inputs, test_start_idx=0):
        predictions = model(inputs)

        pred_residuals_norm = predictions[0, :, 0].numpy()

        pred_residuals = self.inverse_transform(pred_residuals_norm, 'Close_Residual')

        if self.trend_data is not None and self.seasonal_data is not None:
            start_idx = test_start_idx + self.input_width
            pred_trend = self.trend_data[start_idx:start_idx + self.label_width]
            pred_seasonal = self.seasonal_data[start_idx:start_idx + self.label_width]

            reconstructed_prices = pred_residuals + pred_trend + pred_seasonal
            return reconstructed_prices, pred_residuals
        else:
            return pred_residuals, pred_residuals

    def plot(self, model=None, max_subplots=3, rescale=True, reconstruct=False, test_start_idx=0):
        plot_col = self.label_columns[0]
        inputs, labels = self.example
        plt.figure(figsize=(10, 4 * max_subplots))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            ax = plt.subplot(max_n, 1, n + 1)

            input_data = inputs[n, :, plot_col_index].numpy()

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            label_data = labels[n, :, label_col_index].numpy()

            if rescale and self.train_mean is not None:
                input_data = self.inverse_transform(input_data, plot_col)
                label_data = self.inverse_transform(label_data, plot_col)
                ylabel = 'Close Residual-Original Scale'
            else:
                ylabel = 'Close Residual-Normalized'

            plt.ylabel(ylabel, fontsize=11)
            plt.plot(self.input_indices, input_data,
                     label='Inputs', marker='.', zorder=-10, linewidth=1.5)

            plt.scatter(self.label_indices, label_data,
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                pred_data = predictions[n, :, label_col_index].numpy()

                if rescale and self.train_mean is not None:
                    pred_data = self.inverse_transform(pred_data, plot_col)

                plt.scatter(self.label_indices, pred_data,
                            marker='X', edgecolors='k', label='Predictions (Residuals)',
                            c='#ff7f0e', s=64)

            if self.dates is not None and len(self.dates) > 0:
                try:
                    start_idx = max(0, len(self.dates) - len(
                        inputs) * self.label_width - self.input_width + n * self.label_width)
                    end_idx = min(len(self.dates), start_idx + self.input_width + self.label_width)

                    if start_idx < len(self.dates) and end_idx <= len(self.dates):
                        start_date = self.dates.iloc[start_idx].strftime('%Y-%m-%d')
                        end_date = self.dates.iloc[min(end_idx - 1, len(self.dates) - 1)].strftime('%Y-%m-%d')
                        title = f'Sample {n + 1}: {start_date} to {end_date}'
                        plt.title(title, fontsize=10, fontweight='bold')
                except:
                    plt.title(f'Sample {n + 1}', fontsize=10, fontweight='bold')
            else:
                plt.title(f'Sample {n + 1}', fontsize=10, fontweight='bold')

            if n == 0:
                plt.legend(loc='best')

            if n < max_n - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time [Banking Days]', fontsize=11)

        plt.subplots_adjust(hspace=0.45)
        plt.tight_layout()
        plt.show()

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batchsize,
        )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, shuffle=True)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def build_transformer_model(input_width, num_features, out_steps,
                            d_model=128, num_heads=4, ff_dim=256, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(input_width, num_features))

    x = layers.Dense(d_model)(inputs)

    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model
    )(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(d_model)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(out_steps * num_features,
                     kernel_initializer=tf.initializers.zeros())(x)
    outputs = layers.Reshape([out_steps, num_features])(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    input_width = 20
    lr = 0.001

    plt.close('all')
    df = pd.read_csv(r"GOOGL_2006-01-01_to_2018-01-01.csv", )
    df0 = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\nDataset loaded: {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    missing_dates = ['2007-01-02', '2010-04-01']
    missing_dates = pd.to_datetime(missing_dates)
    print(f"\nFilling {len(missing_dates)} missing dates...")

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
    print(f"After filling: {len(df)} rows")

    try:
        stl = STL(df['Close'], seasonal=21, trend=121, period=21)
        stl_result = stl.fit()
    except Exception as e:
        print(f" STL error occurred: {e}")
        from statsmodels.tsa.seasonal import seasonal_decompose
        stl_result = seasonal_decompose(df['Close'], model='additive', period=21, extrapolate='fill_value')

    trend_component = stl_result.trend.values
    seasonal_component = stl_result.seasonal.values
    residual_component = stl_result.resid.values

    print(f"\nSTL Components extracted:")
    print(f"  Trend: Mean={np.nanmean(trend_component):.2f}, Std={np.nanstd(trend_component):.2f}")
    print(f"  Seasonal: Mean={np.nanmean(seasonal_component):.2f}, Std={np.nanstd(seasonal_component):.2f}")
    print(f"  Residual: Mean={np.nanmean(residual_component):.2f}, Std={np.nanstd(residual_component):.2f}")

    df['Close_Trend'] = trend_component
    df['Close_Seasonal'] = seasonal_component
    df['Close_Residual'] = residual_component

    df["month"] = np.sin(df["Date"].dt.month / 12 * 2 * np.pi)
    df["day_sin"] = np.sin((df["Date"].dt.day - 1) / (df["Date"].dt.daysinmonth - 1) * 2 * np.pi)
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year

    df_ = df[['Close_Residual', 'High', 'Low', 'Close', 'Volume', 'month', 'day', "day_sin", 'year']]

    trend_full = df['Close_Trend'].values
    seasonal_full = df['Close_Seasonal'].values
    close_full = df['Close'].values

    dates = df['Date']
    n = len(df_)
    num_features = df_.shape[1]

    split_idx = int(n * 0.7)
    train_df = df_[0:split_idx]
    test_df = df_[split_idx:]
    train_dates = dates[0:split_idx]
    test_dates = dates[split_idx:]

    train_trend = trend_full[0:split_idx]
    test_trend = trend_full[split_idx:]
    train_seasonal = seasonal_full[0:split_idx]
    test_seasonal = seasonal_full[split_idx:]
    test_close_original = close_full[split_idx:]

    print(f"\nTrain set: {len(train_df)} rows ({train_dates.iloc[0].date()} to {train_dates.iloc[-1].date()})")
    print(f"Test set: {len(test_df)} rows ({test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()})")
    print(f"Features: {num_features} (including Close_Residual)")

    train_mean = train_df.mean()
    train_std = train_df.std()

    print(f"\nClose Residual statistics (training set):")
    print(f"  Mean: {train_mean['Close_Residual']:.4f}")
    print(f"  Std: {train_std['Close_Residual']:.4f}")

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    OUT_STEPS = 5

    multi_window = WindowGenerator(
        input_width=input_width,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        test_df=test_df,
        train_df=train_df,
        label_columns=['Close_Residual'],
        train_mean=train_mean,
        train_std=train_std,
        dates=test_dates,
        trend_data=test_trend,
        seasonal_data=test_seasonal)

    print(f"\nWindow configuration:")
    print(multi_window)

    # BUILD TRANSFORMER MODEL (replacing LSTM)
    multi_transformer_model = build_transformer_model(input_width, num_features, OUT_STEPS)

    print(f"\nTransformer model architecture:")
    multi_transformer_model.build(input_shape=(None, input_width, num_features))
    multi_transformer_model.summary()

    def compile_and_fit(model, window, patience=10, lr=lr):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        print(f"\nTraining model (patience={patience}, lr={lr})...")
        history = model.fit(
            window.train,
            epochs=500,
            validation_data=window.test,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    history = compile_and_fit(multi_transformer_model, multi_window, patience=10, lr=0.0005)

    IPython.display.clear_output()

    print("\n" + "=" * 80)
    print(" Results Analysis")

    test_performance = multi_transformer_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    print(f"\n Residual Prediction Performance:")
    print(f"  MSE: {test_performance['loss']:.4f}")
    print(f"  MAE: {test_performance['mean_absolute_error']:.4f}")

    mae_dollars_residual = test_performance['mean_absolute_error'] * train_std['Close_Residual']
    print(f"  MAE in dollars (residuals only): ${mae_dollars_residual:.2f}")

    print(f"\n. Full Price Reconstruction Performance:")

    all_predictions = []
    all_reconstructed = []
    all_labels = []

    batch_count = 0
    for inputs, labels in multi_window.test:
        predictions = multi_transformer_model(inputs).numpy()

        for i in range(len(predictions)):
            pred_resid_norm = predictions[i, :, 0]

            pred_resid = pred_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']

            seq_idx = batch_count * multi_window.batchsize + i
            start_idx = seq_idx + multi_window.input_width

            if start_idx + OUT_STEPS <= len(test_trend):
                pred_trend = test_trend[start_idx:start_idx + OUT_STEPS]
                pred_seasonal = test_seasonal[start_idx:start_idx + OUT_STEPS]

                reconstructed = pred_resid + pred_trend + pred_seasonal
                all_reconstructed.append(reconstructed)

                label_resid_norm = labels[i, :, 0].numpy()
                label_resid = label_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']

                actual_prices = label_resid + pred_trend + pred_seasonal
                all_labels.append(actual_prices)
        batch_count += 1

    if len(all_reconstructed) > 0 and len(all_labels) > 0:
        all_reconstructed = np.array(all_reconstructed)
        all_labels = np.array(all_labels)

        mae_reconstruction = np.mean(np.abs(all_reconstructed - all_labels))
        rmse_reconstruction = np.sqrt(np.mean((all_reconstructed - all_labels) ** 2))
        mape_reconstruction = np.mean(np.abs((all_labels - all_reconstructed) / all_labels)) * 100

        print(f"  MAE (Full Prices): ${mae_reconstruction:.2f}")
        print(f"  RMSE (Full Prices): ${rmse_reconstruction:.2f}")
        print(f"  MAPE (Full Prices): {mape_reconstruction:.2f}%")
        print(f"  MAE vs Mean Price: {(mae_reconstruction / np.mean(all_labels)) * 100:.2f}%")

    print(f"  Current Transformer (STL, residuals): {(mae_dollars_residual / 428.0) * 100:.2f}%")
    if len(all_reconstructed) > 0 and len(all_labels) > 0:
        print(f"  Current Transformer (STL, reconstructed): {mape_reconstruction:.2f}%")

    multi_window.plot(multi_transformer_model, rescale=False)

    example_inputs, example_labels = multi_window.example
    example_predictions = multi_transformer_model(example_inputs).numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    for n in range(3):
        input_resid_norm = example_inputs[n, :, 0].numpy()
        label_resid_norm = example_labels[n, :, 0].numpy()
        pred_resid_norm = example_predictions[n, :, 0]

        input_resid = input_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']
        label_resid = label_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']
        pred_resid = pred_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']

        base_idx = n * 50
        if base_idx + multi_window.input_width + OUT_STEPS <= len(test_trend):
            input_trend = test_trend[base_idx:base_idx + multi_window.input_width]
            input_seasonal = test_seasonal[base_idx:base_idx + multi_window.input_width]

            pred_trend = test_trend[base_idx + multi_window.input_width:base_idx + multi_window.input_width + OUT_STEPS]
            pred_seasonal = test_seasonal[
                base_idx + multi_window.input_width:base_idx + multi_window.input_width + OUT_STEPS]

            input_prices = input_resid + input_trend + input_seasonal
            label_prices = label_resid + pred_trend + pred_seasonal
            pred_prices = pred_resid + pred_trend + pred_seasonal

            start_idx = base_idx
            end_idx = base_idx + multi_window.input_width + OUT_STEPS
            if start_idx < len(test_dates) and end_idx <= len(test_dates):
                start_date = test_dates.iloc[start_idx].strftime('%Y-%m-%d')
                end_date = test_dates.iloc[end_idx - 1].strftime('%Y-%m-%d')
            else:
                start_date = "Unknown"
                end_date = "Unknown"

            ax = axes[n]

            input_indices = np.arange(len(input_prices))
            ax.plot(input_indices, input_prices, label=f'Historical (Past {input_width} days)',
                    marker='.', zorder=-10, linewidth=1.5, color='blue')

            pred_indices = np.arange(len(input_prices), len(input_prices) + len(label_prices))

            ax.scatter(pred_indices, label_prices, edgecolors='k', label='Actual Future Price',
                       c='green', s=input_width, zorder=5, marker='o')

            ax.scatter(pred_indices, pred_prices, edgecolors='k', label='Predicted Future Price',
                       c='orange', s=input_width, zorder=5, marker='X')

            ax.axvline(x=len(input_prices) - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)

            ax.set_ylabel('Price ($)', fontsize=11)
            ax.set_title(f'Sample {n + 1}: Full Price Reconstruction\n ({start_date} to {end_date})',
                         fontsize=12, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            if n < 2:
                ax.set_xticks([])
            else:
                ax.set_xticks(np.arange(0, len(input_prices) + len(label_prices), 20))
            errors = np.abs(pred_prices - label_prices)
            mae = np.mean(errors)
            ax.text(0.02, 0.98, f'Pred MAE: ${mae:.2f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    axes[2].set_xlabel('Time [Banking Days]', fontsize=10)
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()

    all_predictions_full = []
    all_actuals_full = []
    all_dates_full = []

    batch_count = 0
    for inputs, labels in multi_window.test:
        predictions = multi_transformer_model(inputs).numpy()

        for i in range(len(predictions)):
            pred_resid_norm = predictions[i, :, 0]
            pred_resid = pred_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']

            seq_idx = batch_count * multi_window.batchsize + i
            start_idx = seq_idx + multi_window.input_width
            if start_idx + OUT_STEPS <= len(test_trend):
                pred_trend = test_trend[start_idx:start_idx + OUT_STEPS]
                pred_seasonal = test_seasonal[start_idx:start_idx + OUT_STEPS]

                reconstructed_prices = pred_resid + pred_trend + pred_seasonal
                all_predictions_full.extend(reconstructed_prices)

                label_resid_norm = labels[i, :, 0].numpy()
                label_resid = label_resid_norm * train_std['Close_Residual'] + train_mean['Close_Residual']
                actual_prices = label_resid + pred_trend + pred_seasonal
                all_actuals_full.extend(actual_prices)

                for j in range(OUT_STEPS):
                    if start_idx + j < len(test_dates):
                        all_dates_full.append(test_dates.iloc[start_idx + j])

        batch_count += 1

    all_predictions_full = np.array(all_predictions_full)
    all_actuals_full = np.array(all_actuals_full)
    all_dates_full = pd.DatetimeIndex(all_dates_full)

    if len(all_predictions_full) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        axes[0].plot(all_dates_full, all_actuals_full, label='Actual Price',
                     linewidth=2, color='green', alpha=0.7)
        axes[0].plot(all_dates_full, all_predictions_full, label='Transformer Predicted Price',
                     linewidth=2, color='orange', alpha=0.7, linestyle='--')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].set_title('Full Price Reconstruction: Transformer Predictions vs Actual (5-Step Ahead)',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        prediction_error = all_predictions_full - all_actuals_full
        axes[1].bar(all_dates_full, prediction_error, color=['red' if x > 0 else 'blue'
                                                             for x in prediction_error],
                    alpha=0.6, width=1)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('Prediction Error ($)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_title('Prediction Error Over Time (Red=Overestimate, Blue=Underestimate)',
                          fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        mean_error = np.mean(prediction_error)
        std_error = np.std(prediction_error)
        axes[1].text(0.02, 0.95, f'Mean Error: ${mean_error:.2f}\nStd Dev: ${std_error:.2f}',
                     transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()
    else:
        print(" No full price predictions available for plotting")
    return


if __name__ == "__main__":
    main()
