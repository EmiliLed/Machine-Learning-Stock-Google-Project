import tools_final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import build_seq2seq
import build_simple_models

def main():
    # using the tools to get the preprocessed data
    X_train, X_val, X_test, num_features, train_mean, train_std_scaled, dates_train, dates_val, dates_test = tools_final.data_preprocessing_3()
    # model parameters
    predicted_col = 'Close_Residual'  # "Residual_trend_close",'Close_Seasonal',
    seasonal_name = 'Close_Seasonal'
    trend_name = 'Close_Trend'
    index_col = X_train.columns.get_loc(predicted_col)
    index_col_2 = X_train.columns.get_loc(trend_name)
    index_seas = X_train.columns.get_loc(seasonal_name)
    input_length = 100
    output_length = 10
    steps_test_visualization = 50
    # network
    hidden_dim = 64  # Count of hidden neurons in the recurrent units.
    layers_stacked_count = 2
    learning_rate = 0.00001  # Small lr helps not to diverge during training.
    learning_rate_seas = 0.0001
    nb_iters = 150
    lr_decay = 0.92  # default: 0.9 . Simulated annealing.
    momentum = 0.5  # default: 0.0 . Momentum technique in weights update
    lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting
    epochs = 500
    batch_size = 50
    ## override
    # num_features=1
    # flags
    execute_residual_training = True
    execute_seasonal_training = True
    execute_trend_training = True
    autoregressive=False
    # get converted data
    X_train_batches, Y_train_batches, X_train_dates, Y_train_dates, X_train_trend, Y_train_trend, X_train_seas, Y_train_seas, help_vec_train = tools_final.get_sequences_3(
        X_train, dates_train, input_length,
        output_length, predicted_col, trend_name, seasonal_name, autogression=autoregressive)
    X_val_batches, Y_val_batches, X_val_dates, Y_val_dates, X_val_trend, Y_val_trend, X_val_seas, Y_val_seas, help_vec_validation = tools_final.get_sequences_3(
        X_val, dates_test, input_length,
        output_length, predicted_col, trend_name, seasonal_name, autogression=autoregressive)
    X_test_batches, Y_test_batches, X_test_dates, Y_test_dates, X_test_trend, Y_test_trend, X_test_seas, Y_test_seas, help_vec_test = tools_final.get_sequences_3(
        X_test, dates_val, input_length,
        output_length, predicted_col, trend_name, seasonal_name, autogression=autoregressive)

    if execute_residual_training:
        model_predict_resdiual=build_seq2seq.build_gru_seq2seq(input_length, output_length, num_features,
                            hidden_dim, layers_stacked_count)
        model_res_history, model_res_train_pred, model_res_val_pred, model_res_test_pred = build_seq2seq.train_with_teacher_forcing(
            model_predict_resdiual, "Name", X_train_batches, Y_train_batches, X_val_batches, Y_val_batches,
    X_test_batches, Y_test_batches, output_seq_len=output_length, epochs=epochs, batch_size=batch_size,
            lr=learning_rate
        )
        model_res_pred, model_res_actual, model_res_mse, model_res_mae, model_res_r2 = tools_final.evaluate_model_adapted(
            model_res_test_pred, Y_test_batches,
            train_mean, train_std_scaled, help_vec_test, "Name", index_col)
        """for i in range(0, len(model_res_pred), steps_test_visualization):
            print(X_test_batches[i, :, index_col] * train_std_scaled[index_col] + train_mean[index_col])
            tools_final.plot_sequence_predictions(
                (X_test_batches[i, :, index_col] + help_vec_test[i, index_col]) * train_std_scaled[index_col] +
                train_mean[index_col],
                model_res_actual[i], model_res_pred[i])"""

    
    if execute_seasonal_training:
        model_predict_seas = build_seq2seq.build_transformer_seq2seq(input_length, output_length, 1,
                                                                 hidden_dim, layers_stacked_count+3)
        model_seas_history, model_seas_train_pred, model_seas_val_pred, model_seas_test_pred = build_seq2seq.train_with_teacher_forcing(
            model_predict_seas, "LSTM", X_train_seas, Y_train_seas, X_val_seas, Y_val_seas,
            X_test_seas, Y_test_seas, output_seq_len=output_length, epochs=epochs, batch_size=batch_size,
            lr=learning_rate_seas
        )


        model_seas_pred, model_seas_actual, model_seas_mse, model_seas_mae, model_r2 = tools_final.evaluate_model_adapted(
            model_seas_test_pred, Y_test_seas,
            train_mean, train_std_scaled, help_vec_test, "NAme", index_seas
        )
        """for i in range(0, len(model_seas_pred), 50):
            print(X_test_seas[i] * train_std_scaled[index_seas] + train_mean[index_seas])
            tools_final.plot_sequence_predictions(
                (X_test_seas[i] + help_vec_test[i, index_seas]) * train_std_scaled[index_seas] + train_mean[
                    index_col],
                model_seas_actual[i], model_seas_pred[i])"""

    if execute_trend_training:
        model_predict_trend = build_seq2seq.build_gru_seq2seq(input_length, output_length, 1,
                                                             hidden_dim, layers_stacked_count)
        model_trend_history, model_trend_train_pred, model_trend_val_pred, model_trend_test_pred = build_seq2seq.train_with_teacher_forcing(
            model_predict_trend, "NAMe", X_train_trend, Y_train_trend, X_val_trend, Y_val_trend,
            X_test_trend, Y_test_trend, output_seq_len=output_length, epochs=epochs, batch_size=batch_size,
            lr=learning_rate_seas*10
        )

        model_trend_pred, model_trend_actual, model_trend_mse, model_trend_mae, model_r2 = tools_final.evaluate_model_adapted(
            model_trend_test_pred, Y_test_trend,
            train_mean, train_std_scaled, help_vec_test, "name", index_col_2
        )
        """for i in range(0, len(model_trend_pred), 50):
            print(X_test_trend[i] * train_std_scaled[index_col_2] + train_mean[index_col_2])
            tools_final.plot_sequence_predictions(
                (X_test_trend[i] + help_vec_test[i, index_col_2]) * train_std_scaled[index_col_2] + train_mean[
                    index_col_2],
                model_trend_actual[i], model_trend_pred[i])"""

    if execute_seasonal_training and execute_trend_training and execute_residual_training:
        tools_final.calculate_metrics(model_trend_actual + model_res_actual+model_seas_actual, model_trend_pred + model_seas_pred+model_res_pred,save=False)
        for i in range(0, len(model_trend_pred), 50):
            print((X_test_trend[i] + help_vec_test[i, index_col_2]) * train_std_scaled[index_col_2] + train_mean[
                    index_col_2])

            print((X_test_batches[i, :, index_col] + help_vec_test[i, index_col]) * train_std_scaled[
                    index_col])
            print((X_test_seas[i] + help_vec_test[i, index_seas]) * train_std_scaled[index_seas] + train_mean[
                    index_seas])
            tools_final.plot_sequence_predictions(
                (X_test_trend[i] + help_vec_test[i, index_col_2]) * train_std_scaled[index_col_2] + train_mean[
                    index_col_2] + (X_test_batches[i, :, index_col] + help_vec_test[i, index_col]) * train_std_scaled[
                    index_col] + train_mean[index_col]+(X_test_seas[i] + help_vec_test[i, index_seas]) * train_std_scaled[index_seas] + train_mean[
                    index_seas],
                model_trend_actual[i] + model_res_actual[i]+model_seas_actual[i], model_trend_pred[i] + model_seas_pred[i]+model_res_pred[i])

    return

if __name__ == '__main__':
    main()
