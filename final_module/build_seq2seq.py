import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers
import IPython
import IPython.display
from statsmodels.tsa.seasonal import STL
import random
from scipy.optimize import minimize

import keras
from keras.utils import timeseries_dataset_from_array
from keras import layers, models, Input
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler

# used info from https://www.geeksforgeeks.org/machine-learning/seq2seq-model-in-machine-learning/
def volatility_aware_loss(y_true, y_pred):
    # MSE component
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Derivative matching - penalize difference in volatility
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    derivative_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff))

    # Add second derivative (acceleration/curvature)
    y_true_diff2 = y_true_diff[:, 1:] - y_true_diff[:, :-1]
    y_pred_diff2 = y_pred_diff[:, 1:] - y_pred_diff[:, :-1]
    second_derivative_loss = tf.reduce_mean(tf.square(y_true_diff2 - y_pred_diff2))

    # INCREASE WEIGHTS - this is key!
    return mse + 4.0 * derivative_loss + 2.0 * second_derivative_loss


class ScheduledSamplingCallback(keras.callbacks.Callback):
    def __init__(self, initial_ratio=1.0, decay_rate=0.95):
        super().__init__()
        self.ratio = initial_ratio
        self.decay_rate = decay_rate

    def on_epoch_end(self, epoch, logs=None):
        self.ratio = max(0.0, self.ratio * self.decay_rate)
        print(f"\nTeacher forcing ratio: {self.ratio:.3f}")

def train_with_teacher_forcing(model,model_name, X_train, y_train, X_val, y_val,X_test, y_test,
                               output_seq_len, epochs=50, batch_size=32,
                               lr=0.00001, teacher_forcing_ratio=0.5):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),#weight_decay=0.001,clipnorm=2.0,beta_1=0.87,beta_2=0.98,amsgrad=True,use_ema=True),
        loss=volatility_aware_loss,
        metrics=['mae']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    # Create decoder inputs with partial ground truth
    decoder_input_train = np.zeros((len(y_train), output_seq_len, 1))
    decoder_input_val = np.zeros((len(y_val), output_seq_len, 1))
    decoder_input_test = np.zeros((len(y_test), output_seq_len, 1))

    # Use part of ground truth as decoder input (teacher forcing)
    for i in range(len(y_train)):
        for t in range(output_seq_len):
            if np.random.random() < teacher_forcing_ratio and t > 0:
                decoder_input_train[i, t, 0] = y_train[i, t - 1]

    scheduled_sampling = ScheduledSamplingCallback(initial_ratio=0.8, decay_rate=0.95)

    # For training, we'll use a custom training loop
    # But for simplicity, let's use autoregressive prediction during inference

    decoder_input_train = np.zeros((len(y_train), output_seq_len, 1))
    decoder_input_val = np.zeros((len(y_val), output_seq_len, 1))
    decoder_input_test = np.zeros((len(y_test), output_seq_len, 1))

    history = model.fit(
        [X_train, decoder_input_train], y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val, decoder_input_val], y_val),
        callbacks=[early_stop, scheduled_sampling],
        verbose=1,
        shuffle=True
    )

    y_train_pred = model.predict([X_train, decoder_input_train])
    y_val_pred = model.predict([X_val, decoder_input_val])
    y_test_pred = model.predict([X_test, decoder_input_test])
    return history, y_train_pred, y_val_pred, y_test_pred



def build_lstm_seq2seq(input_seq_len, output_seq_len, num_features, hidden_dim, num_layers):
    #Seq2Seq architecture with LSTM encoder-decoder

    encoder_inputs = layers.Input(shape=(input_seq_len, num_features), name='encoder_input')

    encoder = encoder_inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        encoder = layers.LSTM(hidden_dim, return_sequences=return_sequences,
                              return_state=True, name=f'encoder_lstm_{i}',dropout=0.2, recurrent_dropout=0.2)(encoder)
        if i < num_layers - 1:
            encoder = encoder[0]

    if num_layers == 1:
        encoder_outputs, state_h, state_c = encoder
    else:
        _, state_h, state_c = encoder

    encoder_states = [state_h, state_c]

    decoder_inputs = layers.Input(shape=(output_seq_len, 1), name='decoder_input')

    decoder = decoder_inputs
    for i in range(num_layers):
        if i == 0:
            decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True,
                                       return_state=True, name=f'decoder_lstm_{i}')
            decoder, _, _ = decoder_lstm(decoder, initial_state=encoder_states)
        else:
            decoder = layers.LSTM(hidden_dim, return_sequences=True,
                                  name=f'decoder_lstm_{i}',dropout=0.2, recurrent_dropout=0.2)(decoder)

    decoder_outputs = layers.TimeDistributed(
        layers.Dense(1, activation='linear'), name='output')(decoder)
    decoder_outputs = layers.Reshape((output_seq_len,))(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model



def build_gru_seq2seq(input_seq_len, output_seq_len, num_features, hidden_dim, num_layers):
    #Seq2Seq architecture with GRU encoder-decoder https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/
    encoder_inputs = layers.Input(shape=(input_seq_len, num_features), name='encoder_input')

    encoder = encoder_inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        encoder = layers.GRU(hidden_dim, return_sequences=return_sequences,
                             return_state=True, name=f'encoder_gru_{i}')(encoder)
        if i < num_layers - 1:
            encoder = encoder[0]

    # GRU only has one state (hidden state), unlike LSTM which has both h and c
    if num_layers == 1:
        encoder_outputs, state = encoder
    else:
        _, state = encoder

    encoder_states = [state]  # GRU only needs hidden state

    decoder_inputs = layers.Input(shape=(output_seq_len, 1), name='decoder_input')

    decoder = decoder_inputs
    for i in range(num_layers):
        if i == 0:
            decoder_gru = layers.GRU(hidden_dim, return_sequences=True,
                                     return_state=True, name=f'decoder_gru_{i}')
            decoder, _ = decoder_gru(decoder, initial_state=state)
        else:
            decoder = layers.GRU(hidden_dim, return_sequences=True,
                                 name=f'decoder_gru_{i}')(decoder)

    decoder_outputs = layers.TimeDistributed(
        layers.Dense(1, activation='linear'), name='output')(decoder)
    decoder_outputs = layers.Reshape((output_seq_len,))(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def build_lstm_attention_seq2seq(input_seq_len, output_seq_len, num_features,
                                 hidden_dim, num_layers):
    encoder_inputs = layers.Input(shape=(input_seq_len, num_features))

    encoder = encoder_inputs
    encoder_outputs_all = []

    for i in range(num_layers):
        encoder = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,dropout=0.2,  # Add dropout
            recurrent_dropout=0.2,
            name=f'encoder_lstm_{i}'
        )(encoder if i == 0 else encoder[0])
        encoder_outputs_all.append(encoder[0])

    encoder_outputs, state_h, state_c = encoder
    encoder_states = [state_h, state_c]

    decoder_inputs = layers.Input(shape=(output_seq_len, 1))
    decoder = layers.LSTM(hidden_dim, return_sequences=True)(
        decoder_inputs, initial_state=encoder_states
    )

    # Attention mechanism
    attention = layers.Attention()([decoder, encoder_outputs])
    decoder = layers.Concatenate()([decoder, attention])

    decoder_outputs = layers.TimeDistributed(layers.Dense(1))(decoder)
    decoder_outputs = layers.Reshape((output_seq_len,))(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=5000):
        super().__init__()
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

def transformer_encoder_block(x, hidden_dim, num_heads, ff_dim, name):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=hidden_dim, name=f"{name}_mha")(x, x)

    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(hidden_dim),
    ], name=f"{name}_ffn")

    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(0.1)(ffn_output)

    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def positional_encoding(seq_len, hidden_dim):
    seq_len = int(seq_len)
    hidden_dim = int(hidden_dim)

    pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)      # (seq_len, 1)
    i = tf.cast(tf.range(hidden_dim)[tf.newaxis, :], tf.float32)     # (1, hidden_dim)

    angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(hidden_dim, tf.float32))
    angle_rads = pos * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    pe = tf.concat([sines, cosines], axis=-1)

    return tf.cast(pe, tf.float32)


def build_transformer_seq2seq(input_seq_len, output_seq_len, num_features,
                              hidden_dim, num_layers):

    encoder_inputs = layers.Input(shape=(input_seq_len, num_features), name="encoder_input")

    # Project input to hidden_dim
    x = layers.Dense(hidden_dim)(encoder_inputs)

    # Add positional encoding
    pe = positional_encoding(input_seq_len, hidden_dim)
    x = x + pe

    # Transformer encoder blocks
    for i in range(num_layers):
        x = transformer_encoder_block(
            x,
            hidden_dim=hidden_dim,
            num_heads=4,
            ff_dim=hidden_dim * 2,
            name=f"encoder_transformer_{i}"
        )

    pooled = layers.GlobalAveragePooling1D()(x)

    encoder_state_h = layers.Dense(hidden_dim, activation="tanh", name="state_h")(pooled)
    encoder_state_c = layers.Dense(hidden_dim, activation="tanh", name="state_c")(pooled)
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder (same as your LSTM)
    decoder_inputs = layers.Input(shape=(output_seq_len, 1), name="decoder_input")
    decoder = decoder_inputs

    for i in range(num_layers):
        if i == 0:
            decoder, _, _ = layers.LSTM(
                hidden_dim, return_sequences=True, return_state=True,
                name=f"decoder_lstm_{i}"
            )(decoder, initial_state=encoder_states)
        else:
            decoder = layers.LSTM(
                hidden_dim, return_sequences=True,
                name=f"decoder_lstm_{i}", dropout=0.2, recurrent_dropout=0.2
            )(decoder)

    decoder_outputs = layers.TimeDistributed(
        layers.Dense(1, activation="linear"), name="output")(decoder)
    decoder_outputs = layers.Reshape((output_seq_len,))(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
def tcn_block(x, filters, dilation, name):
    conv1 = layers.Conv1D(filters, kernel_size=3, padding="causal",
                          dilation_rate=dilation, activation='relu',
                          name=f"{name}_conv1")(x)
    conv1 = layers.Dropout(0.2)(conv1)

    conv2 = layers.Conv1D(filters, kernel_size=3, padding="causal",
                          dilation_rate=dilation, activation='relu',
                          name=f"{name}_conv2")(conv1)
    conv2 = layers.Dropout(0.2)(conv2)

    # Residual connection
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, kernel_size=1, padding="same",
                          name=f"{name}_residual")(x)

    return layers.Add()([x, conv2])

def build_cnn_seq2seq(input_seq_len, output_seq_len, num_features,
                      hidden_dim, num_layers):

    encoder_inputs = layers.Input(shape=(input_seq_len, num_features), name="encoder_input")

    x = encoder_inputs
    for i in range(num_layers):
        x = tcn_block(x, filters=hidden_dim, dilation=2**i, name=f"tcn_{i}")

    # Use global pooling to create LSTM init states
    pooled = layers.GlobalAveragePooling1D()(x)
    state_h = layers.Dense(hidden_dim, activation="tanh")(pooled)
    state_c = layers.Dense(hidden_dim, activation="tanh")(pooled)
    encoder_states = [state_h, state_c]

    # Decoder (same as your LSTM)
    decoder_inputs = layers.Input(shape=(output_seq_len, 1), name="decoder_input")
    decoder = decoder_inputs

    for i in range(num_layers):
        if i == 0:
            decoder, _, _ = layers.LSTM(
                hidden_dim, return_sequences=True, return_state=True,
                name=f"decoder_lstm_{i}"
            )(decoder, initial_state=encoder_states)
        else:
            decoder = layers.LSTM(
                hidden_dim, return_sequences=True,
                name=f"decoder_lstm_{i}", dropout=0.2, recurrent_dropout=0.2
            )(decoder)

    decoder_outputs = layers.TimeDistributed(
        layers.Dense(1, activation="linear"), name="output")(decoder)
    decoder_outputs = layers.Reshape((output_seq_len,))(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
