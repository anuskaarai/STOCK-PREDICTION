"""
Keras model builder.

Constructs the exact multi-task BiLSTM + Attention architecture
from the 18 hourly.ipynb (v4) notebook.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

from app.config import (
    LSTM_UNITS_1, LSTM_UNITS_2,
    ATTENTION_HEADS, ATTENTION_KEY_DIM,
    DENSE_UNITS, DROPOUT_LSTM, DROPOUT_DENSE,
    LOSS_WEIGHT_RETURN, LOSS_WEIGHT_OPEN_DIR, LOSS_WEIGHT_CLOSE_DIR,
    LEARNING_RATE,
)

logger = logging.getLogger(__name__)


def build_model(lookback: int, n_features: int) -> keras.Model:
    """
    Build the multi-task BiLSTM + Multi-Head Attention model.

    Architecture (from v4 notebook):
        Input(lookback, n_features)
        → Bidirectional LSTM(96, return_sequences=True)
        → Dropout(0.25)
        → Bidirectional LSTM(48, return_sequences=True)
        → Dropout(0.2)
        → MultiHeadAttention(4 heads, key_dim=24)
        → Add + LayerNormalization
        → GlobalAveragePooling1D
        → Dense(64, relu) → BatchNorm → Dropout(0.2)
        → THREE independent heads:
            - return_output: Dense(32) → Dense(2, linear)
            - open_dir_output: Dense(16) → Dense(1, sigmoid)
            - close_dir_output: Dense(16) → Dense(1, sigmoid)

    Args:
        lookback: Number of time steps in input sequence.
        n_features: Number of features per time step.

    Returns:
        Compiled Keras model.
    """
    logger.info(f"Building model (v4): lookback={lookback}, features={n_features}")

    inp = layers.Input(shape=(lookback, n_features), name="sequence_input")

    # BiLSTM layers
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True)
    )(inp)
    x = layers.Dropout(DROPOUT_LSTM)(x)

    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=True)
    )(x)
    x = layers.Dropout(DROPOUT_DENSE)(x) # v4 uses 0.2 here

    # Multi-Head Attention + residual + norm
    attn = layers.MultiHeadAttention(
        num_heads=ATTENTION_HEADS, key_dim=ATTENTION_KEY_DIM
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Pool
    x = layers.GlobalAveragePooling1D()(x)

    # Dense block (shared)
    shared = layers.Dense(DENSE_UNITS, activation="relu")(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(DROPOUT_DENSE)(shared)

    # Regression head (percent returns)
    reg = layers.Dense(32, activation="relu")(shared)
    ret_out = layers.Dense(2, activation="linear", name="return_output")(reg)

    # Open direction head (classification)
    clf_open = layers.Dense(16, activation="relu")(shared)
    open_dir_out = layers.Dense(1, activation="sigmoid", name="open_dir_output")(clf_open)

    # Close direction head (classification)
    clf_close = layers.Dense(16, activation="relu")(shared)
    close_dir_out = layers.Dense(1, activation="sigmoid", name="close_dir_output")(clf_close)

    model = keras.Model(inputs=inp, outputs=[ret_out, open_dir_out, close_dir_out])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "return_output": keras.losses.Huber(delta=0.5),
            "open_dir_output": "binary_crossentropy",
            "close_dir_output": "binary_crossentropy",
        },
        loss_weights={
            "return_output": LOSS_WEIGHT_RETURN,
            "open_dir_output": LOSS_WEIGHT_OPEN_DIR,
            "close_dir_output": LOSS_WEIGHT_CLOSE_DIR,
        },
        metrics={
            "open_dir_output": "accuracy",
            "close_dir_output": "accuracy",
        },
    )

    logger.info(f"Model built (v4): {model.count_params()} parameters")
    return model
