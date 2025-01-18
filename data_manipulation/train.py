import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

def create_sequences(features, targets, seq_length=30):
    """
    Create rolling windows of length seq_length:
     - features[i : i+seq_length]
     - targets[i + seq_length - 1]
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(targets[i + seq_length - 1])
    return np.array(X), np.array(y)

def build_transformer_model(n_features, seq_length):
    """
    Build a multi-output Transformer-based model suitable for time series.
    """
    inputs = keras.Input(shape=(seq_length, n_features))

    # Transformer encoder block
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)
    ffn = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(n_features),
    ])
    x2 = ffn(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + x2)

    # Flatten and project to 3 outputs (close, high, low)
    x = layers.Flatten()(x)
    outputs = layers.Dense(3)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    print("# Intensive Time Series Training with Transformer Models")

    data_folder = "processed_data"
    model_folder = "trained_models"
    os.makedirs(model_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print("No processed CSV files found.")
        return

    for file_name in csv_files:
        print(f"\n--- Training on: {file_name} ---")
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)
        if len(df) < 2:
            print("Not enough rows to shift targets; skipping.")
            continue

        # Shift targets: close, high, low by 1 row
        df["target_close"] = df["close"].shift(-1)
        df["target_high"] = df["high"].shift(-1)
        df["target_low"] = df["low"].shift(-1)
        df.dropna(subset=["target_close","target_high","target_low"], inplace=True)

        # Separate features vs. targets
        feature_cols = [c for c in df.columns if c not in ["target_close","target_high","target_low"]]
        X_all = df[feature_cols].values
        y_all = df[["target_close","target_high","target_low"]].values

        # Time series split (80/20)
        split_index = int(len(X_all) * 0.8)
        X_train_raw, X_test_raw = X_all[:split_index], X_all[split_index:]
        y_train_raw, y_test_raw = y_all[:split_index], y_all[split_index:]

        # Create sequences
        seq_length = 30
        X_train, y_train = create_sequences(X_train_raw, y_train_raw, seq_length)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, seq_length)

        if len(X_train) < 1 or len(X_test) < 1:
            print("Not enough data to form sequences; skipping.")
            continue

        # Build and train the Transformer
        model = build_transformer_model(n_features=X_train.shape[-1], seq_length=seq_length)
        model.summary()

        # Early stopping can help if the dataset is large
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50, batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate quickly
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Final test MSE loss: {loss:.4f}")

        # Save model in a TensorFlow SavedModel format
        model_save_path = os.path.join(model_folder, file_name.replace(".csv", "_transformer"))
        model.save(model_save_path)
        print(f"Saved model to: {model_save_path}")

if __name__ == "__main__":
    main()