"""
ablation.py
Component-level ablation study for the CNN-BiLSTM-RF hybrid model.
All variants trained on same data, evaluated on test set only.
Addresses Reviewer #5.

Variants:
  A) CNN only         — TimeDistributed CNN + Dense, no LSTM, no RF
  B) BiLSTM only      — BiLSTM + Dense, no CNN, no RF
  C) CNN + BiLSTM     — full deep model with direct Dense output, no RF
  D) CNN + BiLSTM + RF — proposed model (reuses saved model, no retraining)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                     Dropout, Bidirectional, LSTM,
                                     BatchNormalization, TimeDistributed)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]


def get_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return dict(Model=name, MAE=round(mae,4), MSE=round(mse,4),
                RMSE=round(rmse,4), R2=round(r2,4))


def run_cnn_only(X_train_td, y_train_seq, X_val_td, y_val_seq,
                 X_test_td, y_test_seq, tgt_scaler, timesteps, n_features):
    """
    Ablation A — TimeDistributed CNN only, no LSTM, no RF.
    Uses same 4D input as proposed model for fair comparison.
    """
    print("\n" + "="*50)
    print("ABLATION A: CNN Only")
    print("="*50)
    model = Sequential([
        TimeDistributed(Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                        input_shape=(timesteps, n_features, 1)),
        TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dense(1)),
    ])
    # Reduce to scalar output by taking last timestep
    from tensorflow.keras.layers import Lambda
    import tensorflow as tf
    model2 = Sequential([
        TimeDistributed(Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                        input_shape=(timesteps, n_features, 1)),
        TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dense(1)),
        Lambda(lambda x: x[:, -1, :])  # take last timestep output
    ])
    model2.compile(optimizer='adam', loss='mse')
    model2.fit(X_train_td, y_train_seq, epochs=50, batch_size=32,
               validation_data=(X_val_td, y_val_seq),
               callbacks=CALLBACKS, verbose=1)

    y_true = tgt_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        model2.predict(X_test_td).reshape(-1,1)).flatten()
    return get_metrics("A) CNN Only (Test Set)", y_true, y_pred)


def run_bilstm_only(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    X_test_seq, y_test_seq, tgt_scaler, timesteps, n_features):
    """Ablation B — BiLSTM only, no CNN, no RF."""
    print("\n" + "="*50)
    print("ABLATION B: BiLSTM Only")
    print("="*50)
    model = Sequential([
        Bidirectional(LSTM(100, activation='tanh'),
                      input_shape=(timesteps, n_features)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32,
              validation_data=(X_val_seq, y_val_seq),
              callbacks=CALLBACKS, verbose=1)

    y_true = tgt_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        model.predict(X_test_seq).reshape(-1,1)).flatten()
    return get_metrics("B) BiLSTM Only (Test Set)", y_true, y_pred)


def run_cnn_bilstm_only(X_train_td, y_train_seq, X_val_td, y_val_seq,
                         X_test_td, y_test_seq, tgt_scaler, timesteps, n_features):
    """Ablation C — CNN + BiLSTM with direct Dense output, no RF."""
    print("\n" + "="*50)
    print("ABLATION C: CNN + BiLSTM (no RF)")
    print("="*50)
    model = Sequential([
        TimeDistributed(Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                        input_shape=(timesteps, n_features, 1)),
        TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(100, activation='tanh', return_sequences=False)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_td, y_train_seq, epochs=50, batch_size=32,
              validation_data=(X_val_td, y_val_seq),
              callbacks=CALLBACKS, verbose=1)

    y_true = tgt_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        model.predict(X_test_td).reshape(-1,1)).flatten()
    return get_metrics("C) CNN + BiLSTM no RF (Test Set)", y_true, y_pred)


def run_full_model(best_rf_model, X_test_features, y_test_seq, tgt_scaler):
    """
    Ablation D — Full proposed model (CNN + BiLSTM + RF).
    Reuses already trained model — no retraining.
    """
    print("\n" + "="*50)
    print("ABLATION D: CNN + BiLSTM + RF (Proposed Model)")
    print("="*50)
    y_true = tgt_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        best_rf_model.predict(X_test_features).reshape(-1,1)).flatten()
    return get_metrics("D) CNN + BiLSTM + RF (Test Set)", y_true, y_pred)


def run_ablation_study(best_rf_model, X_test_features,
                       X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                       X_test_seq, y_test_seq, X_train_td, X_val_td, X_test_td,
                       tgt_scaler, timesteps, n_features):
    """Run full ablation study and save summary."""
    results = []

    results.append(run_cnn_only(
        X_train_td, y_train_seq, X_val_td, y_val_seq,
        X_test_td, y_test_seq, tgt_scaler, timesteps, n_features))

    results.append(run_bilstm_only(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq, tgt_scaler, timesteps, n_features))

    results.append(run_cnn_bilstm_only(
        X_train_td, y_train_seq, X_val_td, y_val_seq,
        X_test_td, y_test_seq, tgt_scaler, timesteps, n_features))

    results.append(run_full_model(
        best_rf_model, X_test_features, y_test_seq, tgt_scaler))

    print("\n" + "="*50)
    print("ABLATION STUDY SUMMARY (Test Set)")
    print("="*50)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv("ablation_results.csv", index=False)
    print("\nSaved to ablation_results.csv")
    return summary