"""
baselines.py
Fair baseline comparison on the same Indian CPCB dataset.
Identical preprocessing, chronological 70/15/15 split, test set metrics only.
Addresses Reviewer #1 and Reviewer #5.

Baselines:
  1. SVR (Nystroem kernel approximation for speed on large datasets)
  2. Standalone Random Forest
  3. Unidirectional LSTM
  4. GRU
  5. CNN-GRU
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout,
                                     Conv1D, MaxPooling1D, Flatten)
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


def run_svr(X_train, y_train, X_test, y_test, tgt_scaler):
    """
    SVR baseline using Nystroem kernel approximation + LinearSVR.
    Approximates RBF-SVR but scales linearly with dataset size.
    Much faster than exact SVR on large datasets.
    """
    print("\n" + "="*50)
    print("BASELINE 1: SVR (Nystroem approximation)")
    print("="*50)
    print(f"  Training on full train set: {len(X_train)} samples")

    svr_pipeline = make_pipeline(
        Nystroem(kernel='rbf', gamma=0.1, n_components=300, random_state=42),
        LinearSVR(C=1.0, max_iter=2000, random_state=42)
    )
    svr_pipeline.fit(X_train, y_train)

    y_true = tgt_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        svr_pipeline.predict(X_test).reshape(-1,1)).flatten()
    return get_metrics("SVR (Test Set)", y_true, y_pred)


def run_standalone_rf(X_train, y_train, X_test, y_test, tgt_scaler):
    """Standalone Random Forest — flat features, no deep feature extraction."""
    print("\n" + "="*50)
    print("BASELINE 2: Standalone Random Forest")
    print("="*50)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_true = tgt_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred = tgt_scaler.inverse_transform(
        rf.predict(X_test).reshape(-1,1)).flatten()
    return get_metrics("Standalone Random Forest (Test Set)", y_true, y_pred)


def run_unidirectional_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                             X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps):
    """Standalone unidirectional LSTM — no CNN, no RF, no bidirectionality."""
    print("\n" + "="*50)
    print("BASELINE 3: Unidirectional LSTM")
    print("="*50)
    model = Sequential([
        LSTM(100, activation='tanh', input_shape=(timesteps, n_features)),
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
    return get_metrics("Unidirectional LSTM (Test Set)", y_true, y_pred)


def run_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps):
    """Standalone GRU — direct architectural comparison to BiLSTM component."""
    print("\n" + "="*50)
    print("BASELINE 4: GRU")
    print("="*50)
    model = Sequential([
        GRU(100, activation='tanh', input_shape=(timesteps, n_features)),
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
    return get_metrics("GRU (Test Set)", y_true, y_pred)


def run_cnn_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps):
    """CNN-GRU — hybrid baseline, direct comparison to proposed CNN-BiLSTM-RF."""
    print("\n" + "="*50)
    print("BASELINE 5: CNN-GRU")
    print("="*50)
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same',
               input_shape=(timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        GRU(100, activation='tanh'),
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
    return get_metrics("CNN-GRU (Test Set)", y_true, y_pred)


def run_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test,
                      X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                      X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps):
    """
    Run all baselines and print summary table.
    All models use identical data, preprocessing, and chronological split as
    the proposed model. Test set metrics only reported.
    """
    results = []

    results.append(run_svr(
        X_train, y_train, X_test, y_test, tgt_scaler))

    results.append(run_standalone_rf(
        X_train, y_train, X_test, y_test, tgt_scaler))

    results.append(run_unidirectional_lstm(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps))

    results.append(run_gru(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps))

    results.append(run_cnn_gru(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        X_test_seq, y_test_seq, tgt_scaler, n_features, timesteps))

    print("\n" + "="*50)
    print("BASELINE SUMMARY — Test Set, Indian CPCB Dataset, Chronological Split")
    print("="*50)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv("baseline_results.csv", index=False)
    print("\nSaved to baseline_results.csv")
    return summary