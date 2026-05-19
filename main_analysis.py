"""
main_analysis.py
Loads saved models and runs baselines, ablation study, and residual analysis.
No retraining of the proposed model — saves time.
Run this after main.py has completed and saved models.
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from data_preprocessing import load_data, remove_outliers_z_transform
from feature_scaling import scale_features
from cnn_lstm_model import extract_features
from baselines import run_all_baselines
from ablation import run_ablation_study
from residual_analysis import run_residual_analysis

# ── Config — must match main.py exactly ───────────────────────────────────────
DATA_PATH = r"D:\AQI\AQI\station_hour\station_hour.csv"
FEATURES  = ['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
             'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
TIMESTEPS = 10

# ── 1. Reproduce data pipeline (no model training) ────────────────────────────
print("Reproducing data pipeline...")
df = load_data(DATA_PATH)
df = df.dropna()
df = df.reset_index(drop=True)

X_raw = df[FEATURES].values
y_raw = df['AQI'].values

n         = len(X_raw)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train_raw = X_raw[:train_end];  y_train_raw = y_raw[:train_end]
X_val_raw   = X_raw[train_end:val_end]; y_val_raw = y_raw[train_end:val_end]
X_test_raw  = X_raw[val_end:];   y_test_raw  = y_raw[val_end:]

print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

# Outlier removal — train only
train_df        = pd.DataFrame(X_train_raw, columns=FEATURES)
train_df['AQI'] = y_train_raw
train_df        = remove_outliers_z_transform(train_df, FEATURES, threshold=5)
X_train_raw     = train_df[FEATURES].values
y_train_raw     = train_df['AQI'].values

# Scale — fit on train only
train_df = pd.DataFrame(X_train_raw, columns=FEATURES)
val_df   = pd.DataFrame(X_val_raw,   columns=FEATURES)
test_df  = pd.DataFrame(X_test_raw,  columns=FEATURES)
train_df['AQI'] = y_train_raw
val_df['AQI']   = y_val_raw
test_df['AQI']  = y_test_raw

train_scaled, val_scaled, test_scaled, tgt_scaler = scale_features(
    train_df, val_df, test_df, FEATURES, 'AQI')

X_train = train_scaled[FEATURES].values;  y_train = train_scaled['AQI'].values
X_val   = val_scaled[FEATURES].values;    y_val   = val_scaled['AQI'].values
X_test  = test_scaled[FEATURES].values;   y_test  = test_scaled['AQI'].values

# Sequences
def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = make_sequences(X_train, y_train, TIMESTEPS)
X_val_seq,   y_val_seq   = make_sequences(X_val,   y_val,   TIMESTEPS)
X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  TIMESTEPS)

n_features = len(FEATURES)
X_train_td = X_train_seq.reshape(X_train_seq.shape[0], TIMESTEPS, n_features, 1)
X_val_td   = X_val_seq.reshape(X_val_seq.shape[0],     TIMESTEPS, n_features, 1)
X_test_td  = X_test_seq.reshape(X_test_seq.shape[0],   TIMESTEPS, n_features, 1)

# ── 2. Load saved models ───────────────────────────────────────────────────────
print("\nLoading saved models...")
try:
    cnn_lstm_model = load_model('cnn_lstm_model.keras')
except Exception:
    cnn_lstm_model = load_model('cnn_lstm_model.h5', compile=False)
    cnn_lstm_model.compile(optimizer='adam', loss='mse')
best_rf_model  = joblib.load('best_rf_model.pkl')
tgt_scaler     = joblib.load('tgt_scaler.pkl')
print("Models loaded.")

# ── 3. Extract features using saved CNN-LSTM ───────────────────────────────────
print("\nExtracting features...")
X_train_features = extract_features(cnn_lstm_model, X_train_td)
X_val_features   = extract_features(cnn_lstm_model, X_val_td)
X_test_features  = extract_features(cnn_lstm_model, X_test_td)

# Proposed model predictions for residual analysis
y_test_pred_inv = tgt_scaler.inverse_transform(
    best_rf_model.predict(X_test_features).reshape(-1,1)).flatten()
y_test_inv = tgt_scaler.inverse_transform(
    y_test_seq.reshape(-1,1)).flatten()

# ── 4. Residual analysis ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("RESIDUAL ANALYSIS")
print("="*50)
run_residual_analysis(y_test_inv, y_test_pred_inv)

# ── 5. Baseline comparisons ───────────────────────────────────────────────────
print("\n" + "="*50)
print("BASELINE COMPARISONS")
print("="*50)
run_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test,
                  X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                  X_test_seq, y_test_seq, tgt_scaler, n_features, TIMESTEPS)

# ── 6. Ablation study ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ABLATION STUDY")
print("="*50)
run_ablation_study(best_rf_model, X_test_features,
                   X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                   X_test_seq, y_test_seq, X_train_td, X_val_td, X_test_td,
                   tgt_scaler, TIMESTEPS, n_features)

print("\nAll analysis complete.")