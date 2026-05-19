from data_preprocessing import load_data, remove_outliers_z_transform
from feature_scaling import scale_features
from cnn_lstm_model import build_cnn_lstm_model, train_cnn_lstm_model, extract_features
from random_forest_model import train_random_forest, tune_random_forest
from evaluation import evaluate_model
from visualization import (plot_actual_vs_predicted, plot_actual_vs_predicted_zoomed,
                            plot_actual_vs_predicted_zoomed_peak,
                            plot_learning_curves, plot_distribution_before_scaling,
                            plot_boxplot_before_scaling, plot_distribution_after_scaling,
                            plot_boxplot_after_scaling)
from baselines import run_all_baselines
from ablation import run_ablation_study
from residual_analysis import run_residual_analysis
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\AQI\AQI\station_hour\station_hour.csv"
FEATURES  = ['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
             'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
TIMESTEPS = 10

# ── 1. Load raw data (nulls retained) ─────────────────────────────────────────
df = load_data(DATA_PATH)
df = df.dropna()                 # drop nulls before split — dropna() fits no parameters so introduces no leakage
df = df.reset_index(drop=True)   # ensure clean 0-based index for chronological slicing

# ── 2. Split FIRST — 70/15/15 chronological split, no shuffling ───────────────
# Chronological order is mandatory for time series — random shuffling breaks
# the temporal structure that CNN-BiLSTM needs to learn from.
X_raw = df[FEATURES].values
y_raw = df['AQI'].values

n          = len(X_raw)
train_end  = int(n * 0.70)
val_end    = int(n * 0.85)

X_train_raw = X_raw[:train_end]
y_train_raw = y_raw[:train_end]
X_val_raw   = X_raw[train_end:val_end]
y_val_raw   = y_raw[train_end:val_end]
X_test_raw  = X_raw[val_end:]
y_test_raw  = y_raw[val_end:]

print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

# ── 3. Visualize BEFORE scaling (train split, raw values) ────────────────────
_train_vis = pd.DataFrame(X_train_raw, columns=FEATURES)
plot_distribution_before_scaling(_train_vis, FEATURES,
    colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plot_boxplot_before_scaling(_train_vis, FEATURES,
    colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
del _train_vis

# ── 5. Outlier removal — train set only ───────────────────────────────────────
train_df        = pd.DataFrame(X_train_raw, columns=FEATURES)
train_df['AQI'] = y_train_raw
train_df        = remove_outliers_z_transform(train_df, FEATURES, threshold=6)
X_train_raw     = train_df[FEATURES].values
y_train_raw     = train_df['AQI'].values

# ── 6. Scale — fit on train, transform val and test ───────────────────────────
train_df = pd.DataFrame(X_train_raw, columns=FEATURES)
val_df   = pd.DataFrame(X_val_raw,   columns=FEATURES)
test_df  = pd.DataFrame(X_test_raw,  columns=FEATURES)
train_df['AQI'] = y_train_raw
val_df['AQI']   = y_val_raw
test_df['AQI']  = y_test_raw

train_scaled, val_scaled, test_scaled, tgt_scaler = scale_features(
    train_df, val_df, test_df, FEATURES, 'AQI')

# ── 7. Visualize AFTER scaling (train set only) ────────────────────────────────
plot_distribution_after_scaling(train_scaled, FEATURES,
    colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plot_boxplot_after_scaling(train_scaled, FEATURES,
    colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])

# ── 8. Extract arrays ──────────────────────────────────────────────────────────
X_train = train_scaled[FEATURES].values
y_train = train_scaled['AQI'].values
X_val   = val_scaled[FEATURES].values
y_val   = val_scaled['AQI'].values
X_test  = test_scaled[FEATURES].values
y_test  = test_scaled['AQI'].values

# ── 9. Build sequences for CNN-LSTM ───────────────────────────────────────────
def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = make_sequences(X_train, y_train, TIMESTEPS)
X_val_seq,   y_val_seq   = make_sequences(X_val,   y_val,   TIMESTEPS)
X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  TIMESTEPS)

# Reshape for TimeDistributed CNN: (samples, timesteps, features, 1)
n_features = len(FEATURES)
X_train_td = X_train_seq.reshape(X_train_seq.shape[0], TIMESTEPS, n_features, 1)
X_val_td   = X_val_seq.reshape(X_val_seq.shape[0],     TIMESTEPS, n_features, 1)
X_test_td  = X_test_seq.reshape(X_test_seq.shape[0],   TIMESTEPS, n_features, 1)

print(f"X_train_td shape: {X_train_td.shape}")

# ── 10. Build and train CNN-LSTM ───────────────────────────────────────────────
cnn_lstm_model = build_cnn_lstm_model(X_train_td.shape[1:])
history        = train_cnn_lstm_model(cnn_lstm_model,
                                      X_train_td, y_train_seq,
                                      X_val_td,   y_val_seq)

# ── 11. Extract features ───────────────────────────────────────────────────────
X_train_features = extract_features(cnn_lstm_model, X_train_td)
X_val_features   = extract_features(cnn_lstm_model, X_val_td)
X_test_features  = extract_features(cnn_lstm_model, X_test_td)

# ── 12. Train and tune Random Forest ──────────────────────────────────────────
rf_model      = train_random_forest(X_train_features, y_train_seq)
best_rf_model = tune_random_forest(X_train_features,  y_train_seq)

# ── 13. Evaluate proposed model ───────────────────────────────────────────────
evaluate_model(best_rf_model,
               X_train_features, y_train_seq,
               X_val_features,   y_val_seq,
               X_test_features,  y_test_seq,
               tgt_scaler)

# ── 14. Visualize predictions ─────────────────────────────────────────────────
plot_learning_curves(history)

y_test_pred_inv = tgt_scaler.inverse_transform(
    best_rf_model.predict(X_test_features).reshape(-1,1)).flatten()
y_test_inv = tgt_scaler.inverse_transform(
    y_test_seq.reshape(-1,1)).flatten()

plot_actual_vs_predicted(y_test_inv, y_test_pred_inv)
plot_actual_vs_predicted_zoomed(y_test_inv, y_test_pred_inv)
plot_actual_vs_predicted_zoomed_peak(y_test_inv, y_test_pred_inv)

'''# ── 15. Residual analysis ─────────────────────────────────────────────────────
run_residual_analysis(y_test_inv, y_test_pred_inv)

# ── 16. Baseline comparisons ──────────────────────────────────────────────────
run_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test,
                  X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                  X_test_seq, y_test_seq, tgt_scaler, n_features, TIMESTEPS)

# ── 17. Ablation study ────────────────────────────────────────────────────────
run_ablation_study(best_rf_model, X_test_features,
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    X_test_seq, y_test_seq, X_train_td, X_val_td, X_test_td,
                    tgt_scaler, TIMESTEPS, n_features)

# ── 18. Save models ───────────────────────────────────────────────────────────
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(tgt_scaler,    'tgt_scaler.pkl')
cnn_lstm_model.save('cnn_lstm_model.keras')
print("Models saved.")'''