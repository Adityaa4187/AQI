import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_actual_vs_predicted(y_test_inv, y_test_pred_inv):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Actual AQI', color='blue')
    plt.plot(y_test_pred_inv, label='Predicted AQI', color='orange', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted AQI (Test Set)')
    plt.xlabel('Samples')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted_zoomed(y_test_inv, y_test_pred_inv, zoom_start=0, zoom_points=336):
    y_test_inv      = np.array(y_test_inv)
    y_test_pred_inv = np.array(y_test_pred_inv)
    zoom_end = min(zoom_start + zoom_points, len(y_test_inv))

    plt.figure(figsize=(14, 5))
    plt.plot(range(zoom_start, zoom_end), y_test_inv[zoom_start:zoom_end],
             label='Actual AQI', color='blue', linewidth=1.2, marker='o', markersize=2)
    plt.plot(range(zoom_start, zoom_end), y_test_pred_inv[zoom_start:zoom_end],
             label='Predicted AQI', color='orange', linewidth=1.2, marker='s', markersize=2, alpha=0.8)
    plt.title(f'Actual vs Predicted AQI — Zoomed View ({zoom_points} samples, ~2 weeks)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/actual_vs_predicted_zoomed.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted_zoomed_peak(y_test_inv, y_test_pred_inv, zoom_points=336):
    """
    Zoomed plot centred on the highest AQI peak in the test set.
    Saves to actual_vs_predicted_zoomed_peak.png.
    Addresses Reviewer #1 — shows model behaviour at extremes.
    """
    y_test_inv      = np.array(y_test_inv)
    y_test_pred_inv = np.array(y_test_pred_inv)

    peak_idx   = int(np.argmax(y_test_inv))
    zoom_start = max(0, peak_idx - zoom_points // 2)
    zoom_end   = min(len(y_test_inv), zoom_start + zoom_points)

    plt.figure(figsize=(14, 5))
    plt.plot(range(zoom_start, zoom_end), y_test_inv[zoom_start:zoom_end],
             label='Actual AQI', color='blue', linewidth=1.2, marker='o', markersize=2)
    plt.plot(range(zoom_start, zoom_end), y_test_pred_inv[zoom_start:zoom_end],
             label='Predicted AQI', color='orange', linewidth=1.2, marker='s', markersize=2, alpha=0.8)
    plt.axvline(x=peak_idx, color='red', linestyle='--', linewidth=1, label='Peak AQI')
    plt.title(f'Actual vs Predicted AQI — Zoomed at Peak ({zoom_points} samples centred on max AQI)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/actual_vs_predicted_zoomed_peak.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_distribution_before_scaling(df, features_to_scale, colors):
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col], kde=True, color=colors[i % len(colors)])
        plt.title(f'Before Scaling/Transformation: {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(f"{SAVE_DIR}/dist_before_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()

def plot_boxplot_before_scaling(df, features_to_scale, colors):
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=df[col], color=colors[i % len(colors)])
        plt.title(f'Box Plot Before Scaling: {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(f"{SAVE_DIR}/box_before_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()

def plot_distribution_after_scaling(df_scaled, features_to_scale, colors):
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.histplot(df_scaled[col], kde=True, color=colors[i % len(colors)])
        plt.title(f'After Scaling/Transformation: {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(f"{SAVE_DIR}/dist_after_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()

def plot_boxplot_after_scaling(df_scaled, features_to_scale, colors):
    for i, col in enumerate(features_to_scale):
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=df_scaled[col], color=colors[i % len(colors)])
        plt.title(f'Box Plot After Scaling: {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.savefig(f"{SAVE_DIR}/box_after_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()