"""
residual_analysis.py
Residual analysis to validate error distribution and model behaviour.
Accepts pre-computed predictions from main_analysis.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_residual_analysis(y_true, y_pred, save_path="plots/residual_analysis.png"):
    """
    Full residual diagnostic plot — 6 panels.

    Parameters
    ----------
    y_true : array — actual AQI (inverse transformed, original scale)
    y_pred : array — predicted AQI (inverse transformed, original scale)
    save_path : str — output path
    """
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    residuals = y_true - y_pred

    print("=" * 50)
    print("RESIDUAL ANALYSIS REPORT")
    print("=" * 50)
    print(f"  Mean of residuals : {residuals.mean():.4f}  (should be ~0)")
    print(f"  Std  of residuals : {residuals.std():.4f}")
    print(f"  Min  / Max        : {residuals.min():.4f} / {residuals.max():.4f}")

    sample = residuals[np.random.choice(
        len(residuals), size=min(5000, len(residuals)), replace=False)]
    stat, p = stats.shapiro(sample)
    print(f"  Shapiro-Wilk      : W={stat:.4f}, p={p:.4f}")
    if p > 0.05:
        print("  -> Residuals appear normally distributed (p > 0.05)")
    else:
        print("  -> Residuals deviate from normality — acceptable for large n")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 — Residuals vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.3, s=5, color='steelblue')
    ax1.axhline(0, color='red', linewidth=1.5, linestyle='--')
    ax1.set_xlabel("Predicted AQI")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted\n(should be random around 0)")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Residuals over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(residuals[:2000], linewidth=0.5, color='steelblue', alpha=0.7)
    ax2.axhline(0, color='red', linewidth=1.5, linestyle='--')
    ax2.set_xlabel("Sample Index (first 2000)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals over Time\n(should show no trend or pattern)")
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Histogram of residuals
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=80, color='steelblue',
             edgecolor='white', alpha=0.8, density=True)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 300)
    ax3.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
             'r-', linewidth=2, label='Normal fit')
    ax3.set_xlabel("Residual value")
    ax3.set_ylabel("Density")
    ax3.set_title("Residual Distribution\n(should approximate normal curve)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4 — Q-Q Plot
    ax4 = fig.add_subplot(gs[1, 0])
    stats.probplot(sample, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot\n(points should follow diagonal)")
    ax4.grid(True, alpha=0.3)

    # Panel 5 — Actual vs Predicted scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_true, y_pred, alpha=0.2, s=5, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax5.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax5.set_xlabel("Actual AQI")
    ax5.set_ylabel("Predicted AQI")
    ax5.set_title("Actual vs Predicted\n(should cluster on diagonal)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6 — Absolute error by AQI range
    ax6 = fig.add_subplot(gs[1, 2])
    bins   = [0, 100, 200, 300, 400, np.inf]
    labels = ['0-100', '100-200', '200-300', '300-400', '400+']
    abs_err = np.abs(residuals)
    grouped = [abs_err[(y_true >= bins[i]) & (y_true < bins[i+1])]
               for i in range(len(bins)-1)]
    grouped = [g for g in grouped if len(g) > 0]
    ax6.boxplot(grouped,
                labels=[labels[i] for i in range(len(bins)-1) if len(
                    abs_err[(y_true >= bins[i]) & (y_true < bins[i+1])]) > 0],
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax6.set_xlabel("AQI Range")
    ax6.set_ylabel("Absolute Error")
    ax6.set_title("Error by AQI Range\n(checks if model struggles at extremes)")
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Residual Analysis — CNN-BiLSTM-RF Model (Test Set)",
                 fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {save_path}")