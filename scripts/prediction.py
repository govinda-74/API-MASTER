import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

Path("outputs/models").mkdir(parents=True, exist_ok=True)

# ── Load prediction CSVs ────────────────────────────────────────────────
lr   = pd.read_csv("outputs/models/lr_predictions.csv",   parse_dates=["date"])
rf   = pd.read_csv("outputs/models/rf_predictions.csv",   parse_dates=["date"])
lstm = pd.read_csv("outputs/models/lstm_predictions.csv", parse_dates=["date"])
fi   = pd.read_csv("outputs/models/rf_feature_importance.csv",
                   header=None, names=["feature", "importance"])
metrics = pd.read_csv("outputs/models/model_metrics.csv")

# ── Helper ──────────────────────────────────────────────────────────────
def format_ax(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

def metric_label(model_name):
    row = metrics[metrics["model"] == model_name].iloc[0]
    return f"MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}  R2={row['R2']:.4f}"

# ── Plot 1: Linear Regression ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(lr["date"], lr["actual"],    label="Actual",              color="#1f77b4", linewidth=1.2)
ax.plot(lr["date"], lr["predicted"], label="LR Predicted",        color="#ff7f0e", linewidth=1,
        linestyle="--", alpha=0.85)
ax.fill_between(lr["date"],
                lr["actual"], lr["predicted"],
                alpha=0.08, color="orange")
ax.set_title(f"Linear Regression — Actual vs Predicted\n{metric_label('Linear Regression')}")
ax.set_ylabel("Price (EUR)")
ax.legend()
format_ax(ax)
plt.tight_layout()
plt.savefig("outputs/models/01_lr_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved 01_lr_actual_vs_predicted.png")

# ── Plot 2: Random Forest ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rf["date"], rf["actual"],    label="Actual",              color="#1f77b4", linewidth=1.2)
ax.plot(rf["date"], rf["predicted"], label="RF Predicted",        color="#d62728", linewidth=1,
        linestyle="--", alpha=0.85)
ax.fill_between(rf["date"],
                rf["actual"], rf["predicted"],
                alpha=0.08, color="red")
ax.set_title(f"Random Forest — Actual vs Predicted\n{metric_label('Random Forest')}")
ax.set_ylabel("Price (EUR)")
ax.legend()
format_ax(ax)
plt.tight_layout()
plt.savefig("outputs/models/02_rf_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved 02_rf_actual_vs_predicted.png")

# ── Plot 3: LSTM ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(lstm["date"], lstm["actual"],    label="Actual",          color="#1f77b4", linewidth=1.2)
ax.plot(lstm["date"], lstm["predicted"], label="LSTM Predicted",  color="#2ca02c", linewidth=1,
        linestyle="--", alpha=0.85)
ax.fill_between(lstm["date"],
                lstm["actual"], lstm["predicted"],
                alpha=0.08, color="green")
ax.set_title(f"LSTM — Actual vs Predicted\n{metric_label('LSTM')}")
ax.set_ylabel("Price (EUR)")
ax.legend()
format_ax(ax)
plt.tight_layout()
plt.savefig("outputs/models/03_lstm_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved 03_lstm_actual_vs_predicted.png")

# ── Plot 4: Model Comparison (all 3 on one chart) ───────────────────────
# Use LSTM date range as common axis (shortest due to lookback window)
common_dates = lstm["date"].values

lr_aligned   = lr[lr["date"].isin(common_dates)].set_index("date")
rf_aligned   = rf[rf["date"].isin(common_dates)].set_index("date")
lstm_aligned = lstm.set_index("date")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(lstm_aligned.index, lstm_aligned["actual"],
        label="Actual", color="#1f77b4", linewidth=1.5, zorder=5)
ax.plot(lr_aligned.index,   lr_aligned["predicted"],
        label=f"Linear Regression  (R2={metrics.loc[metrics.model=='Linear Regression','R2'].values[0]:.4f})",
        color="#ff7f0e", linewidth=1, linestyle="--", alpha=0.9)
ax.plot(rf_aligned.index,   rf_aligned["predicted"],
        label=f"Random Forest      (R2={metrics.loc[metrics.model=='Random Forest','R2'].values[0]:.4f})",
        color="#d62728", linewidth=1, linestyle="-.", alpha=0.9)
ax.plot(lstm_aligned.index, lstm_aligned["predicted"],
        label=f"LSTM               (R2={metrics.loc[metrics.model=='LSTM','R2'].values[0]:.4f})",
        color="#2ca02c", linewidth=1, linestyle=":", alpha=0.9)
ax.set_title("SAP.DE — Model Comparison: Actual vs Predicted (Validation Set)")
ax.set_ylabel("Price (EUR)")
ax.legend(loc="upper left")
format_ax(ax)
plt.tight_layout()
plt.savefig("outputs/models/04_model_comparison.png", dpi=150)
plt.close()
print("Saved 04_model_comparison.png")

# ── Plot 5: RF Feature Importance ───────────────────────────────────────
fi_sorted = fi.sort_values("importance", ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(fi_sorted["feature"], fi_sorted["importance"],
               color="#1f77b4", alpha=0.8)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_xlabel("Importance Score")
ax.set_title("Random Forest — Top Feature Importances")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/models/05_rf_feature_importance.png", dpi=150)
plt.close()
print("Saved 05_rf_feature_importance.png")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== PREDICTION VISUALISATION COMPLETE ===")
print(f"Plots saved to: outputs/models/")
print(metrics.to_string(index=False))