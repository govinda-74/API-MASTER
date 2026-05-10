import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

df = pd.read_csv("data/processed/train/SAP_DE_train.csv", index_col="date", parse_dates=True)

Path("outputs/eda").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["close"], linewidth=1)
ax.set_title("SAP.DE - Closing Price (2015-2024)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (EUR)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("outputs/eda/01_closing_price.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(df.index, df["volume"], width=1, alpha=0.6)
ax.set_title("SAP.DE - Daily Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
plt.tight_layout()
plt.savefig("outputs/eda/02_volume.png", dpi=150)
plt.close()

df["daily_return"] = df["close"].pct_change()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(df.index, df["daily_return"], linewidth=0.5, alpha=0.7)
axes[0].set_title("Daily Returns Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Return")

axes[1].hist(df["daily_return"].dropna(), bins=100, edgecolor="none", alpha=0.8)
axes[1].set_title("Daily Return Distribution")
axes[1].set_xlabel("Return")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/eda/03_daily_returns.png", dpi=150)
plt.close()

corr = df[["open", "high", "low", "close", "volume"]].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("OHLCV Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/eda/04_correlation_heatmap.png", dpi=150)
plt.close()

df["sma_20"]  = df["close"].rolling(window=20).mean()
df["sma_50"]  = df["close"].rolling(window=50).mean()
df["sma_200"] = df["close"].rolling(window=200).mean()

df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

df["macd"]        = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]   = df["macd"] - df["macd_signal"]

delta = df["close"].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

df["bb_mid"]   = df["close"].rolling(window=20).mean()
bb_std         = df["close"].rolling(window=20).std()
df["bb_upper"] = df["bb_mid"] + 2 * bb_std
df["bb_lower"] = df["bb_mid"] - 2 * bb_std
df["bb_width"] = df["bb_upper"] - df["bb_lower"]

df_features = df.dropna()

Path("data/processed/train").mkdir(parents=True, exist_ok=True)
df_features.to_csv("data/processed/train/SAP_DE_features.csv")

print(f"Features dataset shape: {df_features.shape}")
print(f"Date range: {df_features.index.min()} to {df_features.index.max()}")
print(f"Columns: {list(df_features.columns)}")
print(f"Null count: {df_features.isnull().sum().sum()}")