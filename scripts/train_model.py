import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ── 1. Load ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/train/SAP_DE_features.csv",
                 index_col="date", parse_dates=True)

FEATURES = [
    "open", "high", "low", "volume",
    "daily_return",
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_mid", "bb_upper", "bb_lower", "bb_width",
]
TARGET = "close"

X = df[FEATURES].values
y = df[TARGET].values
dates = df.index

# ── 2. Train / Val Split (time-ordered, 80/20) ───────────────────────────
split = int(len(X) * 0.8)

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
dates_val       = dates[split:]

print(f"Train: {split} rows  |  Val: {len(X_val)} rows")

# ── 3. Scale ────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_val_sc   = scaler_X.transform(X_val)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_sc   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# ── Helpers ─────────────────────────────────────────────────────────────
Path("outputs/models").mkdir(parents=True, exist_ok=True)

def calc_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:22s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"model": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

def save_predictions(dates_idx, y_true, y_pred, filename):
    pd.DataFrame({
        "date":      dates_idx,
        "actual":    y_true,
        "predicted": y_pred,
    }).to_csv(f"outputs/models/{filename}", index=False)

results = []

# ── 4. Linear Regression ────────────────────────────────────────────────
print("\n--- Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train_sc, y_train_sc)

lr_pred = scaler_y.inverse_transform(
    lr.predict(X_val_sc).reshape(-1, 1)
).ravel()

results.append(calc_metrics(y_val, lr_pred, "Linear Regression"))
save_predictions(dates_val, y_val, lr_pred, "lr_predictions.csv")
joblib.dump(lr, "models/linear_regression.pkl")

# ── 5. Random Forest ────────────────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf.fit(X_train_sc, y_train_sc)

rf_pred = scaler_y.inverse_transform(
    rf.predict(X_val_sc).reshape(-1, 1)
).ravel()

results.append(calc_metrics(y_val, rf_pred, "Random Forest"))
save_predictions(dates_val, y_val, rf_pred, "rf_predictions.csv")
joblib.dump(rf, "models/random_forest.pkl")

fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  Top 5 Features:")
for feat, imp in fi.head(5).items():
    print(f"    {feat:18s}  {imp:.4f}")
fi.to_csv("outputs/models/rf_feature_importance.csv")

# ── 6. LSTM ─────────────────────────────────────────────────────────────
print("\n--- LSTM ---")
LOOKBACK = 60

def make_sequences(X_sc, y_sc, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X_sc)):
        Xs.append(X_sc[i - lookback:i])
        ys.append(y_sc[i])
    return np.array(Xs), np.array(ys)

# Build sequences from the full scaled dataset (scaler fit on train only)
X_all_sc = np.vstack([X_train_sc, X_val_sc])
y_all_sc = scaler_y.transform(y.reshape(-1, 1)).ravel()

X_seq, y_seq = make_sequences(X_all_sc, y_all_sc, LOOKBACK)

# Keep time order: train sequences must fall entirely within the train window
lstm_split = split - LOOKBACK
X_tr_seq, X_vl_seq = X_seq[:lstm_split], X_seq[lstm_split:]
y_tr_seq, y_vl_seq = y_seq[:lstm_split], y_seq[lstm_split:]
dates_lstm_val      = dates[split:]          # aligns with val window

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)
history = model.fit(
    X_tr_seq, y_tr_seq,
    epochs=50,
    batch_size=32,
    validation_data=(X_vl_seq, y_vl_seq),
    callbacks=[early_stop],
    verbose=1,
)

lstm_pred = scaler_y.inverse_transform(
    model.predict(X_vl_seq).reshape(-1, 1)
).ravel()
y_lstm_actual = scaler_y.inverse_transform(
    y_vl_seq.reshape(-1, 1)
).ravel()

results.append(calc_metrics(y_lstm_actual, lstm_pred, "LSTM"))
save_predictions(dates_lstm_val, y_lstm_actual, lstm_pred, "lstm_predictions.csv")
model.save("models/lstm_model.keras")

# ── 7. Summary ──────────────────────────────────────────────────────────
print("\n=== MODEL COMPARISON (Validation Set) ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("outputs/models/model_metrics.csv", index=False)
print("\nAll models and predictions saved.")