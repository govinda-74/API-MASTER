import pandas as pd
from pathlib import Path

train_raw = pd.read_csv("data/raw/train/SAP_DE_historical.csv")
test_raw = pd.read_csv("data/raw/test/SAP_DE_historical.csv")

train_raw.columns = train_raw.columns.str.lower().str.replace(" ", "_")
test_raw.columns = test_raw.columns.str.lower().str.replace(" ", "_")

train_raw.drop(columns=["dividends", "stock_splits"], inplace=True)

train_raw["date"] = pd.to_datetime(train_raw["date"], utc=True).dt.tz_localize(None).dt.normalize()
test_raw["date"] = pd.to_datetime(test_raw["date"])

train_raw.sort_values("date", inplace=True)
test_raw.sort_values("date", inplace=True)

train_raw.set_index("date", inplace=True)
test_raw.set_index("date", inplace=True)

train_original_rows = len(train_raw)
train_raw["volume"] = train_raw["volume"].replace(0, pd.NA)
train_raw[["open", "high", "low", "close"]] = train_raw[["open", "high", "low", "close"]].ffill()
train_raw["volume"] = train_raw["volume"].ffill()
train_raw.dropna(inplace=True)
train_after_rows = len(train_raw)
rows_removed = train_original_rows - train_after_rows

assert train_raw.isnull().sum().sum() == 0, "Train data still has nulls!"
assert test_raw.isnull().sum().sum() == 0, "Test data still has nulls!"

Path("data/processed/train").mkdir(parents=True, exist_ok=True)
Path("data/processed/test").mkdir(parents=True, exist_ok=True)

train_raw.to_csv("data/processed/train/SAP_DE_train.csv")
test_raw.to_csv("data/processed/test/SAP_DE_test.csv")

print("Preprocessing complete.")
print(f"Train shape: {train_raw.shape}")
print(f"Test shape:  {test_raw.shape}")
print(f"Train date range: {train_raw.index.min()} to {train_raw.index.max()}")
print(f"Test date range:  {test_raw.index.min()} to {test_raw.index.max()}")
print(f"Rows removed (zero-volume forward-fill): {rows_removed}")