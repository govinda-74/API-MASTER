import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/train/SAP_DE_historical.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
df = df.sort_values("date")

df.drop(columns=["dividends", "stock_splits"], errors="ignore", inplace=True)

train = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2024-12-31")].copy()
test  = df[df["date"] >= "2025-01-01"].copy()

Path("data/raw/train").mkdir(parents=True, exist_ok=True)
Path("data/raw/test").mkdir(parents=True, exist_ok=True)

train.to_csv("data/raw/train/SAP_DE_train_raw.csv", index=False)
test.to_csv("data/raw/test/SAP_DE_test_raw.csv", index=False)

print(f"Train: {len(train)} rows | {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Test:  {len(test)} rows  | {test['date'].min().date()} to {test['date'].max().date()}")

def preprocess(df_in):
    df_out = df_in.copy()
    df_out.set_index("date", inplace=True)
    volume_zero = (df_out["volume"] == 0).sum()
    df_out["volume"] = df_out["volume"].replace(0, pd.NA)
    df_out[["open", "high", "low", "close"]] = df_out[["open", "high", "low", "close"]].ffill()
    df_out["volume"] = df_out["volume"].ffill()
    df_out.dropna(inplace=True)
    assert df_out.isnull().sum().sum() == 0, "Nulls remain after preprocessing!"
    return df_out, volume_zero

train_proc, train_zero_vol = preprocess(train)
test_proc,  test_zero_vol  = preprocess(test)

Path("data/processed/train").mkdir(parents=True, exist_ok=True)
Path("data/processed/test").mkdir(parents=True, exist_ok=True)

train_proc.to_csv("data/processed/train/SAP_DE_train.csv")
test_proc.to_csv("data/processed/test/SAP_DE_test.csv")

print(f"Processed Train: {train_proc.shape} | Nulls: {train_proc.isnull().sum().sum()}")
print(f"Processed Test:  {test_proc.shape}  | Nulls: {test_proc.isnull().sum().sum()}")
print(f"Zero-volume rows forward-filled - Train: {train_zero_vol}, Test: {test_zero_vol}")
print(f"NO OVERLAP: Train max {train_proc.index.max()} < Test min {test_proc.index.min()}")