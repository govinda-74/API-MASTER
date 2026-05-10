import pandas as pd
import pandas_market_calendars as mcal
from pathlib import Path

def filter_trading_days(df, start_date, end_date):
    xetra = mcal.get_calendar("XETR")
    schedule = xetra.schedule(start_date=start_date, end_date=end_date)
    valid_days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    return df[df["date"].isin(valid_days)]

def preprocess(df_in):
    df_out = df_in.copy().set_index("date")
    df_out["volume"] = df_out["volume"].replace(0, pd.NA)
    df_out[["open", "high", "low", "close"]] = df_out[["open", "high", "low", "close"]].ffill()
    df_out["volume"] = df_out["volume"].ffill()
    df_out.dropna(inplace=True)
    assert df_out.isnull().sum().sum() == 0, "Nulls remain!"
    return df_out

df = pd.read_csv("data/raw/train/SAP_DE_historical.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")

df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None).dt.normalize()
df.drop(columns=["dividends", "stock_splits"], errors="ignore", inplace=True)
df.sort_values("date", inplace=True)

train = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2024-12-31")].copy()
test  = df[df["date"] >= "2025-01-01"].copy()

train = filter_trading_days(train, "2015-01-01", "2024-12-31")
test  = filter_trading_days(test,  "2025-01-01", str(df["date"].max().date()))

Path("data/raw/train").mkdir(parents=True, exist_ok=True)
Path("data/raw/test").mkdir(parents=True, exist_ok=True)
train.to_csv("data/raw/train/SAP_DE_train_raw.csv", index=False)
test.to_csv("data/raw/test/SAP_DE_test_raw.csv",   index=False)

train_proc = preprocess(train)
test_proc  = preprocess(test)

Path("data/processed/train").mkdir(parents=True, exist_ok=True)
Path("data/processed/test").mkdir(parents=True, exist_ok=True)
train_proc.to_csv("data/processed/train/SAP_DE_train.csv")
test_proc.to_csv("data/processed/test/SAP_DE_test.csv")

print(f"Train: {train_proc.shape} | {train_proc.index.min().date()} to {train_proc.index.max().date()}")
print(f"Test:  {test_proc.shape}  | {test_proc.index.min().date()} to {test_proc.index.max().date()}")
print(f"Train contains 2015-01-01: {pd.Timestamp('2015-01-01') in train_proc.index}")
print(f"Test  contains 2025-01-01: {pd.Timestamp('2025-01-01') in test_proc.index}")