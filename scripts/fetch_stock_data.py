import yfinance as yf
import pandas as pd
from pathlib import Path

symbol = "SAP.DE"
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{symbol.replace('.', '_')}_historical.csv"

print(f"Fetching historical data for {symbol}...")
ticker = yf.Ticker(symbol)
df = ticker.history(period="max")

if df.empty:
    print("ERROR: No data fetched. Check symbol or internet connection.")
else:
    print(f"Fetched {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    df.to_csv(output_file)
    print(f"Saved to {output_file}")
