import requests
import pandas as pd
from pathlib import Path

API_KEY = "WW14CB9SM5Y829DL"

symbol = "SAP.DE"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"

print(f"Fetching data for {symbol} from Alpha Vantage...")
response = requests.get(url)
data = response.json()

if "Time Series (Daily)" not in data:
    print("ERROR:", data.get("Note", data.get("Error Message", "Unknown error")))
    exit()

time_series = data["Time Series (Daily)"]

df = pd.DataFrame(time_series).T
df.index.name = "Date"
df.columns = [c.split(". ")[1] for c in df.columns]
df = df.astype(float)
df = df.sort_index()

print(f"Fetched {len(df)} rows")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{symbol.replace('.', '_')}_historical.csv"
df.to_csv(output_file)
print(f"Saved to {output_file}")
print(df.head())
print(df.tail())