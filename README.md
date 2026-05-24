# SAP.DE Real-Time Financial Analytics Dashboard

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://stockmarketlivedata.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Models-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **End-to-end data science project** — historical + real-time market data ingestion, time-series feature engineering, three machine-learning models (Linear Regression, Random Forest, LSTM), and a production-deployed interactive analytics dashboard for **SAP SE (SAP.DE, Frankfurt XETRA)**.

**🌐 Live Demo:** **https://stockmarketlivedata.streamlit.app/**
**💻 Repository:** https://github.com/govinda-74/API-MASTER

---

## Why this project matters

This project goes beyond "load a CSV, train a model, plot a chart". It demonstrates the full data-science workflow recruiters look for:

- **Data integrity first** — caught a hidden train/test leakage (1-day timezone offset between two data sources) before model training, and re-engineered the split from a single source with XETRA-calendar-aware filtering.
- **Honest model evaluation** — explicitly diagnoses *why* Linear Regression scores R² = 0.9997 (trivial same-day OHLC identity), *why* Random Forest fails to extrapolate (R² = −0.49) and *why* LSTM at R² = 0.857 is the most credible result.
- **Real-time + historical, with graceful fallback** — dashboard reads live yfinance data when available and silently falls back to the bundled CSV when the API rate-limits or fails.
- **Production-deployed** — pinned dependencies, dark-themed Streamlit config, non-blocking auto-refresh (`streamlit-autorefresh`), `.gitignore` hygiene, live on Streamlit Community Cloud.

---

## Live Demo

Click anywhere on the screenshot or use the link below to open the deployed app.

[![Open the Live Dashboard](https://img.shields.io/badge/Open%20Live%20Dashboard-stockmarketlivedata.streamlit.app-26a69a?style=for-the-badge)](https://stockmarketlivedata.streamlit.app/)

Once open, you can:

1. Toggle **Historical (2015–2024)** vs **Live (yfinance)** in the sidebar
2. Switch the lookback window (3mo / 6mo / 1y / 2y) for live data
3. Overlay **SMA 20 / 50 / 200**, **EMA 12 / 26**, **Bollinger Bands**
4. Inspect **RSI (14)** and **MACD** subplots
5. Hit **🔄 Refresh Now** or enable **auto-refresh every 5 min** (non-blocking)

---

## Tech Stack

| Layer                    | Tools                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| **Language**             | Python 3.11                                                                            |
| **Data Engineering**     | Pandas, NumPy, `pandas-market-calendars` (XETRA trading-day filter)                    |
| **Data Sources**         | yfinance (primary), Alpha Vantage (initial fetch)                                      |
| **Visualisation**        | Plotly (interactive), Matplotlib (static EDA / ML plots)                               |
| **Machine Learning**     | scikit-learn (Linear Regression, Random Forest, MinMaxScaler), TensorFlow / Keras (LSTM) |
| **Dashboard**            | Streamlit + `streamlit-autorefresh`                                                    |
| **Deployment**           | Streamlit Community Cloud                                                              |
| **Versioning / Workflow**| Git, GitHub                                                                            |

---

## Project Pipeline

```
 ┌─────────────────┐    ┌────────────────────┐    ┌───────────────────┐
 │  yfinance API   │ ─► │  fetch_stock_data  │ ─► │  raw OHLCV CSV    │
 └─────────────────┘    └────────────────────┘    └─────────┬─────────┘
                                                            │
                                                            ▼
                                              ┌──────────────────────────┐
                                              │  split_data.py           │
                                              │  · UTC → Europe/Berlin   │
                                              │  · XETRA-calendar filter │
                                              │  · 10-yr train / 1-yr+ test
                                              └────────────┬─────────────┘
                                                           │
                                                           ▼
                                              ┌──────────────────────────┐
                                              │ preprocess_data.py       │
                                              │ · zero-volume ffill      │
                                              │ · null assertions        │
                                              └────────────┬─────────────┘
                                                           │
                                                           ▼
                                              ┌──────────────────────────┐
                                              │ technical_indicators.py  │
                                              │ SMA · EMA · MACD · RSI   │
                                              │ Bollinger Bands · returns│
                                              │ → SAP_DE_features.csv    │
                                              │   (2 338 × 19)           │
                                              └────────────┬─────────────┘
                                                           │
                                ┌──────────────────────────┼──────────────────────────┐
                                ▼                          ▼                          ▼
                       ┌────────────────┐          ┌──────────────┐          ┌────────────────┐
                       │ train_model.py │          │ prediction.py│          │  dashboard/    │
                       │ LR · RF · LSTM │          │ 5 eval plots │          │  app.py        │
                       └────────┬───────┘          └──────────────┘          │  Plotly +      │
                                ▼                                            │  Streamlit     │
                       ┌────────────────┐                                    └──────┬─────────┘
                       │ models/*.pkl   │                                           │
                       │ models/*.keras │                                           ▼
                       └────────────────┘                              Streamlit Community Cloud
                                                                       (live + historical mode)
```

---

## Model Results (Validation Set, 468 rows)

| Model                 | MAE       | RMSE      | R²       | Verdict                                              |
| --------------------- | --------- | --------- | -------- | ---------------------------------------------------- |
| **Linear Regression** | **0.470** | **0.640** | **0.9997** | ⚠ Near-trivial — see leakage discussion below       |
| Random Forest         | 30.809    | 44.528    | −0.487   | ⚠ Cannot extrapolate beyond training price range    |
| **LSTM (60-day window)** | **9.036** | **13.818** | **0.857** | ✅ Most honest temporal forecast                    |

### Why I report all three honestly

A common portfolio mistake is to lead with "R² = 0.9997" and call it a success. It isn't, and any data-science interviewer will see through it:

1. **Linear Regression (R² = 0.9997)** — the feature set includes same-day `open`, `high`, `low`. Because `close` is mathematically bounded by `[low, high]` of the same day, LR effectively learns a same-day identity, not a forecast. Removing OHL would crater this number — and that's the *correct* finding.
2. **Random Forest (R² = −0.487)** — SAP.DE grew from ~€50 to ~€230 over the training window. The validation set sits near the all-time highs, but tree ensembles **cannot extrapolate beyond the range seen in training**, so RF underpredicts and scores worse than predicting the mean. This is a textbook tree-regressor failure mode and a teaching moment, not a bug.
3. **LSTM (R² = 0.857)** — uses a true 60-day rolling window with `EarlyStopping` and the scaler fit **only on the training portion**. Higher error than LR, but it's the only model genuinely forecasting from temporal patterns rather than peeking at same-day OHL.

This kind of analysis is what separates a junior portfolio from a hireable one.

---

## Key Engineering Decisions

| Decision                                           | Why it matters                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------- |
| Single-source train/test split from yfinance       | Eliminated a 1-day timezone-offset leakage from mixing yfinance + Alpha Vantage |
| `Europe/Berlin` tz conversion before date stripping| Aligns date labels to the actual XETRA trading session, not UTC                 |
| XETRA trading-calendar filter (`pandas-market-calendars`) | Drops accidental non-trading days (e.g. 2015-01-01, 2025-01-01) introduced by tz shifts |
| Forward-fill on zero-volume rows                   | Preserves data instead of dropping rows with valid OHLC                         |
| Time-ordered 80/20 split (no shuffle)              | Standard for time-series; prevents look-ahead leakage                           |
| `MinMaxScaler` fit on **train only**               | Prevents validation statistics from leaking into the scaler                     |
| 5-min `@st.cache_data(ttl=300)` on live fetch      | Throttles yfinance calls and keeps the deployed app responsive                  |
| Mandatory historical fallback if `yfinance` fails  | Public app stays usable when Yahoo rate-limits the Streamlit Cloud IP           |
| `streamlit-autorefresh` instead of `time.sleep`    | Non-blocking client-side rerun; safe for multi-user deployment                  |
| Pinned `requirements.txt`                          | Reproducible builds on Streamlit Cloud (no silent breakage from new releases)   |

---

## Repository Structure

```
api-master/
├── dashboard/
│   └── app.py                       # Streamlit app (dual-mode: Historical / Live)
├── data/
│   ├── raw/                         # Original yfinance pulls (CSV)
│   └── processed/
│       ├── train/
│       │   ├── SAP_DE_train.csv     # Cleaned 2015–2024 OHLCV (2 537 rows)
│       │   └── SAP_DE_features.csv  # +14 engineered indicators (2 338 × 19)
│       └── test/
│           └── SAP_DE_test.csv      # 2025+ hold-out (341 rows)
├── scripts/
│   ├── fetch_stock_data.py          # Alpha Vantage initial historical fetch
│   ├── split_data.py                # tz-correct, XETRA-aware train/test split
│   ├── preprocess_data.py           # Cleaning + null assertions
│   ├── technical_indicators.py      # SMA, EMA, MACD, RSI, Bollinger Bands
│   ├── train_model.py               # LR + RF + LSTM, metrics, artefacts
│   └── prediction.py                # Actual-vs-predicted + feature-importance plots
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── lstm_model.keras
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
├── outputs/
│   ├── eda/                         # 4 EDA charts (PNG)
│   └── models/                      # 5 prediction charts + metrics CSV
├── .streamlit/
│   └── config.toml                  # Dark theme, headless, telemetry off
├── requirements.txt                 # Pinned versions for reproducible deploy
└── README.md
```

---

## Visual Outputs

### EDA — `outputs/eda/`

| File                          | Insight                                                                |
| ----------------------------- | ---------------------------------------------------------------------- |
| `01_closing_price.png`        | Long-term uptrend ~€50 → ~€230 over 2015-2024                          |
| `02_volume.png`               | Volume spikes coincide with major macro events (COVID-19 March 2020)   |
| `03_daily_returns.png`        | Approximately normal returns with fat tails on shock days              |
| `04_correlation_heatmap.png`  | OHLC mutually ~1.0 correlated; volume weakly negatively correlated     |

### Model evaluation — `outputs/models/`

| File                                  | Insight                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `01_lr_actual_vs_predicted.png`       | LR overlay sits on top of actuals — the leakage tell                      |
| `02_rf_actual_vs_predicted.png`       | RF plateaus under the price range it never saw in training                |
| `03_lstm_actual_vs_predicted.png`     | LSTM tracks trend with realistic lag and bounded error                    |
| `04_model_comparison.png`             | All three models on one axis with R² in the legend                        |
| `05_rf_feature_importance.png`        | `low` (72 %) + `high` (26 %) dominate — confirms the OHL-leakage diagnosis |

---

## Run It Locally

```bash
# 1. Clone
git clone https://github.com/govinda-74/API-MASTER.git
cd API-MASTER

# 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. (Optional) regenerate the data + models end-to-end
python scripts/fetch_stock_data.py
python scripts/split_data.py
python scripts/preprocess_data.py
python scripts/technical_indicators.py
python scripts/train_model.py
python scripts/prediction.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
# → opens http://localhost:8501
```

All raw data, processed CSVs, models and outputs are committed, so step 5 works immediately even if you skip step 4.

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.

- **Live app:** https://stockmarketlivedata.streamlit.app/
- **Main file path:** `dashboard/app.py`
- **Python version:** 3.11
- **Redeploy:** any push to `main` triggers an automatic rebuild

### Deployment safety notes

- `requirements.txt` is fully pinned to avoid silent breakage from new package releases
- `streamlit-autorefresh` replaces `time.sleep`-based polling, so the worker thread is never blocked
- The dashboard always falls back to the bundled historical CSV if yfinance is rate-limited or unreachable from the Streamlit Cloud egress IP
- No secrets are committed; `.streamlit/secrets.toml` is gitignored even though the project currently needs none

---

## What I Learned / What This Project Demonstrates

For recruiters and hiring managers, here is what this project shows about how I work:

- **Data engineering discipline** — timezone normalisation, market-calendar filtering, null-safe preprocessing, and assertion-based contracts in the pipeline
- **Feature engineering** — implemented SMA, EMA, MACD, RSI and Bollinger Bands from first principles (no TA-Lib dependency)
- **Classical ML + deep learning side-by-side** — Linear Regression, Random Forest (n_estimators=200, max_depth=10), and a two-layer LSTM (64 → 32, dropout 0.2, EarlyStopping)
- **Honest model interpretation** — diagnosed leakage in LR, extrapolation failure in RF, and explained the LSTM trade-off (higher MAE, but the only genuinely forecasting model)
- **Production thinking** — pinned deps, non-blocking refresh, graceful API fallback, dark-themed responsive UI, headless server config, public deployment
- **End-to-end ownership** — from a raw API call to a live URL anyone in the world can open

---

## Possible Future Enhancements

- Multi-ticker comparison (DAX components)
- Walk-forward cross-validation for the LSTM
- Train LR/RF on **lagged** features only (no same-day OHL) to remove the leakage and get an honest classical baseline
- Add a sentiment overlay from financial news headlines (FinBERT)
- Cache live data in DuckDB / SQLite for cross-session persistence
- FastAPI backend so the same models can serve REST predictions

---

## Author

**Govinda**
GitHub: [@govinda-74](https://github.com/govinda-74)
Project: [API-MASTER](https://github.com/govinda-74/API-MASTER)
Live app: [stockmarketlivedata.streamlit.app](https://stockmarketlivedata.streamlit.app/)

If you're a recruiter, hiring manager, or fellow data person — I'd love to hear what you think. Open an issue or reach out via GitHub.

---

## License

Released under the **MIT License**. Educational / portfolio project — market data belongs to its respective providers (Yahoo Finance via `yfinance`, Alpha Vantage). Not financial advice.

