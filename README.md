# Real-Time Share Market Analytics Dashboard — SAP.DE

## Project Objective

An end-to-end financial data science project that fetches real-time and historical stock market data for **SAP SE (SAP.DE)** via APIs, performs financial analysis and technical indicator calculations, visualises stock trends through an interactive dashboard, and applies machine learning models to predict future stock prices.

## Company

| Field         | Value                      |
|---------------|----------------------------|
| Company Name  | SAP SE                     |
| Stock Symbol  | SAP.DE                     |
| Market        | Germany (Frankfurt / XETRA)|

## Project Goals

1. Retrieve stock market data using API integration
2. Perform financial data preprocessing and analysis
3. Visualise stock market trends interactively
4. Calculate technical indicators
5. Build predictive machine learning models
6. Create a real-time dashboard
7. Deploy the project online

## Technology Stack

| Category            | Tools / Libraries                              |
|---------------------|------------------------------------------------|
| Language            | Python                                         |
| Data Manipulation   | Pandas, NumPy                                  |
| Visualisation       | Matplotlib, Plotly                             |
| Dashboard           | Streamlit                                      |
| Machine Learning    | Scikit-learn, (LSTM via TensorFlow/Keras)      |
| Data Source         | yfinance / Alpha Vantage API                   |
| Version Control     | Git / GitHub                                   |

## Folder Structure

```
share-market-dashboard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── scripts/
│   ├── fetch_stock_data.py
│   ├── preprocess_data.py
│   ├── technical_indicators.py
│   ├── train_model.py
│   └── prediction.py
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── requirements.txt
├── README.md
└── main.py
```

## Project Phases

### Phase 1 — Data Collection
- Connect to financial API (yfinance / Alpha Vantage)
- Fetch historical OHLCV data for SAP.DE (Open, High, Low, Close, Volume)
- Save raw data as CSV

**Output:** Structured DataFrame + historical stock dataset

### Phase 2 — Data Preprocessing
- Handle missing values, convert data types, parse dates
- Sort data chronologically and produce a clean DataFrame

**Output:** Cleaned financial dataset ready for analysis

### Phase 3 — Exploratory Data Analysis (EDA)
- Analyse price trends, volatility, daily returns, and trading volume
- Produce: closing price chart, volume chart, return distribution, correlation heatmap

**Output:** Key insights on trend direction, volatility, and market activity

### Phase 4 — Technical Indicator Engineering
Calculate:
- **SMA** — Simple Moving Average
- **EMA** — Exponential Moving Average
- **RSI** — Relative Strength Index
- **MACD** — Moving Average Convergence Divergence
- **Bollinger Bands**

**Output:** Feature-engineered dataset

### Phase 5 — Dashboard Development (Streamlit)
Features: company selector, interactive stock charts, technical indicator overlays, real-time refresh, volume analysis, date filtering.

Charts: Candlestick, Moving Averages, RSI, Volume

**Output:** Interactive financial analytics dashboard

### Phase 6 — Machine Learning Model
**Target:** Closing price of SAP.DE

**Features:** Open, High, Low, Volume + Technical Indicators

| Level    | Algorithm                 |
|----------|---------------------------|
| Beginner | Linear Regression         |
| Beginner | Random Forest Regressor   |
| Advanced | LSTM Neural Network       |

**Metrics:** MAE, RMSE, R² Score

### Phase 7 — Prediction Visualisation
- Compare actual vs predicted prices
- Plot prediction trends and produce a model evaluation report

### Phase 8 — Real-Time Data Integration
- Auto-fetch latest stock prices and refresh the dashboard periodically
- Handle API rate limits and errors

**Output:** Live-updating financial dashboard

### Phase 9 — Deployment
Deploy to one of: Streamlit Cloud, Render, Railway, or Hugging Face Spaces.
Connect GitHub repository and publish a public project link.

## Final Deliverables

1. Complete GitHub repository
2. Financial analytics dashboard
3. Machine learning prediction system
4. Project documentation
5. Deployment link

## Possible Future Enhancements

- Multi-company comparison
- Portfolio optimisation
- News sentiment analysis
- Risk prediction
- Cryptocurrency support
- FastAPI backend
- Database integration

## Expected Learning Outcomes

- API integration
- Financial data analysis
- Time-series analysis
- Technical indicators
- Dashboard development
- Machine learning forecasting
- End-to-end project deployment

## Resume Description

> Developed a real-time financial analytics dashboard using Python, APIs, Pandas, Plotly, Streamlit, and machine learning to analyse and predict SAP.DE stock market trends using historical and live financial data.
