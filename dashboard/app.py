import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SAP.DE Financial Dashboard", layout="wide")


# ── Indicator calculation (used for live data) ───────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["sma_20"]  = df["close"].rolling(20).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_12"]  = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]  = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]         = df["ema_12"] - df["ema_26"]
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df["rsi_14"]   = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    df["bb_mid"]   = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df.dropna().reset_index(drop=True)


# ── Data loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_historical() -> pd.DataFrame:
    df = pd.read_csv("data/processed/train/SAP_DE_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_live(period: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        raw = yf.download("SAP.DE", period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return None, "yfinance returned no data. Market may be closed or rate limited."
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        raw.index.name = "date"
        df = raw.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        df = compute_indicators(df)
        return df, None
    except Exception as exc:
        return None, str(exc)


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("Controls")

st.sidebar.markdown("**Data Source**")
data_mode = st.sidebar.radio(
    "Select mode",
    ["Historical (2015–2024)", "Live (yfinance)"],
    index=0,
)

live_period = "1y"
auto_refresh = False

if data_mode == "Live (yfinance)":
    live_period  = st.sidebar.selectbox("Lookback period", ["3mo", "6mo", "1y", "2y"], index=2)
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 min", value=False)
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Date Range**")

# ── Load data ─────────────────────────────────────────────────────────────
if data_mode == "Historical (2015–2024)":
    df = load_historical()
    data_label = "Historical CSV (2015–2024)"
    updated_at = None
else:
    with st.spinner("Fetching live data from yfinance…"):
        df, err = fetch_live(live_period)
    if err or df is None:
        st.error(f"Live data fetch failed: {err}")
        st.info("Falling back to historical CSV.")
        df = load_historical()
        data_label = "Fallback: Historical CSV"
        updated_at = None
    else:
        data_label = f"Live yfinance — {live_period} period"
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

min_date     = df["date"].min().date()
max_date     = df["date"].max().date()
default_start = (df["date"].max() - pd.Timedelta(days=730)).date()
default_start = max(default_start, min_date)

start_date = st.sidebar.date_input("Start", value=default_start,
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End",   value=max_date,
                                   min_value=min_date, max_value=max_date)

st.sidebar.markdown("---")
st.sidebar.markdown("**Moving Averages**")
show_sma20  = st.sidebar.checkbox("SMA 20",  value=True)
show_sma50  = st.sidebar.checkbox("SMA 50",  value=True)
show_sma200 = st.sidebar.checkbox("SMA 200", value=True)
show_ema12  = st.sidebar.checkbox("EMA 12",  value=False)
show_ema26  = st.sidebar.checkbox("EMA 26",  value=False)
show_bb     = st.sidebar.checkbox("Bollinger Bands", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Indicators**")
show_rsi  = st.sidebar.checkbox("RSI (14)", value=True)
show_macd = st.sidebar.checkbox("MACD",     value=True)

# ── Filter ────────────────────────────────────────────────────────────────
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
dff  = df[mask].copy()

if dff.empty:
    st.error("No data in selected date range. Please adjust the dates.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────
st.title("SAP SE (SAP.DE) — Financial Analytics Dashboard")
caption_parts = [
    f"Market: Frankfurt / XETRA",
    f"Source: {data_label}",
    f"Range: {start_date} to {end_date}",
    f"{len(dff)} trading days",
]
if updated_at:
    caption_parts.append(f"Last updated: {updated_at}")
st.caption("  |  ".join(caption_parts))

# ── Key Metrics ───────────────────────────────────────────────────────────
latest        = dff.iloc[-1]
prev          = dff.iloc[-2] if len(dff) > 1 else latest
daily_ret_pct = latest["daily_return"] * 100
period_high   = dff["high"].max()
period_low    = dff["low"].min()
avg_vol_20    = dff["volume"].tail(20).mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close Price",   f"€{latest['close']:.2f}",
          delta=f"{latest['close'] - prev['close']:.2f}")
c2.metric("Daily Return",  f"{daily_ret_pct:.2f}%", delta=f"{daily_ret_pct:.2f}%")
c3.metric("Period High",   f"€{period_high:.2f}")
c4.metric("Period Low",    f"€{period_low:.2f}")
c5.metric("Avg Vol (20d)", f"{avg_vol_20/1e6:.2f}M")

st.markdown("---")

# ── Chart 1: Candlestick + Moving Averages ────────────────────────────────
fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=dff["date"],
    open=dff["open"], high=dff["high"], low=dff["low"], close=dff["close"],
    name="SAP.DE",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
))
if show_sma20:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["sma_20"],
        name="SMA 20",  line=dict(color="orange",       width=1.2)))
if show_sma50:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["sma_50"],
        name="SMA 50",  line=dict(color="royalblue",    width=1.2)))
if show_sma200:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["sma_200"],
        name="SMA 200", line=dict(color="red",          width=1.5)))
if show_ema12:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["ema_12"],
        name="EMA 12",  line=dict(color="mediumpurple", width=1, dash="dot")))
if show_ema26:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["ema_26"],
        name="EMA 26",  line=dict(color="mediumseagreen", width=1, dash="dot")))
if show_bb:
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["bb_upper"],
        name="BB Upper", line=dict(color="gray", width=1, dash="dash")))
    fig_price.add_trace(go.Scatter(x=dff["date"], y=dff["bb_lower"],
        name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
fig_price.update_layout(
    title="Price Chart", height=500, template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_price, use_container_width=True)

# ── Chart 2: Volume ───────────────────────────────────────────────────────
vol_colors = ["#26a69a" if r >= 0 else "#ef5350" for r in dff["daily_return"]]
fig_vol = go.Figure(go.Bar(x=dff["date"], y=dff["volume"],
                           marker_color=vol_colors, name="Volume"))
fig_vol.update_layout(title="Volume", height=200,
                      template="plotly_dark", showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

# ── Chart 3: RSI ──────────────────────────────────────────────────────────
if show_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=dff["date"], y=dff["rsi_14"],
        name="RSI 14", line=dict(color="gold", width=1.5)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Overbought (70)", annotation_position="right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="limegreen",
                      annotation_text="Oversold (30)",   annotation_position="right")
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red",       opacity=0.05, line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="limegreen", opacity=0.05, line_width=0)
    fig_rsi.update_layout(title="RSI (14)", height=250, template="plotly_dark",
                          showlegend=False, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

# ── Chart 4: MACD ─────────────────────────────────────────────────────────
if show_macd:
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dff["macd_hist"]]
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=dff["date"], y=dff["macd_hist"],
        marker_color=hist_colors, name="Histogram"))
    fig_macd.add_trace(go.Scatter(x=dff["date"], y=dff["macd"],
        line=dict(color="royalblue", width=1.5), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=dff["date"], y=dff["macd_signal"],
        line=dict(color="orange", width=1.5), name="Signal"))
    fig_macd.update_layout(
        title="MACD", height=250, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data source: yfinance  |  SAP SE (SAP.DE)  |  Frankfurt / XETRA")

# ── Auto-refresh ──────────────────────────────────────────────────────────
if auto_refresh:
    st_autorefresh(interval=300_000, key="live_data_autorefresh")
