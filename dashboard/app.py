import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="SAP.DE Financial Dashboard",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/train/SAP_DE_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
st.sidebar.markdown("**Date Range**")

min_date = df["date"].min().date()
max_date = df["date"].max().date()
default_start = (df["date"].max() - pd.Timedelta(days=730)).date()

start_date = st.sidebar.date_input("Start", value=default_start, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End",   value=max_date,      min_value=min_date, max_value=max_date)

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

# ── Filter ─────────────────────────────────────────────────────────────
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
dff  = df[mask].copy()

if dff.empty:
    st.error("No data in selected date range. Please adjust the dates.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────
st.title("SAP SE (SAP.DE) — Financial Analytics Dashboard")
st.caption(
    f"Market: Frankfurt / XETRA  |  "
    f"Data: {start_date} to {end_date}  |  "
    f"{len(dff)} trading days"
)

# ── Key Metrics ─────────────────────────────────────────────────────────
latest       = dff.iloc[-1]
daily_ret_pct = latest["daily_return"] * 100
period_high  = dff["high"].max()
period_low   = dff["low"].min()
avg_vol_20   = dff["volume"].tail(20).mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close Price",    f"€{latest['close']:.2f}")
c2.metric("Daily Return",   f"{daily_ret_pct:.2f}%", delta=f"{daily_ret_pct:.2f}%")
c3.metric("Period High",    f"€{period_high:.2f}")
c4.metric("Period Low",     f"€{period_low:.2f}")
c5.metric("Avg Vol (20d)",  f"{avg_vol_20/1e6:.2f}M")

st.markdown("---")

# ── Chart 1: Candlestick + Moving Averages ──────────────────────────────
fig_price = go.Figure()

fig_price.add_trace(go.Candlestick(
    x=dff["date"],
    open=dff["open"], high=dff["high"], low=dff["low"], close=dff["close"],
    name="SAP.DE",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350"
))

if show_sma20:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["sma_20"], name="SMA 20",
        line=dict(color="orange", width=1.2)
    ))
if show_sma50:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["sma_50"], name="SMA 50",
        line=dict(color="royalblue", width=1.2)
    ))
if show_sma200:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["sma_200"], name="SMA 200",
        line=dict(color="red", width=1.5)
    ))
if show_ema12:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["ema_12"], name="EMA 12",
        line=dict(color="mediumpurple", width=1, dash="dot")
    ))
if show_ema26:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["ema_26"], name="EMA 26",
        line=dict(color="mediumseagreen", width=1, dash="dot")
    ))
if show_bb:
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["bb_upper"], name="BB Upper",
        line=dict(color="gray", width=1, dash="dash")
    ))
    fig_price.add_trace(go.Scatter(
        x=dff["date"], y=dff["bb_lower"], name="BB Lower",
        line=dict(color="gray", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(128,128,128,0.1)"
    ))

fig_price.update_layout(
    title="Price Chart",
    height=500,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_price, use_container_width=True)

# ── Chart 2: Volume ─────────────────────────────────────────────────────
vol_colors = ["#26a69a" if r >= 0 else "#ef5350" for r in dff["daily_return"]]
fig_vol = go.Figure(go.Bar(
    x=dff["date"], y=dff["volume"],
    marker_color=vol_colors, name="Volume"
))
fig_vol.update_layout(
    title="Volume", height=200,
    template="plotly_dark", showlegend=False
)
st.plotly_chart(fig_vol, use_container_width=True)

# ── Chart 3: RSI ────────────────────────────────────────────────────────
if show_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=dff["date"], y=dff["rsi_14"],
        name="RSI 14", line=dict(color="gold", width=1.5)
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Overbought (70)", annotation_position="right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="limegreen",
                      annotation_text="Oversold (30)", annotation_position="right")
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red",       opacity=0.05, line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="limegreen", opacity=0.05, line_width=0)
    fig_rsi.update_layout(
        title="RSI (14)", height=250,
        template="plotly_dark", showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ── Chart 4: MACD ───────────────────────────────────────────────────────
if show_macd:
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dff["macd_hist"]]
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(
        x=dff["date"], y=dff["macd_hist"],
        marker_color=hist_colors, name="Histogram"
    ))
    fig_macd.add_trace(go.Scatter(
        x=dff["date"], y=dff["macd"],
        line=dict(color="royalblue", width=1.5), name="MACD"
    ))
    fig_macd.add_trace(go.Scatter(
        x=dff["date"], y=dff["macd_signal"],
        line=dict(color="orange", width=1.5), name="Signal"
    ))
    fig_macd.update_layout(
        title="MACD", height=250,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data source: yfinance  |  SAP SE (SAP.DE)  |  Frankfurt / XETRA")