import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Intraday Trading Command Center", layout="wide")

st.title("🚀 Intraday Trading Decision Matrix & Automated Scanner")
st.markdown("This web app runs Python in the cloud to evaluate your 4 technical pillars and provide live trade decisions.")

# Sidebar for manual override or live stock testing
st.sidebar.header("🔍 Stock Parameter Inputs")
stock_symbol = st.sidebar.selectbox("Select Watchlist Stock", ["RELIANCE.NS", "TCS.NS", "SBIN.NS", "BHARTIARTL.NS", "HCLTECH.NS"])

# Fetch live price using yfinance (as a cloud-friendly alternative to GoogleFinance)
try:
    ticker_data = yf.Ticker(stock_symbol)
    todays_data = ticker_data.history(period="1d")
    current_price = todays_data['Close'].iloc[-1]
    prev_close = ticker_data.info.get('previousClose', current_price)
    change_pct = ((current_price - prev_close) / prev_close) * 100
except:
    current_price = 0.0
    change_pct = 0.0

st.sidebar.markdown(f"**Live Price:** ₹{current_price:,.2f}")
st.sidebar.markdown(f"**Change %:** {change_pct:+.2f}%")

st.divider()

# Interactive Form for the 4 Pillars
st.subheader(f"Analyzing: {stock_symbol}")

col1, col2 = st.columns(2)

with col1:
    candlestick = st.selectbox("1. Candlestick Anatomy", [
        "Strong Breakout / Momentum", 
        "Small Body / Doji / Confused", 
        "Long Upper Wick / Rejection"
    ])
    
    support_resistance = st.selectbox("2. Support / Resistance", [
        "Clean Breakout / Bounce", 
        "Stuck at Zone / Mid-way", 
        "Failed Breakout / Resistance Hold"
    ])

with col2:
    trend = st.selectbox("3. Market Trend", [
        "Strong Trend (Aligned)", 
        "Sideways / Choppy", 
        "Downtrend or Divergence"
    ])
    
    volume = st.selectbox("4. Volume Confirmation", [
        "High Volume (Confirmed)", 
        "Low Volume (Fakeout Risk)", 
        "High Selling Volume"
    ])

st.divider()

# Automated Logic Evaluation (The Decision Engine)
if candlestick == "Strong Breakout / Momentum" and support_resistance == "Clean Breakout / Bounce" and trend == "Strong Trend (Aligned)" and volume == "High Volume (Confirmed)":
    decision = "🟢 YES, GO AHEAD"
    action = "Enter on close / retest; Set SL below breakout candle; Target 1:2+ R:R; Trail with 20 EMA"
    box_color = "success"
elif volume == "Low Volume (Fakeout Risk)" or trend == "Sideways / Choppy":
    decision = "🟡 TEMPORARY HOLD / WAIT"
    action = "Stand aside; wait for range expansion or volume surge; Set price alerts"
    box_color = "warning"
else:
    decision = "🔴 DO NOT TRADE"
    action = "No entry; Protect capital; Avoid trading against trend/rejection"
    box_color = "error"

# Display Final Output
st.subheader("📊 Automated Evaluation Result")
if box_color == "success":
    st.success(f"**Final Trading Decision:** {decision}")
elif box_color == "warning":
    st.warning(f"**Final Trading Decision:** {decision}")
else:
    st.error(f"**Final Trading Decision:** {decision}")

st.info(**Risk Action Plan:** {action})
