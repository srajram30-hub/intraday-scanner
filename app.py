import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Intraday Trading Automated Scanner", layout="wide")

st.title("🚀 Automated Intraday Trading Decision Engine")
st.markdown("This app fetches live market data and **automatically evaluates** your 4 technical pillars using Python math—no manual inputs required!")

# Sidebar watchlist selection
st.sidebar.header("🔍 Watchlist")
stock_symbol = st.sidebar.selectbox("Select Stock", ["RELIANCE.NS", "TCS.NS", "SBIN.NS", "BHARTIARTL.NS", "HCLTECH.NS"])

@st.cache_data(ttl=300)
def fetch_market_data(ticker):
    # Fetching intraday data (5-day, 15-minute interval)
    df = yf.download(ticker, period="5d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    data = fetch_market_data(stock_symbol)
    if data.empty:
        st.error("Could not fetch data for this symbol. Please try another.")
    else:
        # Calculate Technical Indicators
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['Vol_MA20'] = data['Volume'].rolling(window=20).mean()
        
        latest = data.iloc[-1]
        current_price = latest['Close']
        prev_close = data['Close'].iloc[-2]
        change_pct = ((current_price - prev_close) / prev_close) * 100

        st.sidebar.markdown(f"**Live Price:** ₹{current_price:,.2f}")
        st.sidebar.markdown(f"**Change %:** {change_pct:+.2f}%")

        # --- AUTOMATED 4-PILLAR EVALUATION LOGIC ---
        
        # 1. Candlestick Anatomy: Strong Bullish Body (Close > Open and body > 50% of candle range)
        candle_body = abs(latest['Close'] - latest['Open'])
        candle_range = latest['High'] - latest['Low']
        is_strong_candle = (latest['Close'] > latest['Open']) and (candle_range > 0 and (candle_body / candle_range) > 0.5)
        
        # 2. Support / Resistance Breakout: Breaking above recent 20-period high
        recent_high = data['High'].iloc[-21:-1].max()
        is_breakout = latest['Close'] > recent_high
        
        # 3. Market Trend: Price above 20 EMA and EMA sloping up
        is_uptrend = (latest['Close'] > latest['EMA_20']) and (data['EMA_20'].iloc[-1] > data['EMA_20'].iloc[-5])
        
        # 4. Volume Confirmation: Current volume higher than 20-period volume average
        is_high_volume = latest['Volume'] > latest['Vol_MA20']

        # Display Automated Pillar Results
        st.subheader(f"📊 Automated Technical Breakdown: {stock_symbol}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**1. Candlestick Anatomy:** {'🟢 Strong Momentum' if is_strong_candle else '🔴 Weak / Indecisive'}")
            st.markdown(f"**2. Support / Resistance:** {'🟢 Clean Breakout' if is_breakout else '🔴 Stuck / No Breakout'}")
        
        with col2:
            st.markdown(f"**3. Market Trend:** {'🟢 Strong Uptrend' if is_uptrend else '🔴 Choppy / Downtrend'}")
            st.markdown(f"**4. Volume Confirmation:** {'🟢 High & Confirmed' if is_high_volume else '🔴 Low Volume'}")

        st.divider()

        # Final Decision Engine
        if is_strong_candle and is_breakout and is_uptrend and is_high_volume:
            decision = "🟢 YES, GO AHEAD"
            action = "Enter on close / retest; Set SL below breakout candle; Target 1:2+ R:R; Trail with 20 EMA"
            box_type = "success"
        elif not is_high_volume or not is_uptrend:
            decision = "🟡 TEMPORARY HOLD / WAIT"
            action = "Stand aside; wait for range expansion or volume surge; Set price alerts"
            box_type = "warning"
        else:
            decision = "🔴 DO NOT TRADE"
            action = "No entry; Protect capital; Avoid trading against trend/rejection"
            box_type = "error"

        st.subheader("🎯 Final Automated Trading Decision")
        if box_type == "success":
            st.success(f"**Decision:** {decision}")
        elif box_type == "warning":
            st.warning(f"**Decision:** {decision}")
        else:
            st.error(f"**Decision:** {decision}")

        st.info(f"**Risk Action Plan:** {action}")

except Exception as e:
    st.error(f"Error loading market data: {e}")
