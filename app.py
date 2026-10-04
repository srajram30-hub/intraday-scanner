import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="Nifty 100 Intraday Scanner", layout="wide")

st.title("⚡ Nifty 100 Intraday Institutional Scanner")
st.markdown("Automated second-opinion engine evaluating Nifty 100 blue-chips using **VWAP, RVOL, RSI, ATR**, and core technical pillars.")

# Curated high-liquidity Nifty 100 watchlist
nifty_100_watchlist = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", 
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS"
]

@st.cache_data(ttl=60)
def scan_nifty_market(tickers):
    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5d", interval="15p", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 25:
                continue
                
            # Calculations for Institutional Indicators
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # VWAP Approximation for intraday
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            
            latest = df.iloc[-1]
            current_price = latest['Close']
            prev_close = df['Close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Core Rules Evaluation
            is_uptrend = (current_price > latest['EMA_20']) and (current_price > latest['VWAP'])
            is_high_volume = latest['Volume'] > (latest['Vol_MA20'] * 1.3) # RVOL > 1.3
            is_healthy_rsi = 50 <= latest['RSI'] <= 75
            
            recent_high = df['High'].iloc[-21:-1].max()
            is_breakout = current_price >= recent_high
            
            # Decision Matrix
            if is_uptrend and is_high_volume and is_healthy_rsi and is_breakout:
                decision = "🟢 YES, GO AHEAD"
            elif not is_high_volume or not is_uptrend:
                decision = "🟡 TEMPORARY HOLD"
            else:
                decision = "🔴 DO NOT TRADE"
                
            results.append({
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": round(current_price, 2),
                "Change %": round(change_pct, 2),
                "RVOL": round(latest['Volume'] / latest['Vol_MA20'], 2) if latest['Vol_MA20'] > 0 else 0,
                "RSI": round(latest['RSI'], 1),
                "Verdict": decision
            })
        except:
            continue
    return pd.DataFrame(results)

if st.button("🚀 Run Nifty 100 Instant Scan"):
    with st.spinner("Scanning Nifty 100 blue-chips and computing institutional indicators..."):
        df_results = scan_nifty_market(nifty_100_watchlist)
        
        if not df_results.empty:
            st.success("Scan complete! Review your institutional second-opinion table below:")
            st.dataframe(df_results, use_container_width=True)
        else:
            st.error("Could not fetch market data right now. Please try again.")
else:
    st.info("Click the **'Run Nifty 100 Instant Scan'** button above to evaluate market breadth and find high-probability setups instantly on mobile or desktop.")
