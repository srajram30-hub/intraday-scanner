import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="Nifty 100 Intraday Scanner", layout="wide")

st.title("⚡ Nifty 100 Intraday Institutional Scanner")
st.markdown("Automated second-opinion engine with **Built-in Risk Management (Stop-Loss & Targets)**.")

nifty_100_watchlist = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", 
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "COALINDIA.NS", "NESTLEIND.NS", "GRASIM.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "CIPLA.NS", "SBILIFE.NS", "BPCL.NS", "TATAMOTORS.NS",
    "WIPRO.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "DIVISLAB.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS",
    "TECHM.NS", "M&M.NS", "HINDUNILVR.NS", "TATACONSUM.NS", "DRREDDY.NS",
    "CROMPTON.NS", "PGEL.NS", "APOLLOHOSP.NS", "TVSMOTOR.NS", "PAYTM.NS", 
    "INDIGO.NS", "LUPIN.NS", "MAXHEALTH.NS", "TRENT.NS"
]

@st.cache_data(ttl=60)
def scan_nifty_market(tickers):
    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 25:
                continue
                
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # VWAP
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            
            # ATR (14) for Risk Management
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            latest = df.iloc[-1]
            current_price = latest['Close']
            prev_close = df['Close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            is_uptrend = (current_price > latest['EMA_20']) and (current_price > latest['VWAP'])
            is_high_volume = latest['Volume'] > (latest['Vol_MA20'] * 1.3)
            is_healthy_rsi = 50 <= latest['RSI'] <= 75
            
            recent_high = df['High'].iloc[-21:-1].max()
            is_breakout = current_price >= recent_high
            
            # Risk Management Levels calculated automatically
            atr_value = latest['ATR'] if not np.isnan(latest['ATR']) else (current_price * 0.005)
            stop_loss = round(current_price - (1.5 * atr_value), 2)
            target_price = round(current_price + (2.5 * atr_value), 2)
            
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
                "Stop-Loss (₹)": stop_loss,
                "Target (₹)": target_price,
                "Verdict": decision
            })
        except:
            continue
    return pd.DataFrame(results)

if st.button("🚀 Run Nifty 100 Instant Scan"):
    with st.spinner("Scanning market & calculating automated Risk Management levels..."):
        df_results = scan_nifty_market(nifty_100_watchlist)
        
        if not df_results.empty:
            st.success(f"Scan complete! Successfully scanned {len(df_results)} stocks.")
            st.dataframe(df_results, use_container_width=True)
        else:
            st.error("Could not fetch market data right now. Please try again.")
else:
    st.info("Click the **'Run Nifty 100 Instant Scan'** button above to evaluate trades with automated Stop-Loss and Target levels.")
