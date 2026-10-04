import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="Nifty 100 Breakout & Momentum Scanner (v2)", layout="wide")

st.title("⚡ Nifty 100 Intraday Breakout & Momentum Scanner (v2)")
st.markdown("Advanced quantitative scoring model using **Weighted Pillars, Session VWAP, RVOL, and Structure-Aware Risk Management**.")

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
                
            # 1. Technical Indicators Setup
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # CORRECTION: True Session VWAP (Resets daily)
            df['Date'] = df.index.date
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Vol'] = df['Typical_Price'] * df['Volume']
            df['Cum_TP_Vol'] = df['TP_Vol'].groupby(df['Date']).cumsum()
            df['Cum_Vol'] = df['Volume'].groupby(df['Date']).cumsum()
            df['Session_VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
            
            # ATR (14)
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
            
            # --- WEIGHTED SCORING ENGINE (0 - 100) ---
            score = 0
            reasons = []
            invalidation_triggers = []
            
            # Pillar 1: Candle Strength (+15 pts)
            candle_body = abs(latest['Close'] - latest['Open'])
            candle_range = latest['High'] - latest['Low']
            is_strong_candle = (latest['Close'] > latest['Open']) and (candle_range > 0 and (candle_body / candle_range) > 0.4)
            if is_strong_candle:
                score += 15
                reasons.append("Strong Candle (+15)")
            else:
                invalidation_triggers.append("Weak/Doji candle structure")

            # Pillar 2: 20-Bar Breakout (+25 pts)
            recent_high = df['High'].iloc[-21:-1].max()
            is_breakout = current_price >= recent_high
            if is_breakout:
                score += 25
                reasons.append("20-Bar Breakout (+25)")
            else:
                invalidation_triggers.append("Below 20-bar resistance")

            # Pillar 3: Trend - Above 20 EMA (+15 pts)
            is_above_ema = current_price > latest['EMA_20']
            if is_above_ema:
                score += 15
                reasons.append("Above 20 EMA (+15)")
            else:
                invalidation_triggers.append("Price below 20 EMA")

            # Pillar 4: Trend - Above Session VWAP (+15 pts)
            is_above_vwap = current_price > latest['Session_VWAP']
            if is_above_vwap:
                score += 15
                reasons.append("Above Session VWAP (+15)")
            else:
                invalidation_triggers.append("Trading below Session VWAP")

            # Pillar 5: Volume - RVOL > 1.3 (+20 pts)
            rvol = latest['Volume'] / latest['Vol_MA20'] if latest['Vol_MA20'] > 0 else 0
            if rvol > 1.3:
                score += 20
                reasons.append(f"RVOL {rvol:.1f}x (+20)")
            else:
                invalidation_triggers.append(f"Low Volume (RVOL {rvol:.1f}x)")

            # Pillar 6: Momentum - RSI 50-75 (+10 pts)
            rsi_val = latest['RSI'] if not np.isnan(latest['RSI']) else 50
            if 50 <= rsi_val <= 75:
                score += 10
                reasons.append(f"RSI {rsi_val:.1f} (+10)")
            elif rsi_val > 75:
                score += 5
                reasons.append(f"RSI Overbought {rsi_val:.1f} (+5)")
            else:
                invalidation_triggers.append(f"RSI Weak ({rsi_val:.1f})")

            # --- VERDICT TIERS ---
            if score >= 80:
                verdict = "🟢 STRONG SETUP (BUY)"
            elif score >= 65:
                verdict = "🟡 CONDITIONAL WATCH"
            elif score >= 50:
                verdict = "⚪ WEAK / NEUTRAL"
            else:
                verdict = "🔴 AVOID"

            # Structure-Aware Risk Management (Stop-Loss & Target)
            atr_val = latest['ATR'] if not np.isnan(latest['ATR']) else (current_price * 0.005)
            # Structure stop: low of breakout candle vs ATR stop
            structural_sl = latest['Low']
            atr_sl = current_price - (1.5 * atr_val)
            stop_loss = round(max(structural_sl, atr_sl), 2)  # Conservative closer stop
            target_price = round(current_price + (2.5 * atr_val), 2)

            results.append({
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": round(current_price, 2),
                "Change %": round(change_pct, 2),
                "Score": score,
                "Verdict": verdict,
                "Stop-Loss (₹)": stop_loss,
                "Target (₹)": target_price,
                "Why Score?": ", ".join(reasons),
                "Invalidation Factors": ", ".join(invalidation_triggers)
            })
        except:
            continue
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="Score", ascending=False)
    return df_res

if st.button("🚀 Run v2 Weighted Market Scan"):
    with st.spinner("Executing quantitative scoring model across Nifty blue-chips..."):
        df_results = scan_nifty_market(nifty_100_watchlist)
        
        if not df_results.empty:
            st.success(f"Scan complete! Ranked {len(df_results)} stocks by score.")
            st.dataframe(df_results, use_container_width=True)
        else:
            st.error("Could not fetch market data right now. Please try again.")
else:
    st.info("Click the **'Run v2 Weighted Market Scan'** button above to evaluate stocks using multi-factor scoring, session VWAP, and structural risk management.")
