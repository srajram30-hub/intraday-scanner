import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="Master Intraday Breakout & Momentum Scanner", layout="wide")

st.title("⚡ Master Intraday Breakout & Momentum Scanner (v2)")
st.markdown("Quantitative scoring model running across your comprehensive Indian stock universe with **Session VWAP & Weighted Pillars**.")

# Your complete master stock watchlist
master_watchlist = [
    "HFCL.NS", "RBLBANK.NS", "CUB.NS", "SAILIFE.NS", "AEGISLOG.NS", "ANGELONE.NS", 
    "CAMS.NS", "TDPOWERSYS.NS", "NEULANDLAB.NS", "LALPATHLAB.NS", "KARURVYSYA.NS", 
    "TATAELXSI.NS", "ANANDRATHI.NS", "APOLLOTYRE.NS", "NATCOPHARM.NS", "MTARTECH.NS", 
    "TATACHEM.NS", "ARE&M.NS", "KFINTECH.NS", "IGL.NS", "INOXWIND.NS", "GLAND.NS", 
    "TATATECH.NS", "BANDHANBNK.NS", "NAVINFLUOR.NS", "ATHERENERG.NS", "NBCC.NS", 
    "ONESOURCE.NS", "KPITTECH.NS", "CDSL.NS", "SYNGENE.NS", "WOCKPHARMA.NS", 
    "GESHIP.NS", "REDINGTON.NS", "MANAPPURAM.NS", "POONAWALLA.NS", "KIRLOSENG.NS", 
    "DELHIVERY.NS", "HSCL.NS", "PNBHOUSING.NS", "PGEL.NS", "AMBER.NS", "CROMPTON.NS", 
    "KAYNES.NS", "IIFL.NS", "SONACOMS.NS", "AFFLE.NS", "PPLPHARMA.NS", "WELCORP.NS", 
    "HAVELLS.NS", "FORTIS.NS", "PERSISTENT.NS", "NYKAA.NS", "MFSL.NS", "BHEL.NS", 
    "MANKIND.NS", "INDUSTOWER.NS", "SRF.NS", "AUROPHARMA.NS", "PRESTIGE.NS", 
    "FEDERALBNK.NS", "LAURUSLABS.NS", "LUPIN.NS", "GLENMARK.NS", "PHOENIXLTD.NS", 
    "MARICO.NS", "YESBANK.NS", "OIL.NS", "IDFCFIRSTB.NS", "PAYTM.NS", "HEROMOTOCO.NS", 
    "UNITDSPR.NS", "INDHOTEL.NS", "NHPC.NS", "TIINDIA.NS", "SUZLON.NS", "HINDPETRO.NS", 
    "DABUR.NS", "NAUKRI.NS", "INDUSINDBK.NS", "ICICIGI.NS", "NATIONALUM.NS", 
    "JSWENERGY.NS", "GODREJPROP.NS", "GMRAIRPORT.NS", "AUBANK.NS", "ASHOKLEY.NS", 
    "NMDC.NS", "BHARATFORG.NS", "MCX.NS", "DIXON.NS", "APLAPOLLO.NS", "RECLTD.NS", 
    "UPL.NS", "SWIGGY.NS", "POLICYBZR.NS", "INFY.NS", "HDFCLIFE.NS", "HDFCBANK.NS", 
    "SBILIFE.NS", "MAXHEALTH.NS", "TCS.NS", "HCLTECH.NS", "TATACONSUM.NS", "TECHM.NS", 
    "KOTAKBANK.NS", "ASIANPAINT.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "CIPLA.NS", 
    "NESTLEIND.NS", "APOLLOHOSP.NS", "SBIN.NS", "AXISBANK.NS", "ICICIBANK.NS", 
    "SUNPHARMA.NS", "BHARTIARTL.NS", "COALINDIA.NS", "INDIGO.NS", "BAJFINANCE.NS", 
    "BEL.NS", "BSE.NS", "ONGC.NS", "TITAN.NS", "TRENT.NS", "JSWSTEEL.NS", "RELIANCE.NS", 
    "LT.NS", "JIOFIN.NS", "DRREDDY.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "HINDUNILVR.NS", 
    "NTPC.NS", "ITC.NS", "ADANIENT.NS", "M&M.NS", "EICHERMOT.NS", "GRASIM.NS", 
    "ADANIPORTS.NS", "TATASTEEL.NS", "SHRIRAMFIN.NS", "MARUTI.NS", "BAJAJ-AUTO.NS"
]

@st.cache_data(ttl=60)
def scan_master_market(tickers):
    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 25:
                continue
                
            # Technical Indicators
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            # RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # True Session VWAP (Resets daily)
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
            
            # 1. Candle Strength (+15 pts)
            candle_body = abs(latest['Close'] - latest['Open'])
            candle_range = latest['High'] - latest['Low']
            is_strong_candle = (latest['Close'] > latest['Open']) and (candle_range > 0 and (candle_body / candle_range) > 0.4)
            if is_strong_candle:
                score += 15
                reasons.append("Strong Candle (+15)")
            else:
                invalidation_triggers.append("Weak/Doji candle")

            # 2. 20-Bar Breakout (+25 pts)
            recent_high = df['High'].iloc[-21:-1].max()
            is_breakout = current_price >= recent_high
            if is_breakout:
                score += 25
                reasons.append("20-Bar Breakout (+25)")
            else:
                invalidation_triggers.append("Below resistance")

            # 3. Above 20 EMA (+15 pts)
            is_above_ema = current_price > latest['EMA_20']
            if is_above_ema:
                score += 15
                reasons.append("Above 20 EMA (+15)")
            else:
                invalidation_triggers.append("Below 20 EMA")

            # 4. Above Session VWAP (+15 pts)
            is_above_vwap = current_price > latest['Session_VWAP']
            if is_above_vwap:
                score += 15
                reasons.append("Above Session VWAP (+15)")
            else:
                invalidation_triggers.append("Below Session VWAP")

            # 5. RVOL > 1.3 (+20 pts)
            rvol = latest['Volume'] / latest['Vol_MA20'] if latest['Vol_MA20'] > 0 else 0
            if rvol > 1.3:
                score += 20
                reasons.append(f"RVOL {rvol:.1f}x (+20)")
            else:
                invalidation_triggers.append(f"Low RVOL ({rvol:.1f}x)")

            # 6. RSI Health 50-75 (+10 pts)
            rsi_val = latest['RSI'] if not np.isnan(latest['RSI']) else 50
            if 50 <= rsi_val <= 75:
                score += 10
                reasons.append(f"RSI {rsi_val:.1f} (+10)")
            elif rsi_val > 75:
                score += 5
                reasons.append(f"RSI Overbought ({rsi_val:.1f}) (+5)")
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

            # Structure-Aware Risk Management
            atr_val = latest['ATR'] if not np.isnan(latest['ATR']) else (current_price * 0.005)
            structural_sl = latest['Low']
            atr_sl = current_price - (1.5 * atr_val)
            stop_loss = round(max(structural_sl, atr_sl), 2)
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

if st.button("🚀 Run Master Market Scan"):
    with st.spinner(f"Scanning your master list of {len(master_watchlist)} stocks... Please wait a moment."):
        df_results = scan_master_market(master_watchlist)
        
        if not df_results.empty:
            st.success(f"Scan complete! Successfully analyzed and ranked {len(df_results)} stocks.")
            st.dataframe(df_results, use_container_width=True)
        else:
            st.error("Could not fetch market data right now. Please try again.")
else:
    st.info("Click the **'Run Master Market Scan'** button above to evaluate your entire master universe instantly.")
