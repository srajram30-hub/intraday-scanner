import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# Page configuration for mobile-friendly view
st.set_page_config(page_title="Intraday Pulse", page_icon="⚡", layout="centered")

# Custom CSS for Sleek Mobile App UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stock-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 18px;
        font-weight: 700;
        color: #f0f6fc;
    }
    .stock-price {
        font-size: 16px;
        color: #8b949e;
    }
    .badge-green {
        background-color: #238636;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    .badge-yellow {
        background-color: #9e6a03;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    .badge-red {
        background-color: #da3633;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-size: 14px;
        color: #c9d1d9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Intraday Pulse")
st.markdown("Mobile-Optimized Momentum & Breakout Scanner")

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
                
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['Date'] = df.index.date
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Vol'] = df['Typical_Price'] * df['Volume']
            df['Cum_TP_Vol'] = df['TP_Vol'].groupby(df['Date']).cumsum()
            df['Cum_Vol'] = df['Volume'].groupby(df['Date']).cumsum()
            df['Session_VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
            
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
            
            score = 0
            reasons = []
            invalidation_triggers = []
            
            # Scoring logic
            candle_body = abs(latest['Close'] - latest['Open'])
            candle_range = latest['High'] - latest['Low']
            if (latest['Close'] > latest['Open']) and (candle_range > 0 and (candle_body / candle_range) > 0.4):
                score += 15
                reasons.append("Strong Candle (+15)")
            else:
                invalidation_triggers.append("Weak Candle")

            recent_high = df['High'].iloc[-21:-1].max()
            if current_price >= recent_high:
                score += 25
                reasons.append("Breakout (+25)")
            else:
                invalidation_triggers.append("No Breakout")

            if current_price > latest['EMA_20']:
                score += 15
                reasons.append("Above EMA (+15)")
            else:
                invalidation_triggers.append("Below EMA")

            if current_price > latest['Session_VWAP']:
                score += 15
                reasons.append("Above VWAP (+15)")
            else:
                inulation_triggers = invalidation_triggers.append("Below VWAP")

            rvol = latest['Volume'] / latest['Vol_MA20'] if latest['Vol_MA20'] > 0 else 0
            if rvol > 1.3:
                score += 20
                reasons.append(f"RVOL {rvol:.1f}x (+20)")
            else:
                invalidation_triggers.append(f"Low RVOL ({rvol:.1f}x)")

            rsi_val = latest['RSI'] if not np.isnan(latest['RSI']) else 50
            if 50 <= rsi_val <= 75:
                score += 10
                reasons.append(f"RSI {rsi_val:.1f} (+10)")
            elif rsi_val > 75:
                score += 5
                reasons.append(f"RSI Overbought (+5)")
            else:
                invalidation_triggers.append(f"Weak RSI")

            if score >= 80:
                badge_class = "badge-green"
                verdict_text = "🟢 STRONG SETUP"
            elif score >= 65:
                badge_class = "badge-yellow"
                verdict_text = "🟡 WATCHLIST"
            else:
                badge_class = "badge-red"
                verdict_text = "🔴 AVOID"

            atr_val = latest['ATR'] if not np.isnan(latest['ATR']) else (current_price * 0.005)
            stop_loss = round(max(latest['Low'], current_price - (1.5 * atr_val)), 2)
            target_price = round(current_price + (2.5 * atr_val), 2)

            results.append({
                "Stock": ticker.replace(".NS", ""),
                "Price": round(current_price, 2),
                "Change": round(change_pct, 2),
                "Score": score,
                "Badge": badge_class,
                "Verdict": verdict_text,
                "StopLoss": stop_loss,
                "Target": target_price,
                "Why": ", ".join(reasons),
                "Invalidation": ", ".join(invalidation_triggers)
            })
        except:
            continue
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="Score", ascending=False)
    return df_res

# Filter option for mobile users
min_score_filter = st.sidebar.slider("Minimum Score Filter", 0, 100, 0)

if st.button("🚀 Run Mobile Scan", use_container_width=True):
    with st.spinner("Scanning market and building mobile cards..."):
        df_results = scan_master_market(master_watchlist)
        
        if not df_results.empty:
            filtered_df = df_results[df_results["Score"] >= min_score_filter]
            st.success(f"Found {len(filtered_df)} setups matching criteria.")
            
            for index, row in filtered_df.iterrows():
                change_color = "#3fb950" if row['Change'] >= 0 else "#f85149"
                
                # Render clean mobile card UI
                st.markdown(f"""
                    <div class="stock-card">
                        <div class="stock-header">
                            <span>{row['Stock']}</span>
                            <span class="{row['Badge']}">{row['Score']}/100</span>
                        </div>
                        <div class="metric-row">
                            <span class="stock-price">₹{row['Price']:,.2f}</span>
                            <span style="color: {change_color}; font-weight: 600;">{row['Change']:+.2f}%</span>
                            <span style="color: #f0f6fc; font-weight: 500;">{row['Verdict']}</span>
                        </div>
                        <hr style="border-color: #30363d; margin: 8px 0;">
                        <div class="metric-row">
                            <span>🛑 SL: <b>₹{row['StopLoss']}</b></span>
                            <span>🎯 Target: <b>₹{row['Target']}</b></span>
                        </div>
                        <div style="font-size: 12px; color: #8b949e; margin-top: 6px;">
                            <b>Why:</b> {row['Why']}
                        </div>
                        <div style="font-size: 12px; color: #f85149; margin-top: 2px;">
                            <b>Risks:</b> {row['Invalidation']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Could not fetch market data right now. Please try again.")
else:
    st.info("Tap the **'Run Mobile Scan'** button above to load your cards.")
