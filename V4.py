
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import time

# ============================================================
# INTRADAY PULSE V4
# V3 SCANNER + HISTORICAL BACKTEST ENGINE
# ============================================================

st.set_page_config(
    page_title="Intraday Pulse V4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MASTER WATCHLIST
# ============================================================

MASTER_WATCHLIST = [
    "HFCL.NS", "RBLBANK.NS", "CUB.NS", "SAILIFE.NS", "AEGISLOG.NS",
    "ANGELONE.NS", "CAMS.NS", "TDPOWERSYS.NS", "NEULANDLAB.NS",
    "LALPATHLAB.NS", "KARURVYSYA.NS", "TATAELXSI.NS", "ANANDRATHI.NS",
    "APOLLOTYRE.NS", "NATCOPHARM.NS", "MTARTECH.NS", "TATACHEM.NS",
    "ARE&M.NS", "KFINTECH.NS", "IGL.NS", "INOXWIND.NS", "GLAND.NS",
    "TATATECH.NS", "BANDHANBNK.NS", "NAVINFLUOR.NS", "ATHERENERG.NS",
    "NBCC.NS", "ONESOURCE.NS", "KPITTECH.NS", "CDSL.NS", "SYNGENE.NS",
    "WOCKPHARMA.NS", "GESHIP.NS", "REDINGTON.NS", "MANAPPURAM.NS",
    "POONAWALLA.NS", "KIRLOSENG.NS", "DELHIVERY.NS", "HSCL.NS",
    "PNBHOUSING.NS", "PGEL.NS", "AMBER.NS", "CROMPTON.NS", "KAYNES.NS",
    "IIFL.NS", "SONACOMS.NS", "AFFLE.NS", "PPLPHARMA.NS", "WELCORP.NS",
    "HAVELLS.NS", "FORTIS.NS", "PERSISTENT.NS", "NYKAA.NS", "MFSL.NS",
    "BHEL.NS", "MANKIND.NS", "INDUSTOWER.NS", "SRF.NS", "AUROPHARMA.NS",
    "PRESTIGE.NS", "FEDERALBNK.NS", "LAURUSLABS.NS", "LUPIN.NS",
    "GLENMARK.NS", "PHOENIXLTD.NS", "MARICO.NS", "YESBANK.NS", "OIL.NS",
    "IDFCFIRSTB.NS", "PAYTM.NS", "HEROMOTOCO.NS", "UNITDSPR.NS",
    "INDHOTEL.NS", "NHPC.NS", "TIINDIA.NS", "SUZLON.NS", "HINDPETRO.NS",
    "DABUR.NS", "NAUKRI.NS", "INDUSINDBK.NS", "ICICIGI.NS", "NATIONALUM.NS",
    "JSWENERGY.NS", "GODREJPROP.NS", "GMRAIRPORT.NS", "AUBANK.NS",
    "ASHOKLEY.NS", "NMDC.NS", "BHARATFORG.NS", "MCX.NS", "DIXON.NS",
    "APLAPOLLO.NS", "RECLTD.NS", "UPL.NS", "SWIGGY.NS", "POLICYBZR.NS",
    "INFY.NS", "HDFCLIFE.NS", "HDFCBANK.NS", "SBILIFE.NS", "MAXHEALTH.NS",
    "TCS.NS", "HCLTECH.NS", "TATACONSUM.NS", "TECHM.NS", "KOTAKBANK.NS",
    "ASIANPAINT.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "CIPLA.NS",
    "NESTLEIND.NS", "APOLLOHOSP.NS", "SBIN.NS", "AXISBANK.NS",
    "ICICIBANK.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "COALINDIA.NS",
    "INDIGO.NS", "BAJFINANCE.NS", "BEL.NS", "BSE.NS", "ONGC.NS",
    "TITAN.NS", "TRENT.NS", "JSWSTEEL.NS", "RELIANCE.NS", "LT.NS",
    "JIOFIN.NS", "DRREDDY.NS", "ULTRACEMCO.NS", "POWERGRID.NS",
    "HINDUNILVR.NS", "NTPC.NS", "ITC.NS", "ADANIENT.NS", "M&M.NS",
    "EICHERMOT.NS", "GRASIM.NS", "ADANIPORTS.NS", "TATASTEEL.NS",
    "SHRIRAMFIN.NS", "MARUTI.NS", "BAJAJ-AUTO.NS"
]

# ============================================================
# SETTINGS
# ============================================================

st.sidebar.header("⚙️ Scanner / Backtest Settings")

MIN_SCORE = st.sidebar.slider("Minimum display/backtest score", 40, 90, 55, 5)
STRONG_SCORE = st.sidebar.slider("Strong setup score", 70, 95, 80, 5)
RVOL_THRESHOLD = st.sidebar.slider("Minimum RVOL", 1.0, 3.0, 1.30, 0.10)
BREAKOUT_BUFFER = st.sidebar.slider("Breakout buffer %", 0.0, 1.0, 0.15, 0.05) / 100
MAX_EXTENSION = st.sidebar.slider("Maximum 5-bar extension %", 1.0, 6.0, 3.0, 0.5)

DATA_DAYS = st.sidebar.selectbox(
    "Yahoo intraday history",
    [30, 45, 60],
    index=2
)

HOLDING_BARS = st.sidebar.slider(
    "Backtest max holding bars",
    2, 20, 8, 1
)

TARGET_R = st.sidebar.slider(
    "Backtest target (R)",
    1.0, 4.0, 2.0, 0.5
)

BACKTEST_SCORE = st.sidebar.slider(
    "Backtest minimum score",
    50, 95, 80, 5
)

ENTRY_BUFFER = st.sidebar.slider(
    "Entry buffer %",
    0.0, 0.5, 0.0, 0.05
) / 100

MIN_BARS_BETWEEN_TRADES = st.sidebar.slider(
    "Minimum bars between same-stock trades",
    0, 20, 4, 1
)

SHOW_WATCH = st.sidebar.checkbox("Show conditional setups", True)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.main-title {font-size:42px;font-weight:800;margin-bottom:0;}
.subtitle {font-size:16px;color:#777;margin-bottom:25px;}
.stock-card {
    background:#11161c;border-radius:18px;padding:20px;margin-bottom:18px;
    border:1px solid #29313a;box-shadow:0 5px 18px rgba(0,0,0,.20);
}
.stock-header {display:flex;justify-content:space-between;align-items:center;}
.stock-name {font-size:25px;font-weight:800;}
.score {padding:8px 15px;border-radius:20px;font-weight:800;font-size:18px;}
.green {background:#16883b;color:white;}
.yellow {background:#c99b00;color:white;}
.red {background:#a82b2b;color:white;}
.price {font-size:24px;font-weight:700;}
.positive {color:#35c759;font-weight:700;}
.negative {color:#ff4d4d;font-weight:700;}
.info-row {
    display:flex;justify-content:space-between;margin-top:10px;
    padding-top:10px;border-top:1px solid #29313a;
}
.label {color:#999;font-size:13px;}
.value {font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA DOWNLOAD
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def download_market_data(tickers, days):
    all_data = {}
    errors = []
    chunk_size = 35

    for start in range(0, len(tickers), chunk_size):
        chunk = list(tickers[start:start + chunk_size])
        try:
            data = yf.download(
                chunk,
                period=f"{days}d",
                interval="15m",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True
            )

            if data.empty:
                errors.extend(chunk)
                continue

            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].copy()
                        elif ticker in data.columns.get_level_values(1):
                            df = data.xs(ticker, axis=1, level=1).copy()
                        else:
                            errors.append(ticker)
                            continue
                    else:
                        df = data.copy()

                    df = df.dropna(subset=["Close"])
                    if len(df) >= 40:
                        all_data[ticker] = df
                    else:
                        errors.append(ticker)
                except Exception:
                    errors.append(ticker)
        except Exception:
            errors.extend(chunk)

        time.sleep(0.15)

    return all_data, sorted(set(errors))


@st.cache_data(ttl=60, show_spinner=False)
def download_nifty(days):
    try:
        df = yf.download(
            "^NSEI",
            period=f"{days}d",
            interval="15m",
            auto_adjust=False,
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(df) < 40:
        return pd.DataFrame()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - prev),
        abs(df["Low"] - prev)
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Session VWAP
    df["Date"] = df.index.date
    df["TypicalPrice"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["TPVolume"] = df["TypicalPrice"] * df["Volume"]
    df["CumTPVolume"] = df["TPVolume"].groupby(df["Date"]).cumsum()
    df["CumVolume"] = df["Volume"].groupby(df["Date"]).cumsum()
    df["VWAP"] = df["CumTPVolume"] / df["CumVolume"].replace(0, np.nan)

    # 20-bar resistance
    df["Previous20High"] = df["High"].rolling(20).max().shift(1)
    df["Breakout"] = (
        df["Close"] >
        df["Previous20High"] * (1 + BREAKOUT_BUFFER)
    )
    df["FreshBreakout"] = (
        df["Breakout"] &
        ~df["Breakout"].shift(1).fillna(False)
    )

    # Breakout age without future leakage
    age = 999
    ages = []
    for fresh in df["FreshBreakout"].fillna(False):
        if fresh:
            age = 0
        elif age < 999:
            age += 1
        ages.append(age)
    df["BreakoutAge"] = ages

    # Candle structure
    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    body = abs(df["Close"] - df["Open"])
    df["BodyPct"] = body / candle_range
    df["CloseLocation"] = (df["Close"] - df["Low"]) / candle_range
    df["UpperWickPct"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / candle_range

    df["StrongCandle"] = (
        (df["Close"] > df["Open"]) &
        (df["BodyPct"] >= 0.50) &
        (df["CloseLocation"] >= 0.70) &
        (df["UpperWickPct"] <= 0.30)
    )

    # Time-of-day RVOL, calculated only from prior dates
    df["BarTime"] = df.index.strftime("%H:%M")
    historical = df.copy()
    current_date = df["Date"].iloc[-1]
    historical = historical[historical["Date"] < current_date]

    if len(historical):
        ref = historical.groupby("BarTime")["Volume"].mean()
        df["TimeOfDayAvgVolume"] = df["BarTime"].map(ref)
        df["RVOL"] = df["Volume"] / df["TimeOfDayAvgVolume"]
    else:
        df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Momentum
    df["RSIRising"] = df["RSI"] > df["RSI"].shift(1)
    df["Return5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100

    return df

# ============================================================
# MARKET / RELATIVE STRENGTH
# ============================================================

def prepare_nifty(df):
    return calculate_indicators(df) if not df.empty else pd.DataFrame()


def relative_strength_at(stock_df, nifty_df, i):
    try:
        if i < 5:
            return 0.0
        stock_now = stock_df["Close"].iloc[i]
        stock_prev = stock_df["Close"].iloc[i - 5]
        stock_ret = (stock_now / stock_prev - 1) * 100

        ts = stock_df.index[i]
        nifty_slice = nifty_df.loc[:ts]
        if len(nifty_slice) < 6:
            return 0.0

        nifty_now = nifty_slice["Close"].iloc[-1]
        nifty_prev = nifty_slice["Close"].iloc[-6]
        nifty_ret = (nifty_now / nifty_prev - 1) * 100

        return float(stock_ret - nifty_ret)
    except Exception:
        return 0.0


def market_regime_at(nifty, i):
    try:
        row = nifty.iloc[i]
        score = 0
        if row["Close"] > row["EMA20"]:
            score += 5
        if row["Close"] > row["VWAP"]:
            score += 5

        if score >= 10:
            regime = "🟢 BULLISH"
        elif score >= 5:
            regime = "🟡 NEUTRAL"
        else:
            regime = "🔴 BEARISH"

        return score, regime
    except Exception:
        return 5, "🟡 UNKNOWN"

# ============================================================
# SCORE AT A HISTORICAL BAR
# IMPORTANT: ONLY DATA UP TO BAR i IS USED.
# ============================================================

def score_at(df, nifty, i):
    if i < 30:
        return None

    row = df.iloc[i]

    vals = ["Close", "EMA20", "VWAP", "RSI", "ATR", "RVOL",
            "Previous20High", "Return5"]
    if any(pd.isna(row[v]) for v in vals):
        return None

    price = float(row["Close"])
    ema = float(row["EMA20"])
    vwap = float(row["VWAP"])
    rsi = float(row["RSI"])
    atr = float(row["ATR"])
    rvol = float(row["RVOL"])
    prev_high = float(row["Previous20High"])
    extension = float(row["Return5"])

    breakout = bool(row["Breakout"])
    fresh = bool(row["FreshBreakout"])
    strong_candle = bool(row["StrongCandle"])
    rsi_rising = bool(row["RSIRising"])
    breakout_age = int(row["BreakoutAge"])

    market_score, market_regime = market_regime_at(
        nifty,
        min(i, len(nifty) - 1)
    )
    rs = relative_strength_at(df, nifty, i)

    score = 0
    reasons = []
    risks = []

    # Trend 20
    if price > ema:
        score += 10
        reasons.append("Above EMA20")
    else:
        risks.append("Below EMA20")

    if price > vwap:
        score += 10
        reasons.append("Above VWAP")
    else:
        risks.append("Below VWAP")

    # Breakout 25
    if breakout:
        score += 15
        reasons.append("20-bar breakout")

        distance = (price / prev_high - 1) * 100
        if distance >= 0.30:
            score += 5
            reasons.append("Strong breakout distance")
        elif distance >= 0.15:
            score += 3
            reasons.append("Confirmed breakout distance")

        if fresh:
            score += 5
            reasons.append("Fresh breakout")
        elif breakout_age <= 3:
            score += 3
            reasons.append("Recent breakout")
        elif breakout_age > 6:
            risks.append("Aging breakout")
    else:
        risks.append("No confirmed breakout")

    # Volume 15
    if rvol >= 2.0:
        score += 15
        reasons.append(f"Exceptional RVOL {rvol:.1f}x")
    elif rvol >= RVOL_THRESHOLD:
        score += 10
        reasons.append(f"Strong RVOL {rvol:.1f}x")
    elif rvol >= 1.0:
        score += 5
        reasons.append(f"Normal RVOL {rvol:.1f}x")
    else:
        risks.append(f"Low RVOL {rvol:.1f}x")

    # Momentum 15
    if 55 <= rsi <= 70:
        score += 8
        reasons.append(f"Healthy RSI {rsi:.1f}")
    elif 70 < rsi <= 80:
        score += 6
        reasons.append(f"Strong RSI {rsi:.1f}")
    elif 50 <= rsi < 55:
        score += 4
        reasons.append(f"Developing RSI {rsi:.1f}")
    elif rsi > 80:
        score += 2
        risks.append(f"Overheated RSI {rsi:.1f}")
    else:
        risks.append(f"Weak RSI {rsi:.1f}")

    if rsi_rising:
        score += 4
        reasons.append("RSI rising")
    else:
        risks.append("RSI not rising")

    if strong_candle:
        score += 3
        reasons.append("Strong candle")
    else:
        risks.append("Weak candle")

    # Relative strength 10
    if rs >= 1.5:
        score += 10
        reasons.append("Strong RS vs NIFTY")
    elif rs >= 0.75:
        score += 7
        reasons.append("Positive RS vs NIFTY")
    elif rs >= 0.25:
        score += 4
        reasons.append("Moderate RS")
    else:
        risks.append("Weak RS vs NIFTY")

    # Market 10
    if market_score >= 10:
        score += 10
        reasons.append("Bullish NIFTY")
    elif market_score >= 5:
        score += 5
        reasons.append("Neutral NIFTY")
    else:
        risks.append("Bearish NIFTY")

    # Extension/risk 5
    if extension <= 1.5:
        score += 5
        reasons.append("Not extended")
    elif extension <= MAX_EXTENSION:
        score += 3
        reasons.append("Moderately extended")
    else:
        risks.append(f"Extended {extension:.1f}% / 5 bars")

    return {
        "score": int(min(max(score, 0), 100)),
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "rvol": rvol,
        "vwap": vwap,
        "rs": rs,
        "extension": extension,
        "breakout": breakout,
        "fresh": fresh,
        "breakout_age": breakout_age,
        "market_score": market_score,
        "market_regime": market_regime,
        "reasons": reasons,
        "risks": risks
    }

# ============================================================
# STOP / TARGET
# ============================================================

def trade_levels(df, i, price, atr, target_r):
    start = max(0, i - 5)
    swing_low = float(df["Low"].iloc[start:i].min())

    prev_high = df["Previous20High"].iloc[i]

    if pd.notna(prev_high):
        structure_stop = min(
            swing_low,
            float(prev_high) - 0.5 * atr
        )
    else:
        structure_stop = swing_low

    atr_stop = price - 1.5 * atr
    stop = max(structure_stop, atr_stop)

    # Keep stop below entry
    stop = min(stop, price * 0.995)

    risk = price - stop

    if risk <= 0 or not np.isfinite(risk):
        risk = price * 0.01
        stop = price - risk

    target = price + target_r * risk

    return float(stop), float(target), float(risk)

# ============================================================
# LIVE SCANNER
# ============================================================

def run_live_scan(stock_data, nifty):
    nifty_ind = prepare_nifty(nifty)

    if nifty_ind.empty:
        market_info = {"score": 5, "regime": "🟡 UNKNOWN"}
    else:
        ms, mr = market_regime_at(nifty_ind, len(nifty_ind) - 1)
        market_info = {"score": ms, "regime": mr}

    results = []

    for ticker, raw in stock_data.items():
        df = calculate_indicators(raw)
        if df.empty or nifty_ind.empty:
            continue

        i = len(df) - 1
        s = score_at(df, nifty_ind, i)
        if s is None:
            continue

        stop, target, risk = trade_levels(
            df, i, s["price"], s["atr"], TARGET_R
        )

        if s["extension"] > MAX_EXTENSION:
            status = "🟠 EXTENDED"
        elif s["score"] >= STRONG_SCORE and s["breakout"]:
            status = "🟢 ENTER / CONFIRM"
        elif s["score"] >= MIN_SCORE and s["breakout"]:
            status = "🟡 WATCH BREAKOUT"
        elif s["score"] >= MIN_SCORE:
            status = "🟡 WAIT FOR BREAKOUT"
        else:
            status = "🔴 AVOID"

        if s["score"] >= STRONG_SCORE and s["breakout"]:
            verdict = "🟢 STRONG SETUP"
        elif s["score"] >= MIN_SCORE:
            verdict = "🟡 CONDITIONAL WATCH"
        else:
            verdict = "⚪ WEAK / NEUTRAL"

        prev_close = float(df["Close"].iloc[-2])
        change = (s["price"] / prev_close - 1) * 100

        results.append({
            "Stock": ticker.replace(".NS", ""),
            "Score": s["score"],
            "Verdict": verdict,
            "Status": status,
            "Price": round(s["price"], 2),
            "Change %": round(change, 2),
            "Entry": round(s["price"], 2),
            "Stop Loss": round(stop, 2),
            "Target": round(target, 2),
            "Risk/Share": round(risk, 2),
            "R:R": round(target_r, 2),
            "RSI": round(s["rsi"], 1),
            "RVOL": round(s["rvol"], 2),
            "Relative Strength": round(s["rs"], 2),
            "VWAP Distance %": round((s["price"] / s["vwap"] - 1) * 100, 2),
            "Breakout": "YES" if s["breakout"] else "NO",
            "Breakout Age": s["breakout_age"],
            "5-Bar Move %": round(s["extension"], 2),
            "Why Score?": ", ".join(s["reasons"]),
            "Risks": ", ".join(s["risks"]) if s["risks"] else "None"
        })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(
            ["Score", "RVOL", "Relative Strength"],
            ascending=[False, False, False]
        )
    return out, market_info

# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_stock(ticker, raw_df, nifty_raw):
    """
    Event-driven backtest.

    Signal is evaluated at the CLOSE of b
