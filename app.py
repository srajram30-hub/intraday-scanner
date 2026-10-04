import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import time

# ============================================================
# INTRADAY PULSE V5 — QUANTITATIVE MASTERCLASS EDITION
# ============================================================

st.set_page_config(
    page_title="Intraday Pulse V5",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MASTER PARAMETERS
# ============================================================

MIN_SCORE = 60
STRONG_SCORE = 75
RVOL_THRESHOLD = 1.35
BREAKOUT_BUFFER = 0.0015  # 0.15%
MAX_EXTENSION = 2.5
DATA_DAYS = 59            # Safely within Yahoo 15m limit
HOLDING_BARS = 6          # Max intraday holding bars (~1.5 hours)
TARGET_R = 1.5
BACKTEST_SCORE = 75
ENTRY_BUFFER = 0.001
MIN_BARS_BETWEEN_TRADES = 5

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
# DATA DOWNLOAD (Using auto_adjust=True & NIFTYBEES Proxy)
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
                auto_adjust=True,  # Corrects splits automatically
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
def download_nifty_proxy(days):
    try:
        df = yf.download(
            "NIFTYBEES.NS",  # Real volume & price proxy for Nifty
            period=f"{days}d",
            interval="15m",
            auto_adjust=True,
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

# ============================================================
# CAUSAL TECHNICAL INDICATORS (No Look-Ahead Bias)
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(df) < 40:
        return pd.DataFrame()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - prev),
        abs(df["Low"] - prev)
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["Date"] = df.index.date
    df["TypicalPrice"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["TPVolume"] = df["TypicalPrice"] * df["Volume"]
    df["CumTPVolume"] = df["TPVolume"].groupby(df["Date"]).cumsum()
    df["CumVolume"] = df["Volume"].groupby(df["Date"]).cumsum()
    df["VWAP"] = df["CumTPVolume"] / df["CumVolume"].replace(0, np.nan)

    df["Previous20High"] = df["High"].rolling(20).max().shift(1)
    
    # Robust boolean dtypes to avoid object-dtype bugs
    df["Breakout"] = (df["Close"] > df["Previous20High"] * (1 + BREAKOUT_BUFFER))
    prev_breakout = df["Breakout"].shift(1, fill_value=False).astype(bool)
    df["FreshBreakout"] = (df["Breakout"] & ~prev_breakout)

    age = 999
    ages = []
    for fresh in df["FreshBreakout"].fillna(False):
        if fresh:
            age = 0
        elif age < 999:
            age += 1
        ages.append(age)
    df["BreakoutAge"] = ages

    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    body = abs(df["Close"] - df["Open"])
    df["BodyPct"] = body / candle_range
    df["CloseLocation"] = (df["Close"] - df["Low"]) / candle_range
    df["UpperWickPct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range

    df["StrongCandle"] = (
        (df["Close"] > df["Open"]) &
        (df["BodyPct"] >= 0.50) &
        (df["CloseLocation"] >= 0.70) &
        (df["UpperWickPct"] <= 0.30)
    )

    df["IsGreen"] = (df["Close"] > df["Open"])
    prev_green = df["IsGreen"].shift(1, fill_value=False).astype(bool)
    df["ConsecutiveGreen"] = (df["IsGreen"] & prev_green)

    # Causal RVOL Calculation (Strictly Past Days Only)
    df["BarTime"] = df.index.strftime("%H:%M")
    rvol_list = []
    
    # Compute expanding historical time-of-day mean volume to prevent lookahead bias
    for idx, row in df.iterrows():
        b_time = row["BarTime"]
        b_date = row["Date"]
        past_bars = df[(df.index < idx) & (df["BarTime"] == b_time) & (df["Date"] < b_date)]
        if len(past_bars) >= 3:
            avg_vol = past_bars["Volume"].mean()
            rvol_val = row["Volume"] / avg_vol if avg_vol > 0 else 1.0
        else:
            rvol_val = 1.0
        rvol_list.append(rvol_val)
        
    df["RVOL"] = rvol_list

    df["RSIRising"] = df["RSI"] > df["RSI"].shift(1)
    df["Return5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100

    return df


def prepare_nifty(df):
    return calculate_indicators(df) if not df.empty else pd.DataFrame()


def get_market_regime(nifty_df, ts):
    """Exact timestamp matching for NIFTY market regime"""
    if nifty_df.empty:
        return 5, "🟡 UNKNOWN"
    try:
        pos = nifty_df.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            pos = 0
        row = nifty_df.iloc[pos]
        score = 0
        if row["Close"] > row["EMA20"]:
            score += 5
        if row["Close"] > row["VWAP"]:
            score += 5

        regime = "🟢 BULLISH" if score >= 10 else "🟡 NEUTRAL" if score >= 5 else "🔴 BEARISH"
        return score, regime
    except Exception:
        return 5, "🟡 UNKNOWN"


def score_at(df, nifty_df, i):
    if i < 30:
        return None

    row = df.iloc[i]
    ts = df.index[i]

    vals = ["Close", "EMA20", "VWAP", "RSI", "ATR", "RVOL", "Previous20High", "Return5", "ConsecutiveGreen"]
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
    consecutive_green = bool(row["ConsecutiveGreen"])
    rsi_rising = bool(row["RSIRising"])
    breakout_age = int(row["BreakoutAge"])

    market_score, market_regime = get_market_regime(nifty_df, ts)

    # Relative Strength calculation using timestamp lookup
    try:
        stock_now = price
        stock_prev = df["Close"].iloc[i - 5]
        stock_ret = (stock_now / stock_prev - 1) * 100

        nifty_slice = nifty_df.loc[:ts]
        if len(nifty_slice) >= 6:
            nifty_now = nifty_slice["Close"].iloc[-1]
            nifty_prev = nifty_slice["Close"].iloc[-6]
            nifty_ret = (nifty_now / nifty_prev - 1) * 100
            rs = float(stock_ret - nifty_ret)
        else:
            rs = 0.0
    except Exception:
        rs = 0.0

    score = 0
    reasons = []
    risks = []

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

    if breakout:
        score += 15
        reasons.append("20-bar breakout")
        if fresh:
            score += 5
            reasons.append("Fresh breakout")
    else:
        risks.append("No breakout")

    if rvol >= RVOL_THRESHOLD:
        score += 15
        reasons.append(f"Causal RVOL {rvol:.1f}x")
    else:
        risks.append(f"Low RVOL {rvol:.1f}x")

    if 50 <= rsi <= 75:
        score += 8
        reasons.append(f"Healthy RSI {rsi:.1f}")

    if rsi_rising:
        score += 4
        reasons.append("RSI rising")

    if consecutive_green:
        score += 5
        reasons.append("Consecutive green persistence")

    if strong_candle:
        score += 3
        reasons.append("Strong candle close")

    if rs >= 0.5:
        score += 10
        reasons.append("Positive RS vs Nifty")

    if market_score >= 5:
        score += 5
        reasons.append("Supportive market regime")

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


def trade_levels(df, i, price, atr):
    start = max(0, i - 5)
    swing_low = float(df["Low"].iloc[start:i].min())
    prev_high = df["Previous20High"].iloc[i]

    if pd.notna(prev_high):
        structure_stop = min(swing_low, float(prev_high) - 0.5 * atr)
    else:
        structure_stop = swing_low

    atr_stop = price - 1.5 * atr
    stop = max(structure_stop, atr_stop)
    stop = min(stop, price * 0.995)

    risk = price - stop
    if risk <= 0 or not np.isfinite(risk):
        risk = price * 0.01
        stop = price - risk

    target = price + TARGET_R * risk
    return float(stop), float(target), float(risk)


def run_live_scan(stock_data, nifty_raw):
    nifty_ind = prepare_nifty(nifty_raw)
    market_info = {"score": 5, "regime": "🟡 UNKNOWN"}
    if not nifty_ind.empty:
        ms, mr = get_market_regime(nifty_ind, nifty_ind.index[-1])
        market_info = {"score": ms, "regime": mr}

    results = []
    for ticker, raw in stock_data.items():
        df = calculate_indicators(raw)
        if df.empty or nifty_ind.empty:
            continue

        # Drop incomplete live forming candle if needed, evaluate last closed bar (-2) or current if closed
        i = len(df) - 2 if df.index[-1].time() < datetime.strptime("15:30", "%H:%M").time() else len(df) - 1
        if i < 30:
            continue

        s = score_at(df, nifty_ind, i)
        if s is None:
            continue

        stop, target, risk = trade_levels(df, i, s["price"], s["atr"])

        if s["score"] >= STRONG_SCORE and s["breakout"]:
            status = "🟢 ENTER / CONFIRM"
            verdict = "🟢 STRONG SETUP"
        elif s["score"] >= MIN_SCORE and s["breakout"]:
            status = "🟡 WATCH BREAKOUT"
            verdict = "🟡 CONDITIONAL WATCH"
        else:
            status = "🔴 AVOID"
            verdict = "⚪ NEUTRAL"

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
            "R:R": round(TARGET_R, 2),
            "RSI": round(s["rsi"], 1),
            "RVOL": round(s["rvol"], 2),
            "Relative Strength": round(s["rs"], 2),
            "VWAP Distance %": round((s["price"] / s["vwap"] - 1) * 100, 2),
            "Breakout": "YES" if s["breakout"] else "NO",
            "Breakout Age": s["breakout_age"],
            "Why Score?": ", ".join(s["reasons"]),
            "Risks": ", ".join(s["risks"]) if s["risks"] else "None"
        })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["Score", "RVOL", "Relative Strength"], ascending=[False, False, False])
    return out, market_info

# ============================================================
# MASTERCLASS BACKTEST ENGINE (Buy-Stop Fills + Session Square-off)
# ============================================================

def backtest_stock(ticker, raw_df, nifty_raw):
    df = calculate_indicators(raw_df)
    nifty = prepare_nifty(nifty_raw)

    if df.empty or nifty.empty or len(df) < 50:
        return []

    trades = []
    last_trade_i = -9999

    for i in range(35, len(df) - 1 - HOLDING_BARS):
        if i - last_trade_i <= MIN_BARS_BETWEEN_TRADES:
            continue

        signal = score_at(df, nifty, i)
        if signal is None:
            continue

        if signal["score"] < BACKTEST_SCORE:
            continue

        if not signal["breakout"]:
            continue

        entry_index = i + 1
        entry_time = df.index[entry_index]
        
        # Enforce intraday session boundary (Do not cross date boundary)
        if df.index[i].date() != df.index[entry_index].date():
            continue

        # Buy-Stop Fill Verification
        desired_entry = float(signal["price"] * (1 + ENTRY_BUFFER))
        next_bar_high = float(df["High"].iloc[entry_index])
        next_bar_open = float(df["Open"].iloc[entry_index])

        if next_bar_high < desired_entry:
            continue  # Order never triggered / filled

        entry = max(next_bar_open, desired_entry)
        stop, target, risk = trade_levels(df, i, entry, signal["atr"])

        exit_price = None
        exit_time = None
        exit_reason = None
        bars_held = 0

        # Restrict holding period strictly within the same session date
        current_date = df.index[i].date()
        end = entry_index
        while end < len(df) and end <= entry_index + HOLDING_BARS and df.index[end].date() == current_date:
            end += 1
        end -= 1

        if end < entry_index:
            continue

        for j in range(entry_index, end + 1):
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])
            bar_open = float(df["Open"].iloc[j])

            # Optimistic stop handling & Gap check
            if low <= stop:
                exit_price = min(stop, bar_open)  # Gap down protection
                exit_reason = "STOP"
                exit_time = df.index[j]
                bars_held = j - entry_index + 1
                break

            if high >= target:
                exit_price = target
                exit_reason = "TARGET"
                exit_time = df.index[j]
                bars_held = j - entry_index + 1
                break

        if exit_price is None:
            j = end
            exit_price = float(df["Close"].iloc[j])
            exit_time = df.index[j]
            bars_held = j - entry_index + 1
            exit_reason = "SESSION_SQUARE_OFF"

        pnl = exit_price - entry
        r_multiple = pnl / risk if risk > 0 else 0

        outcome = "WIN" if r_multiple > 0 else "LOSS" if r_multiple < 0 else "BREAKEVEN"

        trades.append({
            "Stock": ticker.replace(".NS", ""),
            "Signal Time": df.index[i],
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Score": signal["score"],
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "R": round(r_multiple, 3),
            "P&L/Share": round(pnl, 2),
            "Outcome": outcome,
            "Exit Reason": exit_reason,
            "Bars Held": bars_held,
            "RSI": round(signal["rsi"], 1),
            "RVOL": round(signal["rvol"], 2),
            "RS vs Nifty": round(signal["rs"], 2),
            "Market": signal["market_regime"]
        })

        last_trade_i = i

    return trades


@st.cache_data(ttl=300, show_spinner=False)
def run_full_backtest(tickers, days):
    stock_data, download_errors = download_market_data(tickers, days)
    nifty = download_nifty_proxy(days)

    all_trades = []
    errors = list(download_errors)

    for ticker, raw in stock_data.items():
        try:
            trades = backtest_stock(ticker, raw, nifty)
            all_trades.extend(trades)
        except Exception:
            errors.append(ticker)

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("Signal Time").reset_index(drop=True)

    return trades_df, sorted(set(errors))


def calculate_backtest_stats(trades):
    if trades.empty:
        return {}

    total = len(trades)
    wins = int((trades["Outcome"] == "WIN").sum())
    losses = int((trades["Outcome"] == "LOSS").sum())
    breakeven = int((trades["Outcome"] == "BREAKEVEN").sum())

    win_rate = wins / total * 100 if total > 0 else 0
    gross_profit = trades.loc[trades["R"] > 0, "R"].sum()
    gross_loss = abs(trades.loc[trades["R"] < 0, "R"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy = trades["R"].mean()
    avg_win = trades.loc[trades["R"] > 0, "R"].mean() if wins else 0
    avg_loss = trades.loc[trades["R"] < 0, "R"].mean() if losses else 0

    # Prepend 0 to properly track starting drawdown from 0
    cumulative_r = pd.concat([pd.Series([0.0]), trades["R"].cumsum()]).reset_index(drop=True)
    peak = cumulative_r.cummax()
    drawdown = cumulative_r - peak
    max_drawdown = abs(drawdown.min())

    return {
        "Trades": total,
        "Wins": wins,
        "Losses": losses,
        "Breakeven": breakeven,
        "Win Rate %": win_rate,
        "Avg Win R": avg_win,
        "Avg Loss R": avg_loss,
        "Profit Factor": profit_factor,
        "Expectancy R/Trade": expectancy,
        "Total R": trades["R"].sum(),
        "Max Drawdown R": max_drawdown
    }

# ============================================================
# UI INTERFACE
# ============================================================

st.markdown("# ⚡ Intraday Pulse V5")
st.markdown("Quantitative Masterclass — Causal Indicators & Strict Session Execution")

scan_tab, backtest_tab = st.tabs(["🚀 Live Scanner", "📈 Backtest"])

# ============================================================
# LIVE SCANNER TAB
# ============================================================

with scan_tab:
    if st.button("🚀 Run Masterclass Scan", use_container_width=True, type="primary"):
        with st.spinner(f"Scanning {len(MASTER_WATCHLIST)} stocks with causal filters..."):
            stock_data, errors = download_market_data(tuple(MASTER_WATCHLIST), DATA_DAYS)
            nifty = download_nifty_proxy(DATA_DAYS)
            results, market_info = run_live_scan(stock_data, nifty)

        st.session_state["live_results"] = results
        st.session_state["market_info"] = market_info
        st.session_state["scan_time"] = datetime.now()

    if "live_results" not in st.session_state:
        st.info(f"Tap **Run Masterclass Scan** above to scan your {len(MASTER_WATCHLIST)}-stock universe.")
    else:
        results = st.session_state["live_results"]
        market_info = st.session_state["market_info"]

        st.markdown(f"### Market Regime: {market_info['regime']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(MASTER_WATCHLIST))
        c2.metric("Analyzed", len(results))
        c3.metric("Strong Setups", int((results["Score"] >= STRONG_SCORE).sum()))
        c4.metric("Breakouts", int((results["Breakout"] == "YES").sum()))

        st.markdown("---")
        st.subheader("📋 Master Stock Information Table")
        
        if results.empty:
            st.warning("No stocks match the current criteria.")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Scan CSV",
                results.to_csv(index=False).encode("utf-8"),
                "masterclass_scan_results.csv",
                "text/csv",
                use_container_width=True
            )

# ============================================================
# BACKTEST TAB
# ============================================================

with backtest_tab:
    st.subheader("📈 Masterclass Historical Backtest Engine")
    st.warning("Simulating strict causal rules: timestamp NIFTY alignment, causal expanding RVOL, buy-stop fill verification, and same-session square-off.")

    if st.button("📊 Run Masterclass Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running rigorous event-by-event backtest..."):
            trades, errors = run_full_backtest(tuple(MASTER_WATCHLIST), DATA_DAYS)

        st.session_state["backtest_trades"] = trades
        st.session_state["backtest_errors"] = errors
        st.session_state["backtest_time"] = datetime.now()

    if "backtest_trades" not in st.session_state:
        st.info("Tap **Run Masterclass Backtest** above to measure rigorous performance metrics.")
    else:
        trades = st.session_state["backtest_trades"]

        if trades.empty:
            st.error("No historical trades matched the rigorous audit rules.")
        else:
            stats = calculate_backtest_stats(trades)

            st.subheader("🎯 Backtest Performance Metrics")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Win Rate", f"{stats['Win Rate %']:.1f}%")
            bc2.metric("Total Trades", stats["Trades"])
            bc3.metric("Expectancy", f"{stats['Expectancy R/Trade']:+.3f}R")
            bc4.metric("Profit Factor", f"{stats['Profit Factor']:.2f}" if np.isfinite(stats["Profit Factor"]) else "∞")

            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total R", f"{stats['Total R']:+.2f}R")
            bc2.metric("Avg Win", f"{stats['Avg Win R']:+.2f}R")
            bc3.metric("Avg Loss", f"{stats['Avg Loss R']:+.2f}R")
            bc4.metric("Max Drawdown", f"{stats['Max Drawdown R']:.2f}R")

            st.subheader("📈 Cumulative Equity Curve (R)")
            equity = pd.concat([pd.Series([0.0]), trades["R"].cumsum()]).reset_index(drop=True)
            st.line_chart(pd.DataFrame({"Cumulative R": equity.values}))

            st.subheader("📒 Complete Trade Log")
            st.dataframe(trades, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Trade Log CSV",
                trades.to_csv(index=False).encode("utf-8"),
                "masterclass_backtest_trade_log.csv",
                "text/csv",
                use_container_width=True
            )
