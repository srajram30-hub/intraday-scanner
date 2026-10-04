import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import time

# ============================================================
# INTRADAY PULSE — ELITE SNIPER QUANTITATIVE EDITION
# ============================================================

st.set_page_config(
    page_title="Intraday Pulse Elite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# UNCOMPROMISING INSTITUTIONAL PARAMETERS
# ============================================================

MIN_SCORE = 75
STRONG_SCORE = 88
RVOL_THRESHOLD = 2.00     # Strict institutional volume multiplier (2x)
BREAKOUT_BUFFER = 0.0020  # 0.20% clear of resistance to avoid false wicks
MAX_EXTENSION = 2.0       # Strict anti-chasing extension limit (2%)
DATA_DAYS = 60
HOLDING_BARS = 8
TARGET_R = 2.5            # Higher R:R target to maximize expectancy
BACKTEST_SCORE = 88       # Elite threshold to eliminate over-trading
ENTRY_BUFFER = 0.001      # Confirmed breakout entry buffer
MIN_BARS_BETWEEN_TRADES = 8

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
# STRICT TECHNICAL INDICATORS & SCORING
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
    df["Breakout"] = (
        df["Close"] >
        df["Previous20High"] * (1 + BREAKOUT_BUFFER)
    )
    df["FreshBreakout"] = (
        df["Breakout"] &
        ~df["Breakout"].shift(1).fillna(False)
    )

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
    df["UpperWickPct"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / candle_range

    # Mandatory strict candle confirmation
    df["StrongCandle"] = (
        (df["Close"] > df["Open"]) &
        (df["BodyPct"] >= 0.55) &
        (df["CloseLocation"] >= 0.75) &
        (df["UpperWickPct"] <= 0.25)
    )

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

    df["RSIRising"] = df["RSI"] > df["RSI"].shift(1)
    df["Return5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100

    return df


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

    # UNCOMPROMISING FILTER: Disqualify immediately if market is bearish
    if market_score < 10:
        return None

    score = 0
    reasons = []
    risks = []

    # Strict Trend Gate
    if price > ema and price > vwap:
        score += 20
        reasons.append("Strictly above EMA20 & VWAP")
    else:
        return None  # Disqualify if not above both trend benchmarks

    # Strict Breakout Gate
    if breakout and strong_candle:
        score += 25
        reasons.append("Clean confirmed breakout with strong candle")
        if fresh:
            score += 10
            reasons.append("Pristine fresh breakout")
    else:
        return None  # Disqualify without a clean breakout + strong candle

    # Strict Volume Gate (Must meet 2.0x RVOL)
    if rvol >= RVOL_THRESHOLD:
        score += 20
        reasons.append(f"Institutional RVOL {rvol:.1f}x")
    else:
        return None  # Disqualify low volume breakouts

    # Momentum Gate
    if 55 <= rsi <= 75 and rsi_rising:
        score += 15
        reasons.append(f"Healthy rising RSI {rsi:.1f}")
    else:
        return None

    # Relative Strength Gate
    if rs >= 1.0:
        score += 10
        reasons.append("Strong RS vs NIFTY")
    else:
        return None

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


def run_live_scan(stock_data, nifty):
    nifty_ind = prepare_nifty(nifty)
    market_info = {"score": 5, "regime": "🟡 UNKNOWN"}
    if not nifty_ind.empty:
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

        stop, target, risk = trade_levels(df, i, s["price"], s["atr"])

        verdict = "🟢 ELITE SNIPER SETUP"
        prev_close = float(df["Close"].iloc[-2])
        change = (s["price"] / prev_close - 1) * 100

        results.append({
            "Stock": ticker.replace(".NS", ""),
            "Score": s["score"],
            "Verdict": verdict,
            "Status": "🟢 ENTER / CONFIRM",
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
# BACKTEST ENGINE (Elite Rules)
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

        entry_index = i + 1
        entry_time = df.index[entry_index]
        entry = float(df["Open"].iloc[entry_index])

        if ENTRY_BUFFER > 0:
            entry = max(entry, signal["price"] * (1 + ENTRY_BUFFER))

        stop, target, risk = trade_levels(df, i, entry, signal["atr"])

        exit_price = None
        exit_time = None
        exit_reason = None
        bars_held = 0

        end = min(entry_index + HOLDING_BARS, len(df) - 1)

        for j in range(entry_index, end + 1):
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            if low <= stop and high >= target:
                exit_price = stop
                exit_reason = "STOP_AND_TARGET_SAME_BAR"
                exit_time = df.index[j]
                bars_held = j - entry_index + 1
                break

            if low <= stop:
                exit_price = stop
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
            exit_reason = "TIME_EXIT"

        pnl = exit_price - entry
        r_multiple = pnl / risk if risk > 0 else 0

        if r_multiple > 0:
            outcome = "WIN"
        elif r_multiple < 0:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

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
            "RS vs NIFTY": round(signal["rs"], 2),
            "5-Bar Move %": round(signal["extension"], 2),
            "Market": signal["market_regime"]
        })

        last_trade_i = i

    return trades


@st.cache_data(ttl=300, show_spinner=False)
def run_full_backtest(tickers, days):
    stock_data, download_errors = download_market_data(tickers, days)
    nifty = download_nifty(days)

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

    cumulative_r = trades["R"].cumsum()
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

st.markdown("# ⚡ Intraday Pulse Elite")
st.markdown("Uncompromising High-Probability Breakout & Momentum Sniper")

scan_tab, backtest_tab = st.tabs(["🚀 Live Elite Scanner", "📈 Elite Backtest"])

# ============================================================
# LIVE SCANNER TAB
# ============================================================

with scan_tab:
    if st.button("🚀 Run Elite Sniper Scan", use_container_width=True, type="primary"):
        with st.spinner(f"Scanning {len(MASTER_WATCHLIST)} stocks with strict institutional filters..."):
            stock_data, errors = download_market_data(tuple(MASTER_WATCHLIST), DATA_DAYS)
            nifty = download_nifty(DATA_DAYS)
            results, market_info = run_live_scan(stock_data, nifty)

        st.session_state["live_results"] = results
        st.session_state["market_info"] = market_info
        st.session_state["scan_time"] = datetime.now()

    if "live_results" not in st.session_state:
        st.info(f"Tap **Run Elite Sniper Scan** above to scan your universe with strict institutional rules.")
    else:
        results = st.session_state["live_results"]
        market_info = st.session_state["market_info"]

        st.markdown(f"### Market Regime: {market_info['regime']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(MASTER_WATCHLIST))
        c2.metric("Elite Setups", len(results))
        c3.metric("Min Score", MIN_SCORE)
        c4.metric("Target R:R", f"1 : {TARGET_R}")

        st.markdown("---")
        st.subheader("📋 Master Stock Information Table (Elite Filtered)")
        
        if results.empty:
            st.warning("No stocks meet the ultra-strict elite sniper criteria right now. Patience is part of the edge!")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Elite Results CSV",
                results.to_csv(index=False).encode("utf-8"),
                "elite_scan_results.csv",
                "text/csv",
                use_container_width=True
            )

# ============================================================
# BACKTEST TAB
# ============================================================

with backtest_tab:
    st.subheader("📈 Elite Strategy Backtest Engine")
    st.warning("Simulating strict institutional rules: 2x RVOL, mandatory bullish market regime, candle structure confirmation, and 2.5R targets.")

    if st.button("📊 Run Elite Historical Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running strict event-by-event backtest across all stocks..."):
            trades, errors = run_full_backtest(tuple(MASTER_WATCHLIST), DATA_DAYS)

        st.session_state["backtest_trades"] = trades
        st.session_state["backtest_errors"] = errors
        st.session_state["backtest_time"] = datetime.now()

    if "backtest_trades" not in st.session_state:
        st.info("Tap **Run Elite Historical Backtest** above to measure true strict-rule performance.")
    else:
        trades = st.session_state["backtest_trades"]

        if trades.empty:
            st.error("No historical trades met the ultra-strict rules. Try adjusting historical window or check market conditions.")
        else:
            stats = calculate_backtest_stats(trades)

            st.subheader("🎯 Elite Backtest Performance Metrics")
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
            equity = trades["R"].cumsum()
            st.line_chart(pd.DataFrame({"Cumulative R": equity.values}))

            st.subheader("📒 Complete Elite Trade Log")
            st.dataframe(trades, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Elite Trade Log CSV",
                trades.to_csv(index=False).encode("utf-8"),
                "elite_backtest_trade_log.csv",
                "text/csv",
                use_container_width=True
            )
