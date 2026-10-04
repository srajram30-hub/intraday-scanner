import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# INTRADAY PULSE — DAILY SWING EDITION (Higher Highs / Lower Lows)
# ============================================================

st.set_page_config(page_title="Daily Swing Pulse", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

IST = "Asia/Kolkata"


def _ver(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


STRETCH = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}

# ---------------- Daily Swing Parameters ----------------
DATA_PERIOD = "2y"             # 2 years of daily data for robust swing backtesting
EMA_TREND = 20                 # Daily 20 EMA baseline
HOLDING_DAYS = 15              # Max swing holding days
TARGET_R = 2.0                 # 2:1 Reward-to-Risk target for daily swings
BACKTEST_SCORE = 75
SLIPPAGE_PCT = 0.05            # Daily slippage is minimal
COST_ROUND_TRIP_PCT = 0.15     # Delivery / Swing brokerage + STT estimate
OOS_FRACTION = 0.30
MAX_CONCURRENT_POSITIONS = 5
RISK_PER_TRADE_PCT = 1.0       # 1% capital risked per swing trade

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
    "SHRIRAMFIN.NS", "MARUTI.NS", "BAJAJ-AUTO.NS",
]

# ============================================================
# DATA DOWNLOAD (Daily Interval)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_daily_data(tickers, period):
    all_data, errors = {}, []
    chunk_size = 35
    for start in range(0, len(tickers), chunk_size):
        chunk = list(tickers[start:start + chunk_size])
        try:
            data = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
            if data.empty:
                errors.extend(chunk)
                continue
            for t in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data.columns.get_level_values(0):
                            df = data[t].copy()
                        elif t in data.columns.get_level_values(1):
                            df = data.xs(t, axis=1, level=1).copy()
                        else:
                            errors.append(t)
                            continue
                    else:
                        df = data.copy()
                    
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 100:
                        all_data[t] = df
                    else:
                        errors.append(t)
                except Exception:
                    errors.append(t)
        except Exception:
            errors.extend(chunk)
        time.sleep(0.15)
    return all_data, sorted(set(errors))

# ============================================================
# DAILY INDICATORS & MARKET STRUCTURE (HH / HL / LL / LH)
# ============================================================

def calculate_daily_indicators(raw):
    df = raw[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 50:
        return pd.DataFrame()

    df["EMA20"] = df["Close"].ewm(span=EMA_TREND, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # --- MARKET STRUCTURE: Higher Highs / Higher Lows & Lower Lows / Lower Highs ---
    # Rolling 5-day local peaks and troughs
    df["SwingHigh"] = df["High"].rolling(5, center=True).max()
    df["SwingLow"] = df["Low"].rolling(5, center=True).min()
    
    # Forward fill swing levels for trend classification
    df["PrevSwingHigh"] = df["SwingHigh"].shift(5)
    df["PrevSwingLow"] = df["SwingLow"].shift(5)

    # Structure conditions
    df["HigherHigh"] = df["High"] > df["PrevSwingHigh"]
    df["HigherLow"] = df["Low"] > df["PrevSwingLow"]
    df["LowerLow"] = df["Low"] < df["PrevSwingLow"]
    df["LowerHigh"] = df["High"] < df["PrevSwingHigh"]

    # --- LONG SWING SETUP (Uptrend + Pullback to EMA20 + Resumption) ---
    df["LongTrend"] = (df["Close"] > df["EMA20"]) & df["HigherHigh"] & df["HigherLow"]
    near_ema_l = (df["Low"] <= df["EMA20"] * 1.01) & (df["Low"] >= df["EMA20"] * 0.98)
    df["LongPullback"] = near_ema_l | (df["Low"] <= df["Low"].rolling(3).min() * 1.005)
    df["LongResumption"] = (df["Close"] > df["Open"]) & (df["Close"] > df["High"].shift(1))
    df["LongSignal"] = df["LongTrend"] & df["LongPullback"] & df["LongResumption"]

    # --- SHORT SWING SETUP (Downtrend + Rally to EMA20 resistance + Resumption) ---
    df["ShortTrend"] = (df["Close"] < df["EMA20"]) & df["LowerLow"] & df["LowerHigh"]
    near_ema_s = (df["High"] >= df["EMA20"] * 0.99) & (df["High"] <= df["EMA20"] * 1.02)
    df["ShortPullback"] = near_ema_s | (df["High"] >= df["High"].rolling(3).max() * 0.995)
    df["ShortResumption"] = (df["Close"] < df["Open"]) & (df["Close"] < df["Low"].shift(1))
    df["ShortSignal"] = df["ShortTrend"] & df["ShortPullback"] & df["ShortResumption"]

    return df


def build_daily_frame(raw, mode):
    df = calculate_daily_indicators(raw)
    if df.empty:
        return pd.DataFrame()

    if mode == "🟢 BULLISH (Long Swings Only)":
        df["Signal"] = df["LongSignal"]
        df["Direction"] = "LONG"
    elif mode == "🔴 BEARISH (Short Swings Only)":
        df["Signal"] = df["ShortSignal"]
        df["Direction"] = "SHORT"
    else:
        df["Signal"] = False
        df["Direction"] = "NONE"

    df["Score"] = 80
    need = ["EMA20", "RSI", "ATR"]
    df["Valid"] = df[need].notna().all(axis=1) & df["Signal"]
    return df


def swing_trade_levels(direction, df, i, entry, atr):
    if direction == "LONG":
        swing_low = float(df["Low"].iloc[max(0, i - 5):i + 1].min())
        stop = min(swing_low - 0.2 * atr, entry - 1.5 * atr)
        risk = entry - stop
        target = entry + TARGET_R * risk
    else:
        swing_high = float(df["High"].iloc[max(0, i - 5):i + 1].max())
        stop = max(swing_high + 0.2 * atr, entry + 1.5 * atr)
        risk = stop - entry
        target = entry - TARGET_R * risk
    return float(stop), float(target), float(risk)

# ============================================================
# LIVE SCANNER (Daily)
# ============================================================

def run_daily_scan(stock_data, mode):
    rows, skipped = [], []
    for ticker, raw in stock_data.items():
        df = build_daily_frame(raw, mode)
        if df.empty:
            skipped.append(ticker)
            continue
        i = len(df) - 1
        last = df.iloc[i]
        if not last["Valid"]:
            skipped.append(ticker)
            continue

        direction = last["Direction"]
        close = float(last["Close"])
        stop, target, risk = swing_trade_levels(direction, df, i, close, float(last["ATR"]))

        rows.append({
            "Stock": ticker.replace(".NS", ""), "Direction": direction,
            "Close Price": round(close, 2), "Stop Loss": round(stop, 2),
            "Target": round(target, 2), "Risk/Share": round(risk, 2), "R:R": TARGET_R,
            "RSI": round(float(last["RSI"]), 1),
            "Structure": "Higher High / Higher Low" if direction == "LONG" else "Lower Low / Lower High"
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Stock").reset_index(drop=True)
    return out, sorted(skipped)

# ============================================================
# PORTFOLIO BACKTEST (Daily Swing Simulation)
# ============================================================

def run_daily_backtest(tickers, period, mode):
    stock_data, dl_errors = download_daily_data(tickers, period)
    errors, skipped, signals = list(dl_errors), [], []

    for ticker, raw in stock_data.items():
        try:
            df = build_daily_frame(raw, mode)
            if df.empty:
                skipped.append(ticker)
                continue

            o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
            atr = df["ATR"].values
            sig = df["Valid"].values
            direction = df["Direction"].iloc[0]
            n = len(df)

            for i in range(50, n - 1):
                if not sig[i]:
                    continue
                entry_idx = i + 1
                entry = o[entry_idx] * (1 + SLIPPAGE_PCT / 100 if direction == "LONG" else 1 - SLIPPAGE_PCT / 100)
                stop, target, risk = swing_trade_levels(direction, df, i, entry, atr[i])

                exit_price = reason = None
                exit_idx = min(n - 1, entry_idx + HOLDING_DAYS)

                for k in range(entry_idx, exit_idx + 1):
                    bar_h, bar_l, bar_o = h[k], l[k], o[k]
                    if direction == "LONG":
                        if bar_l <= stop:
                            exit_price = min(stop, bar_o)
                            reason = "STOP"
                            exit_idx = k
                            break
                        if bar_h >= target:
                            exit_price = target
                            reason = "TARGET"
                            exit_idx = k
                            break
                    else:
                        if bar_h >= stop:
                            exit_price = max(stop, bar_o)
                            reason = "STOP"
                            exit_idx = k
                            break
                        if bar_l <= target:
                            exit_price = target
                            reason = "TARGET"
                            exit_idx = k
                            break

                if exit_price is None:
                    exit_idx = min(n - 1, entry_idx + HOLDING_DAYS)
                    exit_price = c[exit_idx]
                    reason = "TIME_EXIT"

                if direction == "LONG":
                    pnl = exit_price - entry - entry * COST_ROUND_TRIP_PCT / 100
                else:
                    pnl = entry - exit_price - entry * COST_ROUND_TRIP_PCT / 100

                r = pnl / risk if risk > 0 else 0.0

                signals.append({
                    "ticker": ticker.replace(".NS", ""), "direction": direction,
                    "signal_date": df.index[i].date(), "entry_date": df.index[entry_idx].date(),
                    "exit_date": df.index[exit_idx].date(), "entry": round(entry, 2),
                    "stop": round(stop, 2), "target": round(target, 2), "exit": round(exit_price, 2),
                    "R (net)": round(r, 3), "Outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "BREAKEVEN",
                    "Exit Reason": reason, "Days Held": exit_idx - entry_idx + 1
                })
        except Exception:
            errors.append(ticker)

    trades_df = pd.DataFrame(signals)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    return trades_df, sorted(set(errors)), sorted(set(skipped))

# ============================================================
# STATS & UI
# ============================================================

def wilson_ci(wins, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half) * 100, (centre + half) * 100


def calculate_stats(trades):
    if trades.empty: return {}
    t = trades.sort_values("exit_date")
    r, n = t["R (net)"], len(t)
    wins, losses = int((r > 0).sum()), int((r < 0).sum())
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.concatenate([[0.0], r.cumsum().values])
    max_dd = -(eq - np.maximum.accumulate(eq)).min()
    lo, hi = wilson_ci(wins, n)
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

    return {
        "Trades": n, "Wins": wins, "Losses": losses,
        "Win Rate %": wins / n * 100, "Win Rate 95% CI": f"{lo:.0f}-{hi:.0f}%",
        "Avg Win R": r[r > 0].mean() if wins else 0.0,
        "Avg Loss R": r[r < 0].mean() if losses else 0.0,
        "Profit Factor": gp / gl if gl > 0 else np.inf,
        "Expectancy R": r.mean(), "Per-trade t-stat": r.mean() / se if se and se > 0 else np.nan,
        "Total R": r.sum(), "Max Drawdown R": max_dd,
        "Return % (non-compounded)": r.sum() * RISK_PER_TRADE_PCT,
    }


st.markdown("# 📈 Daily Swing Pulse")
st.caption("Daily Candle Swing Trading Edition with Market Structure (Higher Highs / Lower Lows).")

market_mode = st.selectbox(
    "🌐 Swing Market Direction (Manual Control)",
    [
        "🟢 BULLISH (Long Swings Only)",
        "🔴 BEARISH (Short Swings Only)"
    ]
)

scan_tab, backtest_tab = st.tabs(["🚀 EOD Daily Scan", "📈 Swing Backtest"])

with scan_tab:
    if st.button("🚀 Run Daily Swing Scan", type="primary", key="scan", **STRETCH):
        try:
            with st.spinner(f"Scanning {len(MASTER_WATCHLIST)} daily charts..."):
                stock_data, dl_errors = download_daily_data(tuple(MASTER_WATCHLIST), DATA_PERIOD)
                results, skipped = run_daily_scan(stock_data, market_mode)
            st.session_state.update(daily_results=results, daily_skipped=skipped, scan_time=datetime.now())
        except Exception as e:
            st.error(f"Scan failed: {e}")

    if "daily_results" not in st.session_state:
        st.info("Select your market direction above and tap **Run Daily Swing Scan**.")
    else:
        results = st.session_state["daily_results"]
        c1, c2 = st.columns(2)
        c1.metric("Watchlist", len(MASTER_WATCHLIST))
        c2.metric("Qualifying Setups", len(results))

        if results.empty:
            st.warning("No daily swing setups match the current structure filter.")
        else:
            st.dataframe(results, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Daily Setups CSV", results.to_csv(index=False).encode("utf-8"),
                               "daily_swing_setups.csv", "text/csv", **STRETCH)

with backtest_tab:
    st.subheader("📈 Multi-Day Swing Strategy Backtest")
    st.warning(f"Simulating daily swing trades over {DATA_PERIOD}. Mode: {market_mode}. Target R: {TARGET_R}.")

    if st.button("📊 Run Swing Backtest", type="primary", key="bt", **STRETCH):
        try:
            with st.spinner("Running daily swing simulation..."):
                trades, errors, skipped = run_daily_backtest(tuple(MASTER_WATCHLIST), DATA_PERIOD, market_mode)
            st.session_state.update(swing_trades=trades)
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if "swing_trades" not in st.session_state:
        st.info("Tap **Run Swing Backtest** to measure multi-day performance.")
    else:
        trades = st.session_state["swing_trades"]
        if trades.empty:
            st.error("No historical swing trades matched the rules.")
        else:
            s = calculate_stats(trades)
            m = st.columns(4)
            m[0].metric("Win Rate", f"{s['Win Rate %']:.1f}%")
            m[1].metric("Trades", s["Trades"])
            m[2].metric("Expectancy (net)", f"{s['Expectancy R']:+.3f}R")
            m[3].metric("Profit Factor", f"{s['Profit Factor']:.2f}" if np.isfinite(s["Profit Factor"]) else "∞")

            m = st.columns(4)
            m[0].metric("Total R", f"{s['Total R']:+.2f}R")
            m[1].metric("Avg Win / Loss", f"{s['Avg Win R']:+.2f} / {s['Avg Loss R']:+.2f}R")
            m[2].metric("Max Drawdown", f"{s['Max Drawdown R']:.2f}R")
            m[3].metric("Return %", f"{s['Return % (non-compounded)']:+.1f}%")

            st.subheader("Cumulative Return % (Non-compounded)")
            eq = trades.sort_values("exit_date")["R (net)"].cumsum() * RISK_PER_TRADE_PCT
            st.line_chart(pd.DataFrame({"Cumulative return %": eq.values}))

            st.subheader("📒 Swing Trade Log")
            st.dataframe(trades, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Swing Log CSV", trades.to_csv(index=False).encode("utf-8"),
                               "swing_backtest_log.csv", "text/csv", **STRETCH)
