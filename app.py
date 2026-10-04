import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# INTRADAY PULSE — 30-MINUTE DAILY GATEKEEPER EDITION (Updated Rule A)
# ============================================================

st.set_page_config(page_title="Intraday Pulse", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

IST = "Asia/Kolkata"


def _ver(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


STRETCH = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}

# ---------------- Parameters (30-Min Interval) ----------------
MIN_SCORE = 65
STRONG_SCORE = 80
RVOL_THRESHOLD = 1.35
BREAKOUT_BUFFER = 0.0015
MAX_EXTENSION = 2.5
DATA_DAYS = 59
HOLDING_BARS = 12              # ~1 trading day on 30m charts (12 bars)
TARGET_R = 1.5
BACKTEST_SCORE = 80
ENTRY_BUFFER = 0.001
SKIP_OPEN_BARS = 2             # Skip first two 30m bars of the session (9:15 - 10:15 noise)
LAST_SIGNAL_TIME = "14:30"
COOLDOWN_BARS = 4
MIN_DAILY_TURNOVER = 5e7
SLIPPAGE_PCT = 0.05
COST_ROUND_TRIP_PCT = 0.10
OOS_FRACTION = 0.30
MAX_CONCURRENT_POSITIONS = 4
RISK_PER_TRADE_PCT = 0.5
DAILY_LOSS_LIMIT_R = 3.0

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
# DATA (30-Minute Interval)
# ============================================================

def clean_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_localize(IST) if df.index.tz is None else df.index.tz_convert(IST)
    df = df.dropna(subset=["Close"])
    if len(df) and df.index[-1] + pd.Timedelta(minutes=30) > pd.Timestamp.now(tz=IST):
        df = df.iloc[:-1]
    return df


@st.cache_data(ttl=60, show_spinner=False)
def download_market_data(tickers, days):
    all_data, errors = {}, []
    chunk_size = 35
    for start in range(0, len(tickers), chunk_size):
        chunk = list(tickers[start:start + chunk_size])
        try:
            data = yf.download(chunk, period=f"{days}d", interval="30m", auto_adjust=True,
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
                    df = clean_frame(df)
                    if len(df) >= 50:
                        all_data[t] = df
                    else:
                        errors.append(t)
                except Exception:
                    errors.append(t)
        except Exception:
            errors.extend(chunk)
        time.sleep(0.15)
    return all_data, sorted(set(errors))


@st.cache_data(ttl=60, show_spinner=False)
def download_nifty(days):
    for _ in range(2):
        df = yf.download("^NSEI", period=f"{days}d", interval="30m",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = clean_frame(df)
        if not df.empty:
            return df
        time.sleep(1)
    raise RuntimeError("Could not download NIFTY (^NSEI) data.")

# ============================================================
# INDICATORS & DAILY GATEKEEPER FILTER (Updated Rule A)
# ============================================================

def calculate_indicators(raw):
    df = raw[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 50:
        return pd.DataFrame()

    df["Date"] = df.index.date
    g = df.groupby("Date")
    df["BarNo"] = g.cumcount()
    first = df["BarNo"] == 0
    df["DayOpen"] = g["Open"].transform("first")
    prev_close = df["Close"].shift(1).where(~first, df["Open"])

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"] - prev_close
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    df["RSIRising"] = df["RSI"] > df["RSI"].shift(1)

    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).groupby(df["Date"]).cumsum() / \
        df["Volume"].groupby(df["Date"]).cumsum().replace(0, np.nan)

    df["Previous20High"] = df["High"].rolling(20).max().shift(1)
    df["Breakout"] = df["Close"] > df["Previous20High"] * (1 + BREAKOUT_BUFFER)
    df["FreshBreakout"] = df["Breakout"] & ~df["Breakout"].shift(1, fill_value=False)
    idx = pd.Series(np.arange(len(df)), index=df.index)
    last_fresh = idx.where(df["FreshBreakout"]).ffill()
    df["BreakoutAge"] = (idx - last_fresh).fillna(999).astype(int)

    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    df["BodyPct"] = (df["Close"] - df["Open"]).abs() / rng
    df["CloseLocation"] = (df["Close"] - df["Low"]) / rng
    df["UpperWickPct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / rng
    df["StrongCandle"] = ((df["Close"] > df["Open"]) & (df["BodyPct"] >= 0.5) &
                          (df["CloseLocation"] >= 0.7) & (df["UpperWickPct"] <= 0.3))
    green = df["Close"] > df["Open"]
    df["TwoGreen"] = green & green.shift(1, fill_value=False)

    df["BarTime"] = df.index.strftime("%H:%M")
    ref = df.groupby("BarTime")["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["RVOL"] = df["Volume"] / ref.replace(0, np.nan)

    ref5 = g["Close"].shift(5).fillna(df["DayOpen"])
    df["Return5"] = (df["Close"] / ref5 - 1) * 100

    # --- DAILY GATEKEEPER FILTER (Updated Rules) ---
    daily_df = g.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    
    # Rule A: Yesterday's Close > Day-Before-Yesterday's High
    yesterday_close = daily_df["Close"].shift(1)
    day_before_yesterday_high = daily_df["High"].shift(2)
    rule_a = yesterday_close > day_before_yesterday_high

    # Rule B: Yesterday's High > Day-Before-Yesterday's High (breaking yesterday's high)
    yesterday_high = daily_df["High"].shift(1)
    rule_b = yesterday_high > day_before_yesterday_high

    daily_df["AllowedToday"] = (rule_a | rule_b).shift(1).fillna(False)
    df["DailyAllowed"] = df["Date"].map(daily_df["AllowedToday"].to_dict()).fillna(False)
    return df


def prepare_nifty(raw):
    n = calculate_indicators(raw)
    if n.empty:
        return n
    n["MScore"] = (n["Close"] > n["EMA20"]).astype(int) * 5 + (n["Close"] > n["DayOpen"]).astype(int) * 5
    return n


def regime_label(score):
    return "🟢 BULLISH" if score >= 10 else "🟡 NEUTRAL" if score >= 5 else "🔴 BEARISH"


def components(df):
    c = df
    return [
        ("Above EMA20", c["Close"] > c["EMA20"], 10, "Below EMA20"),
        ("Above VWAP", c["Close"] > c["VWAP"], 10, "Below VWAP"),
        ("20-bar breakout", c["Breakout"], 15, "No breakout"),
        ("Fresh breakout", c["FreshBreakout"], 5, "Not a fresh breakout"),
        (f"Volume expansion (RVOL >= {RVOL_THRESHOLD})", c["RVOL"] >= RVOL_THRESHOLD, 15, "Normal/low volume"),
        ("Healthy RSI (50-75)", c["RSI"].between(50, 75), 8, "RSI outside 50-75"),
        ("RSI rising", c["RSIRising"], 4, "RSI not rising"),
        ("Two green bars", c["TwoGreen"], 5, "Single-bar move"),
        ("Strong closing range", c["StrongCandle"], 3, "Weak candle"),
        ("Positive RS vs NIFTY", c["RS"] >= 0.5, 10, "Weak RS vs NIFTY"),
        ("Supportive market", c["MScore"] >= 5, 5, "Weak market"),
        ("Daily Gatekeeper Passed", c["DailyAllowed"], 10, "Daily filter failed"),
    ]

MAX_RAW = 100


def build_frame(raw, nifty_ind):
    df = calculate_indicators(raw)
    if df.empty:
        return df

    daily_turnover = (df["Close"] * df["Volume"]).groupby(df["Date"]).sum()
    liquid_days = daily_turnover.shift(1).rolling(10, min_periods=3).median() >= MIN_DAILY_TURNOVER
    df["Liquid"] = df["Date"].isin(set(liquid_days[liquid_days].index))
    if not df["Liquid"].any():
        return pd.DataFrame()

    nf = nifty_ind[["Return5", "MScore"]].reindex(df.index, method="ffill")
    df["RS"] = df["Return5"] - nf["Return5"]
    df["MScore"] = nf["MScore"]

    raw_pts = sum(pts * m.astype(int) for _, m, pts, _ in components(df))
    df["Score"] = (raw_pts / MAX_RAW * 100).round().astype(int)

    need = ["EMA20", "VWAP", "RSI", "ATR", "RVOL", "Previous20High", "Return5", "RS", "MScore", "DailyAllowed"]
    df["Valid"] = df[need].notna().all(axis=1) & (df["BarNo"] >= SKIP_OPEN_BARS) & df["Liquid"] & df["DailyAllowed"]
    df["InWindow"] = df["BarTime"] <= LAST_SIGNAL_TIME
    df["Extended"] = df["Return5"] > MAX_EXTENSION
    df["Signal"] = df["Valid"] & df["Breakout"] & ~df["Extended"] & df["InWindow"]
    return df


def trade_levels(df, i, entry, atr):
    swing_low = float(df["Low"].iloc[max(0, i - 5):i + 1].min())
    prev_high = df["Previous20High"].iloc[i]
    structure = swing_low if pd.isna(prev_high) else min(swing_low, float(prev_high) - 0.5 * atr)
    stop = max(structure, entry - 1.5 * atr)
    stop = min(stop, entry * 0.995)
    risk = entry - stop
    return float(stop), float(entry + TARGET_R * risk), float(risk)

# ============================================================
# LIVE SCAN
# ============================================================

def run_live_scan(stock_data, nifty_raw):
    nifty_ind = prepare_nifty(nifty_raw)
    if nifty_ind.empty:
        raise RuntimeError("Not enough NIFTY data.")
    asof = nifty_ind.index[-1]
    ms = int(nifty_ind["MScore"].iloc[-1])
    market_info = {"score": ms, "regime": regime_label(ms), "asof": asof}

    rows, skipped = [], []
    for ticker, raw in stock_data.items():
        df = build_frame(raw, nifty_ind)
        if df.empty:
            skipped.append(ticker)
            continue
        i = len(df) - 1
        last = df.iloc[i]
        fresh_bar = df.index[i] == asof
        if not fresh_bar or not last["Valid"]:
            skipped.append(ticker)
            continue

        comps = components(df)
        reasons = [lab for lab, m, _, _ in comps if bool(m.iloc[-1])]
        risks = [rl for _, m, _, rl in comps if not bool(m.iloc[-1])]

        score = int(last["Score"])
        if last["Signal"] and score >= STRONG_SCORE:
            status = "🟢 ENTER ON TRIGGER"
        elif last["Signal"] and score >= MIN_SCORE:
            status = "🟡 WATCH BREAKOUT"
        else:
            status = "🔴 AVOID"

        close = float(last["Close"])
        trigger = close * (1 + ENTRY_BUFFER)
        stop, target, risk = trade_levels(df, i, trigger, float(last["ATR"]))
        prev_days = df.loc[df["Date"] < last["Date"], "Close"]
        change = (close / float(prev_days.iloc[-1]) - 1) * 100 if len(prev_days) else np.nan

        rows.append({
            "Stock": ticker.replace(".NS", ""), "Score": score, "Status": status,
            "Price": round(close, 2), "Change % (day)": round(change, 2),
            "Buy-Stop Trigger": round(trigger, 2), "Stop Loss": round(stop, 2),
            "Target": round(target, 2), "Risk/Share": round(risk, 2), "R:R": TARGET_R,
            "RSI": round(float(last["RSI"]), 1), "RVOL": round(float(last["RVOL"]), 2),
            "RS vs NIFTY": round(float(last["RS"]), 2),
            "Breakout": "YES" if last["Breakout"] else "NO",
            "Why Score?": ", ".join(reasons), "Risks": ", ".join(risks) or "None",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Score", "RVOL", "RS vs NIFTY"], ascending=False).reset_index(drop=True)
    return out, market_info, sorted(skipped)

# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def collect_signals(ticker, df):
    o, h = df["Open"].values, df["High"].values
    c, atr, dates = df["Close"].values, df["ATR"].values, df["Date"].values
    sig = (df["Signal"] & (df["Score"] >= BACKTEST_SCORE)).values
    n, out = len(df), []
    for i in range(20, n - 1):
        if not sig[i]:
            continue
        j = i + 1
        if dates[j] != dates[i]:
            continue
        trigger = c[i] * (1 + ENTRY_BUFFER)
        if o[j] >= trigger:
            fill = o[j]
        elif h[j] >= trigger:
            fill = trigger
        else:
            continue
        fill *= 1 + SLIPPAGE_PCT / 100
        stop, target, risk = trade_levels(df, i, fill, atr[i])
        k_end = j
        while k_end < n - 1 and k_end - j + 1 < HOLDING_BARS and dates[k_end + 1] == dates[j]:
            k_end += 1
        out.append({"ticker": ticker.replace(".NS", ""), "signal_time": df.index[i],
                    "entry_time": df.index[j], "entry_idx": j, "k_end": k_end,
                    "score": int(df["Score"].iloc[i]), "fill": fill, "stop": stop,
                    "target": target, "risk": risk, "df": df, "raw_i": i})
    return out


def simulate_trade(sig):
    df, j, k_end = sig["df"], sig["entry_idx"], sig["k_end"]
    stop, target, fill, risk = sig["stop"], sig["target"], sig["fill"], sig["risk"]
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])

    exit_price = reason = None
    exit_k = k_end
    for k in range(j, k_end + 1):
        if k > j and o[k] <= stop:
            exit_price, reason, exit_k = o[k], "GAP_STOP", k
            break
        if l[k] <= stop:
            exit_price, exit_k = stop, k
            reason = "STOP_AND_TARGET_SAME_BAR" if h[k] >= target else "STOP"
            break
        if h[k] >= target:
            exit_price, reason, exit_k = target, "TARGET", k
            break
    if exit_price is None:
        exit_price = c[k_end]
        reason = "TIME_EXIT" if k_end - j + 1 >= HOLDING_BARS else "EOD_EXIT"
    if reason != "TARGET":
        exit_price *= 1 - SLIPPAGE_PCT / 100

    pnl = exit_price - fill - fill * COST_ROUND_TRIP_PCT / 100
    return exit_price, reason, exit_k, pnl, (pnl / risk if risk > 0 else 0.0)


def run_full_backtest(tickers, days):
    stock_data, dl_errors = download_market_data(tickers, days)
    nifty_ind = prepare_nifty(download_nifty(days))
    errors, skipped, signals = list(dl_errors), [], []

    for ticker, raw in stock_data.items():
        try:
            df = build_frame(raw, nifty_ind)
            if df.empty:
                skipped.append(ticker)
                continue
            signals.extend(collect_signals(ticker, df))
        except Exception:
            errors.append(ticker)

    signals.sort(key=lambda s: (s["entry_time"], -s["score"]))

    active, trades, cooldown, day_book = [], [], {}, {}
    for sig in signals:
        et, tk = sig["entry_time"], sig["ticker"]
        active = [x for x in active if x > et]
        if cooldown.get(tk, et) > et:
            continue
        if len(active) >= MAX_CONCURRENT_POSITIONS:
            continue
        day = et.date()
        realized = sum(r for x_exit, r in day_book.get(day, []) if x_exit <= et)
        if realized <= -DAILY_LOSS_LIMIT_R:
            continue

        df, j = sig["df"], sig["entry_idx"]
        exit_price, reason, exit_k, pnl, r = simulate_trade(sig)
        exit_time = df.index[exit_k]
        active.append(exit_time)
        cooldown[tk] = df.index[min(len(df) - 1, exit_k + COOLDOWN_BARS)]
        day_book.setdefault(day, []).append((exit_time, r))

        row = df.iloc[sig["raw_i"]]
        trades.append({
            "Stock": tk, "Signal Time": sig["signal_time"], "Entry Time": et, "Exit Time": exit_time,
            "Score": sig["score"], "Entry": round(sig["fill"], 2), "Stop": round(sig["stop"], 2),
            "Target": round(sig["target"], 2), "Exit": round(exit_price, 2),
            "Risk %": round(sig["risk"] / sig["fill"] * 100, 2), "R (net)": round(r, 3),
            "P&L/Share (net)": round(pnl, 2),
            "Outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "BREAKEVEN",
            "Exit Reason": reason, "Bars Held": exit_k - j + 1,
            "RSI": round(float(row["RSI"]), 1), "RVOL": round(float(row["RVOL"]), 2),
            "RS vs NIFTY": round(float(row["RS"]), 2),
            "Market": regime_label(int(row["MScore"])),
        })

    out = pd.DataFrame(trades)
    if not out.empty:
        out = out.sort_values("Signal Time").reset_index(drop=True)
    return out, sorted(set(errors)), sorted(set(skipped))

# ============================================================
# STATS
# ============================================================

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half) * 100, (centre + half) * 100


def daily_r(trades):
    return trades.groupby(trades["Entry Time"].dt.date)["R (net)"].sum()


def calculate_backtest_stats(trades):
    if trades.empty:
        return {}
    t = trades.sort_values("Exit Time")
    r, n = t["R (net)"], len(t)
    wins, losses = int((r > 0).sum()), int((r < 0).sum())
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.concatenate([[0.0], r.cumsum().values])
    max_dd = -(eq - np.maximum.accumulate(eq)).min()
    lo, hi = wilson_ci(wins, n)
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

    d = daily_r(t)
    nd = len(d)
    dse = d.std(ddof=1) / np.sqrt(nd) if nd > 1 else np.nan
    return {
        "Trades": n, "Wins": wins, "Losses": losses,
        "Win Rate %": wins / n * 100, "Win Rate 95% CI": f"{lo:.0f}-{hi:.0f}%",
        "Avg Win R": r[r > 0].mean() if wins else 0.0,
        "Avg Loss R": r[r < 0].mean() if losses else 0.0,
        "Profit Factor": gp / gl if gl > 0 else np.inf,
        "Expectancy R": r.mean(),
        "Per-trade t-stat": r.mean() / se if se and se > 0 else np.nan,
        "Trading Days": nd,
        "Daily t-stat": d.mean() / dse if dse and dse > 0 else np.nan,
        "Winning Days %": (d > 0).mean() * 100,
        "Worst Day R": d.min(), "Best Day R": d.max(),
        "Total R": r.sum(), "Max Drawdown R": max_dd,
        "Return % (non-compounded)": r.sum() * RISK_PER_TRADE_PCT,
        "Max Drawdown %": max_dd * RISK_PER_TRADE_PCT,
    }


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)


def split_stats(trades):
    days = sorted(trades["Signal Time"].dt.date.unique())
    parts = {"All days": trades}
    if len(days) >= 10:
        cut = days[int(len(days) * (1 - OOS_FRACTION))]
        d = trades["Signal Time"].dt.date
        parts["First 70% (in-sample)"] = trades[d < cut]
        parts["Last 30% (out-of-sample)"] = trades[d >= cut]
    cols = {}
    for name, part in parts.items():
        s = calculate_backtest_stats(part)
        if s:
            cols[name] = {k: _fmt(v) for k, v in s.items()}
    return pd.DataFrame(cols)

# ============================================================
# UI
# ============================================================

st.markdown("# ⚡ Intraday Pulse")
st.caption("30-Minute Interval + Daily Gatekeeper Edition (Rule A: Yesterday Close > Day Before High).")

scan_tab, backtest_tab = st.tabs(["🚀 Live Scanner", "📈 Backtest"])

with scan_tab:
    if st.button("🚀 Run Instant Market Scan", type="primary", key="scan", **STRETCH):
        try:
            with st.spinner(f"Scanning {len(MASTER_WATCHLIST)} 30-minute charts with Daily Filter..."):
                stock_data, dl_errors = download_market_data(tuple(MASTER_WATCHLIST), DATA_DAYS)
                nifty = download_nifty(DATA_DAYS)
                results, market_info, skipped = run_live_scan(stock_data, nifty)
            st.session_state.update(live_results=results, market_info=market_info,
                                    live_errors=dl_errors, live_skipped=skipped,
                                    scan_time=datetime.now())
        except Exception as e:
            st.error(f"Scan failed: {e}")

    if "live_results" not in st.session_state:
        st.info("Tap **Run Instant Market Scan** to analyse the 30-minute watchlist.")
    else:
        results, mi = st.session_state["live_results"], st.session_state["market_info"]
        now = pd.Timestamp.now(tz=IST)
        asof = mi["asof"]
        st.markdown(f"### Market Regime: {mi['regime']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Watchlist", len(MASTER_WATCHLIST))
        c2.metric("Analysed", len(results))
        c3.metric("Strong Setups", int((results["Status"] == "🟢 ENTER ON TRIGGER").sum()) if len(results) else 0)
        c4.metric("Breakouts", int((results["Breakout"] == "YES").sum()) if len(results) else 0)

        if results.empty:
            st.warning("No stocks passed the daily gatekeeper and intraday breakout filters.")
        else:
            st.dataframe(results, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Results CSV", results.to_csv(index=False).encode("utf-8"),
                               "30m_scan_results.csv", "text/csv", **STRETCH)

with backtest_tab:
    st.subheader("📈 30-Minute Strategy Backtest")
    st.warning("Net of costs. 30-minute interval with the Daily Gatekeeper pre-filter enabled.")

    if st.button("📊 Run Historical Backtest", type="primary", key="bt", **STRETCH):
        try:
            with st.spinner("Running 30-minute portfolio simulation..."):
                trades, errors, skipped = run_full_backtest(tuple(MASTER_WATCHLIST), DATA_DAYS)
            st.session_state.update(backtest_trades=trades, bt_errors=errors, bt_skipped=skipped)
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if "backtest_trades" not in st.session_state:
        st.info("Tap **Run Historical Backtest** to measure performance.")
    else:
        trades = st.session_state["backtest_trades"]
        if trades.empty:
            st.error("No historical trades matched the rules.")
        else:
            s = calculate_backtest_stats(trades)
            m = st.columns(4)
            m[0].metric("Win Rate", f"{s['Win Rate %']:.1f}%")
            m[1].metric("Trades", s["Trades"])
            m[2].metric("Expectancy (net)", f"{s['Expectancy R']:+.3f}R")
            m[3].metric("Profit Factor", f"{s['Profit Factor']:.2f}" if np.isfinite(s["Profit Factor"]) else "∞")
            m = st.columns(4)
            m[0].metric("Total R", f"{s['Total R']:+.2f}R")
            m[1].metric("Avg Win / Loss", f"{s['Avg Win R']:+.2f} / {s['Avg Loss R']:+.2f}R")
            m[2].metric("Max Drawdown", f"{s['Max Drawdown R']:.2f}R")
            m[3].metric("Daily t-stat", f"{s['Daily t-stat']:.2f}")

            st.subheader("Cumulative return % (non-compounded)")
            eq = trades.sort_values("Exit Time")["R (net)"].cumsum() * RISK_PER_TRADE_PCT
            st.line_chart(pd.DataFrame({"Cumulative return %": eq.values}))

            st.subheader("📒 Trade Log")
            st.dataframe(trades, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Trade Log CSV", trades.to_csv(index=False).encode("utf-8"),
                               "30m_backtest_trade_log.csv", "text/csv", **STRETCH)
