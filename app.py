import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# INTRADAY PULSE — v8 (Bi-Directional Long/Short & 3-Way Regime Filter)
# ============================================================

st.set_page_config(page_title="Intraday Pulse", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

IST = "Asia/Kolkata"


def _ver(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


STRETCH = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}

# ---------------- Parameters ----------------
MIN_SCORE = 65
STRONG_SCORE = 80
RVOL_THRESHOLD = 1.20
MAX_EXTENSION = 3.0
DATA_DAYS = 59
HOLDING_BARS = 8
TARGET_R = 1.2
BACKTEST_SCORE = 80
ENTRY_BUFFER = 0.0005
SKIP_OPEN_BARS = 3
LAST_SIGNAL_TIME = "14:15"
COOLDOWN_BARS = 3
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
# DATA
# ============================================================

def clean_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_localize(IST) if df.index.tz is None else df.index.tz_convert(IST)
    df = df.dropna(subset=["Close"])
    if len(df) and df.index[-1] + pd.Timedelta(minutes=15) > pd.Timestamp.now(tz=IST):
        df = df.iloc[:-1]
    return df


@st.cache_data(ttl=60, show_spinner=False)
def download_market_data(tickers, days):
    all_data, errors = {}, []
    chunk_size = 35
    for start in range(0, len(tickers), chunk_size):
        chunk = list(tickers[start:start + chunk_size])
        try:
            data = yf.download(chunk, period=f"{days}d", interval="15m", auto_adjust=True,
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
                    if len(df) >= 60:
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
        df = yf.download("^NSEI", period=f"{days}d", interval="15m",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = clean_frame(df)
        if not df.empty:
            return df
        time.sleep(1)
    raise RuntimeError("Could not download NIFTY (^NSEI) data.")

# ============================================================
# INDICATORS & BI-DIRECTIONAL PULLBACK LOGIC
# ============================================================

def calculate_indicators(raw):
    df = raw[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 60:
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

    # --- LONG PULLBACK SETUP ---
    df["InUptrend"] = (df["Close"] > df["EMA20"]) & (df["Close"] > df["VWAP"])
    df["Previous20High"] = df["High"].rolling(20).max().shift(1)
    df["RecentBreakout"] = (df["High"].rolling(15).max() > df["Previous20High"])
    near_vwap_l = (df["Low"].rolling(3).min() <= df["VWAP"] * 1.005)
    near_ema_l = (df["Low"].rolling(3).min() <= df["EMA20"] * 1.005)
    df["InPullbackZoneLong"] = near_vwap_l | near_ema_l
    df["GreenBar"] = df["Close"] > df["Open"]
    df["LongResumption"] = df["GreenBar"] & (df["Close"] > df["High"].shift(1))
    df["LongSignal"] = df["InUptrend"] & df["RecentBreakout"] & df["InPullbackZoneLong"] & df["LongResumption"]

    # --- SHORT PULLBACK SETUP ---
    df["InDowntrend"] = (df["Close"] < df["EMA20"]) & (df["Close"] < df["VWAP"])
    df["Previous20Low"] = df["Low"].rolling(20).min().shift(1)
    df["RecentBreakdown"] = (df["Low"].rolling(15).min() < df["Previous20Low"])
    near_vwap_s = (df["High"].rolling(3).max() >= df["VWAP"] * 0.995)
    near_ema_s = (df["High"].rolling(3).max() >= df["EMA20"] * 0.995)
    df["InPullbackZoneShort"] = near_vwap_s | near_ema_s
    df["RedBar"] = df["Close"] < df["Open"]
    df["ShortResumption"] = df["RedBar"] & (df["Close"] < df["Low"].shift(1))
    df["ShortSignal"] = df["InDowntrend"] & df["RecentBreakdown"] & df["InPullbackZoneShort"] & df["ShortResumption"]

    df["BarTime"] = df.index.strftime("%H:%M")
    ref = df.groupby("BarTime")["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["RVOL"] = df["Volume"] / ref.replace(0, np.nan)

    ref5 = g["Close"].shift(5).fillna(df["DayOpen"])
    df["Return5"] = (df["Close"] / ref5 - 1) * 100
    return df


def prepare_nifty(raw):
    n = calculate_indicators(raw)
    if n.empty:
        return n
    # Regime scoring for Nifty
    price = n["Close"]
    ema = n["EMA20"]
    vwap = n["VWAP"]
    
    # 3-way regime classification per bar
    # Bullish: price > EMA20 and price > VWAP
    # Bearish: price < EMA20 and price < VWAP
    # Choppy: otherwise (sideways / straddling EMA20)
    regimes = []
    for c, e, v in zip(price, ema, vwap):
        if pd.isna(c) or pd.isna(e) or pd.isna(v):
            regimes.append("🟡 CHOPPY")
        elif c > e and c > v:
            regimes.append("🟢 BULLISH")
        elif c < e and c < v:
            regimes.append("🔴 BEARISH")
        else:
            regimes.append("🟡 CHOPPY")
    n["MarketRegime"] = regimes
    n["MScore"] = (n["MarketRegime"] == "🟢 BULLISH").astype(int) * 10 - (n["MarketRegime"] == "🔴 BEARISH").astype(int) * 10
    return n


def get_market_regime(nifty_ind):
    if nifty_ind.empty:
        return "🟡 CHOPPY"
    return nifty_ind["MarketRegime"].iloc[-1]


def build_frame(raw, nifty_ind, mode="AUTO"):
    df = calculate_indicators(raw)
    if df.empty:
        return pd.DataFrame()

    daily_turnover = (df["Close"] * df["Volume"]).groupby(df["Date"]).sum()
    liquid_days = daily_turnover.shift(1).rolling(20, min_periods=5).median() >= MIN_DAILY_TURNOVER
    df["Liquid"] = df["Date"].isin(set(liquid_days[liquid_days].index))
    if not df["Liquid"].any():
        return pd.DataFrame()

    nf = nifty_ind[["Return5", "MarketRegime"]].reindex(df.index, method="ffill")
    df["RS"] = df["Return5"] - nf["Return5"]
    df["Regime"] = nf["MarketRegime"]

    # Determine active signal based on mode and regime
    if mode == "🟢 BULLISH (Longs Only)":
        df["Signal"] = df["LongSignal"]
        df["Direction"] = "LONG"
    elif mode == "🔴 BEARISH (Shorts Only)":
        df["Signal"] = df["ShortSignal"]
        df["Direction"] = "SHORT"
    elif mode == "🟡 CHOPPY (Sit Out)":
        df["Signal"] = False
        df["Direction"] = "NONE"
    else:  # AUTO
        # If market regime is bullish, look for longs; if bearish, look for shorts; if choppy, no signal
        df["Direction"] = np.where(df["Regime"] == "🟢 BULLISH", "LONG",
                           np.where(df["Regime"] == "🔴 BEARISH", "SHORT", "NONE"))
        df["Signal"] = np.where(df["Direction"] == "LONG", df["LongSignal"],
                       np.where(df["Direction"] == "SHORT", df["ShortSignal"], False))

    df["Score"] = 85  # Standard score for valid pullback setups matching direction
    need = ["EMA20", "VWAP", "RSI", "ATR", "RVOL", "Return5", "RS"]
    df["Valid"] = df[need].notna().all(axis=1) & (df["BarNo"] >= SKIP_OPEN_BARS) & df["Liquid"] & (df["BarTime"] <= LAST_SIGNAL_TIME)
    df["FinalSignal"] = df["Valid"] & df["Signal"]
    return df


def trade_levels(direction, df, i, entry, atr):
    if direction == "LONG":
        swing_low = float(df["Low"].iloc[max(0, i - 2):i + 1].min())
        stop = min(swing_low - 0.1 * atr, entry - 1.0 * atr)
        stop = min(stop, entry * 0.995)
        risk = entry - stop
        target = entry + TARGET_R * risk
    else:  # SHORT
        swing_high = float(df["High"].iloc[max(0, i - 2):i + 1].max())
        stop = max(swing_high + 0.1 * atr, entry + 1.0 * atr)
        stop = max(stop, entry * 1.005)
        risk = stop - entry
        target = entry - TARGET_R * risk
    return float(stop), float(target), float(risk)

# ============================================================
# LIVE SCAN
# ============================================================

def run_live_scan(stock_data, nifty_raw, mode):
    nifty_ind = prepare_nifty(nifty_raw)
    if nifty_ind.empty:
        raise RuntimeError("Not enough NIFTY data.")
    asof = nifty_ind.index[-1]
    current_regime = get_market_regime(nifty_ind)
    market_info = {"regime": current_regime, "asof": asof}

    rows, skipped = [], []
    for ticker, raw in stock_data.items():
        df = build_frame(raw, nifty_ind, mode)
        if df.empty:
            skipped.append(ticker)
            continue
        i = len(df) - 1
        last = df.iloc[i]
        fresh_bar = df.index[i] == asof
        if not fresh_bar or not last["Valid"]:
            skipped.append(ticker)
            continue

        direction = last["Direction"]
        is_signal = bool(last["FinalSignal"])

        if is_signal and direction == "LONG":
            status = "🟢 ENTER LONG PULLBACK"
        elif is_signal and direction == "SHORT":
            status = "🔴 ENTER SHORT PULLBACK"
        else:
            status = "⚪ NO SETUP / CHOP"

        close = float(last["Close"])
        trigger = close * (1 + ENTRY_BUFFER) if direction == "LONG" else close * (1 - ENTRY_BUFFER)
        stop, target, risk = trade_levels(direction, df, i, trigger, float(last["ATR"]))
        prev_days = df.loc[df["Date"] < last["Date"], "Close"]
        change = (close / float(prev_days.iloc[-1]) - 1) * 100 if len(prev_days) else np.nan

        rows.append({
            "Stock": ticker.replace(".NS", ""), "Direction": direction, "Status": status,
            "Price": round(close, 2), "Change % (day)": round(change, 2),
            "Trigger": round(trigger, 2), "Stop Loss": round(stop, 2),
            "Target": round(target, 2), "Risk/Share": round(risk, 2), "R:R": TARGET_R,
            "RSI": round(float(last["RSI"]), 1), "RVOL": round(float(last["RVOL"]), 2),
            "RS vs NIFTY": round(float(last["RS"]), 2),
            "Market Regime": last["Regime"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["RVOL", "RS vs NIFTY"], ascending=False).reset_index(drop=True)
    return out, market_info, sorted(skipped)

# ============================================================
# PORTFOLIO BACKTEST (Bi-Directional Simulation)
# ============================================================

def collect_signals(ticker, df):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    atr, dates = df["ATR"].values, df["Date"].values
    sig = df["FinalSignal"].values
    direction_arr = df["Direction"].values
    n, out = len(df), []

    for i in range(30, n - 1):
        if not sig[i]:
            continue
        j = i + 1
        if dates[j] != dates[i]:
            continue
        direction = direction_arr[i]
        
        if direction == "LONG":
            trigger = c[i] * (1 + ENTRY_BUFFER)
            if o[j] >= trigger: fill = o[j]
            elif h[j] >= trigger: fill = trigger
            else: continue
            fill *= 1 + SLIPPAGE_PCT / 100
        elif direction == "SHORT":
            trigger = c[i] * (1 - ENTRY_BUFFER)
            if o[j] <= trigger: fill = o[j]
            elif l[j] <= trigger: fill = trigger
            else: continue
            fill *= 1 - SLIPPAGE_PCT / 100
        else:
            continue

        stop, target, risk = trade_levels(direction, df, i, fill, atr[i])
        k_end = j
        while k_end < n - 1 and k_end - j + 1 < HOLDING_BARS and dates[k_end + 1] == dates[j]:
            k_end += 1

        out.append({"ticker": ticker.replace(".NS", ""), "direction": direction,
                    "signal_time": df.index[i], "entry_time": df.index[j],
                    "entry_idx": j, "k_end": k_end, "fill": fill, "stop": stop,
                    "target": target, "risk": risk, "df": df, "raw_i": i})
    return out


def simulate_trade(sig):
    df, j, k_end = sig["df"], sig["entry_idx"], sig["k_end"]
    stop, target, fill, risk = sig["stop"], sig["target"], sig["fill"], sig["risk"]
    direction = sig["direction"]
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])

    exit_price = reason = None
    exit_k = k_end

    for k in range(j, k_end + 1):
        if direction == "LONG":
            if k > j and o[k] <= stop: exit_price, reason, exit_k = o[k], "GAP_STOP", k; break
            if h[k] >= target: exit_price, reason, exit_k = target, "TARGET", k; break
            if l[k] <= stop: exit_price, exit_k = stop, k; reason = "STOP"; break
        else: # SHORT
            if k > j and o[k] >= stop: exit_price, reason, exit_k = o[k], "GAP_STOP", k; break
            if l[k] <= target: exit_price, reason, exit_k = target, "TARGET", k; break
            if h[k] >= stop: exit_price, exit_k = stop, k; reason = "STOP"; break

    if exit_price is None:
        exit_price = c[k_end]
        reason = "TIME_EXIT" if k_end - j + 1 >= HOLDING_BARS else "EOD_EXIT"
    
    if direction == "LONG":
        if reason != "TARGET": exit_price *= 1 - SLIPPAGE_PCT / 100
        pnl = exit_price - fill - fill * COST_ROUND_TRIP_PCT / 100
    else:
        if reason != "TARGET": exit_price *= 1 + SLIPPAGE_PCT / 100
        pnl = fill - exit_price - fill * COST_ROUND_TRIP_PCT / 100

    return exit_price, reason, exit_k, pnl, (pnl / risk if risk > 0 else 0.0)


def run_full_backtest(tickers, days, mode):
    stock_data, dl_errors = download_market_data(tickers, days)
    nifty_ind = prepare_nifty(download_nifty(days))
    errors, skipped, signals = list(dl_errors), [], []

    for ticker, raw in stock_data.items():
        try:
            df = build_frame(raw, nifty_ind, mode)
            if df.empty:
                skipped.append(ticker)
                continue
            signals.extend(collect_signals(ticker, df))
        except Exception:
            errors.append(ticker)

    signals.sort(key=lambda s: s["entry_time"])

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
            "Stock": tk, "Direction": sig["direction"], "Signal Time": sig["signal_time"],
            "Entry Time": et, "Exit Time": exit_time, "Entry": round(sig["fill"], 2),
            "Stop": round(sig["stop"], 2), "Target": round(sig["target"], 2), "Exit": round(exit_price, 2),
            "Risk %": round(sig["risk"] / sig["fill"] * 100, 2), "R (net)": round(r, 3),
            "P&L/Share (net)": round(pnl, 2),
            "Outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "BREAKEVEN",
            "Exit Reason": reason, "Bars Held": exit_k - j + 1,
            "Market Regime": row["Regime"],
        })

    out = pd.DataFrame(trades)
    if not out.empty:
        out = out.sort_values("Signal Time").reset_index(drop=True)
    return out, sorted(set(errors)), sorted(set(skipped))

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


def daily_r(trades):
    return trades.groupby(trades["Entry Time"].dt.date)["R (net)"].sum()


def calculate_backtest_stats(trades):
    if trades.empty: return {}
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
        if s: cols[name] = {k: _fmt(v) for k, v in s.items()}
    return pd.DataFrame(cols)


st.markdown("# ⚡ Intraday Pulse")
st.caption("Bi-Directional & 3-Way Market Regime Edition. Research tool only.")

# Market Mode Filter Selector
market_mode = st.selectbox(
    "🌐 Market Filter Mode (Select Regime Strategy)",
    [
        "🌐 AUTO (Follow Nifty Regime: Long in Bull, Short in Bear, Sit out in Chop)",
        "🟢 BULLISH (Longs Only)",
        "🔴 BEARISH (Shorts Only)",
        "🟡 CHOPPY (Sit Out / No Trades)"
    ]
)

scan_tab, backtest_tab = st.tabs(["🚀 Live Scanner", "📈 Backtest"])

with scan_tab:
    if st.button("🚀 Run Bi-Directional Scan", type="primary", key="scan", **STRETCH):
        try:
            with st.spinner(f"Scanning {len(MASTER_WATCHLIST)} stocks under mode: {market_mode[:15]}..."):
                stock_data, dl_errors = download_market_data(tuple(MASTER_WATCHLIST), DATA_DAYS)
                nifty = download_nifty(DATA_DAYS)
                results, market_info, skipped = run_live_scan(stock_data, nifty, market_mode)
            st.session_state.update(live_results=results, market_info=market_info,
                                    live_errors=dl_errors, live_skipped=skipped,
                                    scan_time=datetime.now())
        except Exception as e:
            st.error(f"Scan failed: {e}")

    if "live_results" not in st.session_state:
        st.info("Select your market mode above and tap **Run Bi-Directional Scan**.")
    else:
        results, mi = st.session_state["live_results"], st.session_state["market_info"]
        st.markdown(f"### Detected Nifty Regime: {mi['regime']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Watchlist", len(MASTER_WATCHLIST))
        c2.metric("Analysed", len(results))
        c3.metric("Long Setups", int((results["Direction"] == "LONG").sum()) if len(results) else 0)
        c4.metric("Short Setups", int((results["Direction"] == "SHORT").sum()) if len(results) else 0)

        if results.empty:
            st.warning("No setups match the current bi-directional filter.")
        else:
            st.dataframe(results, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Results CSV", results.to_csv(index=False).encode("utf-8"),
                               "bidirectional_scan.csv", "text/csv", **STRETCH)

with backtest_tab:
    st.subheader("📈 Bi-Directional Portfolio Backtest")
    st.warning(
        f"Net of costs ({COST_ROUND_TRIP_PCT}% round trip + {SLIPPAGE_PCT}% slippage per side). "
        f"Active mode: {market_mode}. Long & Short pullback engines enabled.")

    if st.button("📊 Run Bi-Directional Backtest", type="primary", key="bt", **STRETCH):
        try:
            with st.spinner("Running bi-directional portfolio simulation..."):
                trades, errors, skipped = run_full_backtest(tuple(MASTER_WATCHLIST), DATA_DAYS, market_mode)
            st.session_state.update(backtest_trades=trades, bt_errors=errors, bt_skipped=skipped)
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if "backtest_trades" not in st.session_state:
        st.info("Tap **Run Bi-Directional Backtest** to measure performance.")
    else:
        trades = st.session_state["backtest_trades"]
        if trades.empty:
            st.error("No historical trades matched the rules under this mode.")
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

            st.subheader("In-sample vs out-of-sample")
            st.dataframe(split_stats(trades), **STRETCH)

            st.subheader("Cumulative return % (non-compounded)")
            eq = trades.sort_values("Exit Time")["R (net)"].cumsum() * RISK_PER_TRADE_PCT
            st.line_chart(pd.DataFrame({"Cumulative return %": eq.values}))

            st.subheader("📒 Trade Log")
            st.dataframe(trades, hide_index=True, **STRETCH)
            st.download_button("⬇️ Download Trade Log CSV", trades.to_csv(index=False).encode("utf-8"),
                               "bidirectional_backtest_log.csv", "text/csv", **STRETCH)
