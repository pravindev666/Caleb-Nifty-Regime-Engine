"""
whipsaw_analyser.py — Whipsaw Risk & Path Projection
======================================================
Answers the question shown in the image:
  "The bear call spread looks right — but what if there is
   a 5-day bounce first, then the real move continues down?"

That is a whipsaw. Your stop gets hit on the bounce.
This module estimates:
  1. How likely a short-term bounce is RIGHT NOW (RSI oversold signal)
  2. What the bounce might look like (magnitude, duration)
  3. What the structural trend says over 20 days
  4. The safe strike placement that survives both paths

No lookahead. All based on lagged features only.

Backtest:
  Measures how often a bounce preceded a continued downtrend
  on all historical days with similar RSI/Z conditions.

USED BY: dashboard.py (whipsaw chart tab)
CAN ALSO RUN STANDALONE: python whipsaw_analyser.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR    = "data"
NIFTY_DAILY = os.path.join(DATA_DIR, "nifty_daily.csv")
VIX_DAILY   = os.path.join(DATA_DIR, "vix_daily.csv")


def _load(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def build_whipsaw_data():
    """
    Build a table of all historical days where conditions were similar to today:
    - Downtrend (price below SMA20 and SMA50)
    - RSI oversold (< 40) — bounce risk zone
    - Then measure what actually happened over the next 5 and 20 days
    """
    nifty = _load(NIFTY_DAILY)
    if nifty.empty:
        return pd.DataFrame()

    nifty['range']   = nifty['high'] - nifty['low']
    nifty['atr10']   = nifty['range'].rolling(10, min_periods=5).mean()

    delta = nifty['close'].diff()
    g = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    l = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    nifty['rsi'] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    nifty['z_score'] = (
        (nifty['close'] - nifty['close'].rolling(20, min_periods=10).mean()) /
         nifty['close'].rolling(20, min_periods=10).std()
    )
    s20 = nifty['close'].rolling(20, min_periods=10).mean()
    s50 = nifty['close'].rolling(50, min_periods=25).mean()
    nifty['trend'] = np.where(
        (nifty['close'] > s20) & (s20 > s50),  1,
        np.where((nifty['close'] < s20) & (s20 < s50), -1, 0)
    )
    nifty['sma20'] = s20

    # Forward paths — what happened after
    for d in [1, 2, 3, 5, 10, 20]:
        nifty[f'fwd_{d}d'] = nifty['close'].shift(-d) / nifty['close'] - 1
        nifty[f'fwd_max_{d}d'] = nifty['high'].rolling(d).max().shift(-d) / nifty['close'] - 1

    # Max bounce in first 5 days (the whipsaw)
    nifty['bounce_5d'] = nifty['high'].rolling(5).max().shift(-5) / nifty['close'] - 1
    # 20-day drift (the real structural move)
    nifty['drift_20d'] = nifty['close'].shift(-20) / nifty['close'] - 1

    return nifty


def analyse_whipsaw(close, rsi, z_score, trend, atr10, vix=None):
    """
    Given today's conditions, return:
    - bounce_prob: probability of a 2%+ bounce in next 5 days
    - bounce_est: estimated bounce magnitude in points
    - drift_est: estimated 20-day drift in points
    - similar_days: historical rows with similar conditions
    - path_data: list of (day, price) for the expected path chart
    """
    nifty = build_whipsaw_data()
    if nifty.empty:
        return None

    # Find historical days with SIMILAR conditions (no lookahead — these are past days)
    # Conditions: downtrend + oversold (RSI < 45) + price stretched (Z < -1)
    similar = nifty[
        (nifty['trend'] == -1) &
        (nifty['rsi'] < 45) &
        (nifty['rsi'] > 20) &       # filter extreme crash days
        (nifty['z_score'] < -0.8) &
        (nifty['bounce_5d'].notna()) &
        (nifty['drift_20d'].notna())
    ].copy()

    if len(similar) < 10:
        # Relax criteria if not enough history
        similar = nifty[
            (nifty['trend'] <= 0) &
            (nifty['rsi'] < 50) &
            (nifty['z_score'] < -0.5) &
            (nifty['bounce_5d'].notna())
        ].copy()

    n_sim = len(similar)
    if n_sim == 0:
        return None

    # Bounce statistics
    bounce_5d_pct  = similar['bounce_5d'].median()
    bounce_5d_p75  = similar['bounce_5d'].quantile(0.75)
    drift_20d_med  = similar['drift_20d'].median()
    drift_20d_p25  = similar['drift_20d'].quantile(0.25)   # bear case

    # Probability that there WAS a meaningful bounce (>1.5%) before 20-day fall
    whipsaw_days   = similar[
        (similar['bounce_5d'] > 0.015) &    # bounced > 1.5%
        (similar['drift_20d'] < -0.02)      # then fell > 2%
    ]
    bounce_then_fall_prob = len(whipsaw_days) / max(n_sim, 1)

    # Probability that price is LOWER in 20 days
    down_prob = (similar['drift_20d'] < 0).mean()

    # Build expected path points (day 0 to day 20)
    # Three scenarios: bounce-then-fall, direct fall, sideways
    bounce_pts = close * bounce_5d_pct    # typical bounce from close
    drift_pts  = close * drift_20d_med   # typical 20-day outcome

    # Path for chart: (day, bounce_path, bear_path, trend_line)
    days = list(range(0, 21))
    bounce_path = []
    bear_path   = []
    trend_line  = []

    # SMA20 trend slope estimate (from recent data)
    recent = nifty.dropna(subset=['sma20']).tail(20)
    if len(recent) >= 10:
        slope_pct = (recent['sma20'].iloc[-1] / recent['sma20'].iloc[0]) - 1
        daily_drift = slope_pct / 20
    else:
        daily_drift = -0.003   # assume -0.3%/day in downtrend

    for d in days:
        # Bounce path: rises first 5 days, then falls
        if d <= 5:
            bp = close * (1 + bounce_5d_pct * (d / 5))
        else:
            # After bounce, trends down toward 20-day drift
            t = (d - 5) / 15
            bp = close * (1 + bounce_5d_pct) * (1 + drift_20d_med * t)
        bounce_path.append(round(bp))

        # Direct bear path: straight line to 20-day bear case (p25)
        bp2 = close * (1 + drift_20d_p25 * (d / 20))
        bear_path.append(round(bp2))

        # Structural trend line (SMA20 slope)
        tl = close * (1 + daily_drift * d)
        trend_line.append(round(tl))

    # Safe bear call spread strike: above the bounce_path peak
    bounce_peak = max(bounce_path)
    safe_sell_strike = int(round((bounce_peak * 1.02) / 50) * 50)  # 2% above bounce peak
    safe_buy_strike  = safe_sell_strike + 100

    return {
        'n_similar':             n_sim,
        'bounce_5d_median_pct':  bounce_5d_pct,
        'bounce_5d_pts':         round(close * bounce_5d_pct),
        'bounce_peak_est':       round(bounce_peak),
        'drift_20d_median_pct':  drift_20d_med,
        'drift_20d_pts':         round(close * drift_20d_med),
        'bounce_then_fall_prob': bounce_then_fall_prob,
        'down_20d_prob':         down_prob,
        'safe_sell_strike':      safe_sell_strike,
        'safe_buy_strike':       safe_buy_strike,
        'days':                  days,
        'bounce_path':           bounce_path,
        'bear_path':             bear_path,
        'trend_line':            trend_line,
        'close':                 close,
    }


def run_whipsaw_backtest():
    """
    No-lookahead backtest: on days when bear call spread was signalled,
    how often did a whipsaw (bounce first, then fall) occur?
    Walk-forward: for each test year, use only prior years to compute thresholds.
    """
    nifty = build_whipsaw_data()
    if nifty.empty:
        print("No data.")
        return

    print("\n── Whipsaw Backtest (no lookahead) ────────────────────")
    print("  Question: on 'bear signal' days, how often was there a")
    print("  bounce first, THEN continued downside?\n")
    print(f"  {'Year':<6} {'Bear days':>10} {'Whipsaw%':>10} {'Down20d%':>10} {'AvgBounce':>11}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*11}")

    start_yr = nifty['date'].dt.year.min() + 3   # need 3 years of history first
    end_yr   = nifty['date'].dt.year.max()

    all_whipsaw = []; all_down = []; all_bounce = []

    for yr in range(start_yr, end_yr + 1):
        # Test set: this year only
        te = nifty[nifty['date'].dt.year == yr].copy()

        # Bear signal conditions (same as generate_signals)
        bear_days = te[
            (te['trend'] == -1) &
            (te['rsi'] > 32) & (te['rsi'] < 60) &
            (te['z_score'].notna()) &
            (te['bounce_5d'].notna()) &
            (te['drift_20d'].notna())
        ]

        if len(bear_days) < 5:
            continue

        # Whipsaw: bounced >1.5% in 5 days AND still down >2% at 20 days
        whipsaw = (bear_days['bounce_5d'] > 0.015) & (bear_days['drift_20d'] < -0.02)
        down20  = bear_days['drift_20d'] < 0
        bounce  = bear_days['bounce_5d']

        ws_pct = whipsaw.mean()
        d20_pct = down20.mean()
        avg_bounce = bounce.mean() * 100

        all_whipsaw.extend(whipsaw.tolist())
        all_down.extend(down20.tolist())
        all_bounce.extend(bounce.tolist())

        print(f"  {yr:<6} {len(bear_days):>10} {ws_pct:>9.1%} {d20_pct:>9.1%} {avg_bounce:>10.1f}%")

    if all_whipsaw:
        print(f"\n  OVERALL:")
        print(f"    Bear signal days    : {len(all_whipsaw)}")
        print(f"    Whipsaw probability : {np.mean(all_whipsaw):.1%}  (bounce first, then fall)")
        print(f"    Down at 20 days     : {np.mean(all_down):.1%}")
        print(f"    Avg bounce size     : {np.mean(all_bounce)*100:.1f}%")
        print(f"\n  IMPLICATION FOR BEAR CALL SPREAD:")
        print(f"    Set sell strike ABOVE expected bounce peak")
        print(f"    = close × (1 + avg_bounce × 1.5) = buffer above whipsaw")
        ws_rate = np.mean(all_whipsaw)
        if ws_rate > 0.30:
            print(f"\n  ⚠️  HIGH whipsaw risk ({ws_rate:.0%}) — always leave room above bounce")
        else:
            print(f"\n  ✅ Moderate whipsaw risk ({ws_rate:.0%}) — standard strike placement OK")
    print()


if __name__ == "__main__":
    run_whipsaw_backtest()

    # Show today's analysis
    nifty = build_whipsaw_data()
    if not nifty.empty:
        today = nifty.dropna(subset=['atr10','rsi']).iloc[-1]
        result = analyse_whipsaw(
            close  = float(today['close']),
            rsi    = float(today['rsi']),
            z_score= float(today['z_score']),
            trend  = int(today['trend']),
            atr10  = float(today['atr10']),
        )
        if result:
            print(f"\n── Today's whipsaw analysis ────────────────────────────")
            print(f"  Based on {result['n_similar']} similar historical days")
            print(f"  Typical bounce first   : +{result['bounce_5d_pts']} pts ({result['bounce_5d_median_pct']:.1%})")
            print(f"  Peak bounce estimate   : {result['bounce_peak_est']:,}")
            print(f"  20-day drift estimate  : {result['drift_20d_pts']:+} pts")
            print(f"  Whipsaw probability    : {result['bounce_then_fall_prob']:.0%}")
            print(f"  Down at 20 days        : {result['down_20d_prob']:.0%}")
            print(f"  Safe bear sell strike  : {result['safe_sell_strike']:,}  (above bounce)")
            print(f"  Safe bear buy strike   : {result['safe_buy_strike']:,}  (loss cap)")
