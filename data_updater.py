"""
data_updater.py — Fetches and maintains ALL data sources
=========================================================
Sources managed:
  yfinance (auto):  nifty_15m, vix_15m, nifty_daily, bank_nifty_daily,
                    sp500_daily, vix_daily
  NSE scrape:       fii_dii_daily, pcr_daily
  Derived:          vix_term_daily (computed from vix_daily)

Run:  python data_updater.py
"""

import os, warnings
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")

DATA_DIR    = "data"
NIFTY_15M   = os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv")
VIX_15M     = os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv")
NIFTY_DAILY = os.path.join(DATA_DIR, "nifty_daily.csv")
BANKNIFTY   = os.path.join(DATA_DIR, "bank_nifty_daily.csv")
SP500       = os.path.join(DATA_DIR, "sp500_daily.csv")
VIX_DAILY   = os.path.join(DATA_DIR, "vix_daily.csv")
FII_DII     = os.path.join(DATA_DIR, "fii_dii_daily.csv")
PCR         = os.path.join(DATA_DIR, "pcr_daily.csv")
VIX_TERM    = os.path.join(DATA_DIR, "vix_term_daily.csv")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def _save(df, path):
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    df.to_csv(path, index=False)


def _fetch_yf_daily(ticker, period="max"):
    raw = yf.download(ticker, period=period, interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    dc = 'datetime' if 'datetime' in raw.columns else 'date'
    raw = raw.rename(columns={dc: 'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    if raw['date'].dt.tz is not None:
        raw['date'] = raw['date'].dt.tz_localize(None)
    return raw[['date', 'open', 'high', 'low', 'close', 'volume']].copy()


def _fetch_yf_15m(ticker, period="60d"):
    raw = yf.download(ticker, period=period, interval="15m",
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    dc = 'datetime' if 'datetime' in raw.columns else 'date'
    raw = raw.rename(columns={dc: 'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    if raw['date'].dt.tz is not None:
        raw['date'] = raw['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    return raw[['date', 'open', 'high', 'low', 'close', 'volume']].copy()


def _append_new(existing, fresh):
    if existing.empty:
        return fresh
    cutoff = existing['date'].max()
    new    = fresh[fresh['date'] > cutoff]
    if new.empty:
        return existing
    return pd.concat([existing, new], ignore_index=True)


# ── YFINANCE SOURCES ──────────────────────────────────────────────────────────

def update_15m(path, ticker):
    print(f"  {ticker} 15m ... ", end="", flush=True)
    existing = _load(path)
    fresh    = _fetch_yf_15m(ticker)
    if fresh.empty:
        print("no data")
        return
    combined = _append_new(existing, fresh)
    _save(combined, path)
    added = len(combined) - len(existing)
    print(f"+{added} rows → {len(combined)} total")


def update_daily_yf(path, ticker):
    print(f"  {ticker} daily ... ", end="", flush=True)
    existing = _load(path)
    fresh    = _fetch_yf_daily(ticker)
    if fresh.empty:
        print("no data")
        return
    combined = _append_new(existing, fresh)
    _save(combined, path)
    added = len(combined) - len(existing)
    print(f"+{added} rows → {len(combined)} total")


def rebuild_nifty_daily():
    """Aggregate nifty_15m to daily OHLC and merge VIX close."""
    print(f"  Rebuilding nifty_daily ... ", end="", flush=True)
    if not os.path.exists(NIFTY_15M):
        print("nifty_15m missing")
        return
    m = pd.read_csv(NIFTY_15M)
    m.columns = [c.lower() for c in m.columns]
    m['date'] = pd.to_datetime(m['date'])
    m['d']    = m['date'].dt.date
    daily = m.groupby('d').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low',  'min'),    close=('close', 'last')
    ).reset_index().rename(columns={'d': 'date'})
    daily['date'] = pd.to_datetime(daily['date'])

    if os.path.exists(VIX_DAILY):
        vd = pd.read_csv(VIX_DAILY)
        vd.columns = [c.lower() for c in vd.columns]
        vd['date'] = pd.to_datetime(vd['date'])
        vd = vd.rename(columns={'close': 'vix'})[['date', 'vix']]
        daily = daily.merge(vd, on='date', how='left')

    _save(daily, NIFTY_DAILY)
    print(f"{len(daily)} rows")


# ── FII/DII — NSE SCRAPE ──────────────────────────────────────────────────────

def update_fii_dii():
    """
    Scrape FII/DII net flows from NSE India API.
    Columns: date, fii_net, dii_net  (crores INR)
    """
    print(f"  FII/DII (NSE) ... ", end="", flush=True)
    existing = _load(FII_DII)
    headers  = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nseindia.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    rows = []
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact",
                  headers=headers, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                try:
                    dt      = pd.to_datetime(item.get('date', ''), dayfirst=True)
                    fii_net = float(str(item.get('fiiNet', 0)).replace(',', '') or 0)
                    dii_net = float(str(item.get('diiNet', 0)).replace(',', '') or 0)
                    if fii_net != 0:
                        rows.append({'date': dt, 'fii_net': fii_net, 'dii_net': dii_net})
                except Exception:
                    continue
    except Exception as e:
        print(f"NSE unreachable ({type(e).__name__}). Keeping existing.")
        return

    if not rows:
        print("no data returned")
        return

    fresh    = pd.DataFrame(rows).dropna(subset=['date'])
    combined = _append_new(existing, fresh)
    _save(combined, FII_DII)
    added = len(combined) - len(existing)
    print(f"+{added} rows → {len(combined)} total")


# ── PCR — NSE OPTION CHAIN SCRAPE ────────────────────────────────────────────

def update_pcr():
    """
    Compute today's Put/Call Ratio from NSE NIFTY option chain.
    PCR = total put OI / total call OI across all strikes.
    Columns: date, pcr
    """
    print(f"  PCR (NSE option chain) ... ", end="", flush=True)
    existing = _load(PCR)
    headers  = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nseindia.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers, timeout=10)
        r = s.get(
            "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            headers=headers, timeout=10
        )
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return
        data = r.json()
        put_oi = call_oi = 0
        for item in data.get('records', {}).get('data', []):
            put_oi  += item.get('PE', {}).get('openInterest', 0)
            call_oi += item.get('CE', {}).get('openInterest', 0)
        if call_oi == 0:
            print("zero call OI")
            return
        pcr   = round(put_oi / call_oi, 4)
        today = pd.Timestamp('today').normalize()
        fresh = pd.DataFrame([{'date': today, 'pcr': pcr}])
        combined = _append_new(existing, fresh)
        _save(combined, PCR)
        added = len(combined) - len(existing)
        print(f"PCR={pcr:.3f}  +{added} rows → {len(combined)} total")
    except Exception as e:
        print(f"failed ({type(e).__name__}). Keeping existing.")


# ── VIX TERM STRUCTURE — DERIVED ─────────────────────────────────────────────

def update_vix_term():
    """
    Compute VIX term structure entirely from vix_daily.csv.
    vix_near = 5-day EMA  (short-term fear)
    vix_far  = 21-day EMA (medium-term fear)
    When vix_near > vix_far (inverted) → expect big range day soon.
    No external fetch needed.
    """
    print(f"  VIX term structure (derived) ... ", end="", flush=True)
    if not os.path.exists(VIX_DAILY):
        print("vix_daily missing")
        return
    vd = pd.read_csv(VIX_DAILY)
    vd.columns = [c.lower() for c in vd.columns]
    vd['date']     = pd.to_datetime(vd['date'])
    vd             = vd.sort_values('date').reset_index(drop=True)
    vd['vix_near'] = vd['close'].ewm(span=5,  adjust=False).mean().round(6)
    vd['vix_far']  = vd['close'].ewm(span=21, adjust=False).mean().round(6)
    term = vd[['date', 'vix_near', 'vix_far']].dropna()
    _save(term, VIX_TERM)
    print(f"{len(term)} rows")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_update():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n── Data update ─────────────────────────────────────────")

    # 15-minute intraday bars
    update_15m(NIFTY_15M, "^NSEI")
    update_15m(VIX_15M,   "^INDIAVIX")

    # Daily OHLC from yfinance
    update_daily_yf(VIX_DAILY, "^INDIAVIX")   # India VIX — NOT ^VIX (that is US CBOE VIX)
    update_daily_yf(BANKNIFTY, "^NSEBANK")
    update_daily_yf(SP500,     "^GSPC")

    # Rebuild nifty_daily from 15m + VIX
    rebuild_nifty_daily()

    # Derived: VIX term structure
    update_vix_term()

    # NSE live scrape
    update_fii_dii()
    update_pcr()

    print("── Done ────────────────────────────────────────────────\n")


if __name__ == "__main__":
    run_update()
