import os, warnings, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t

warnings.filterwarnings("ignore")
DATA_DIR = "data"
PATHS = { k: os.path.join(DATA_DIR, f"{k}_daily.csv") for k in ['nifty','vix','bank_nifty','sp500'] }
PATHS.update({ 'vterm': os.path.join(DATA_DIR, "vix_term_daily.csv"), 'fii': os.path.join(DATA_DIR, "fii_dii_daily.csv"), 'pcr': os.path.join(DATA_DIR, "pcr_daily.csv"), 'n15m': os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv"), 'v15m': os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv") })

def _load(key):
    p = PATHS.get(key)
    if not p or not os.path.exists(p): return pd.DataFrame()
    df = pd.read_csv(p); df.columns = [c.lower() for c in df.columns]
    dcol = 'datetime' if 'datetime' in df.columns else 'date'
    df = df.rename(columns={dcol: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz: df['date'] = df['date'].dt.tz_localize(None)
    return df.sort_values('date').reset_index(drop=True)

def build_features():
    df = _load('nifty')
    if df.empty: return df
    # Base features
    df['range'] = df['high'] - df['low']
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    df['atr10'] = df['range'].rolling(10).mean().shift(1)
    df['atr20'] = df['range'].rolling(20).mean().shift(1)
    d = df['close'].diff()
    df['rsi'] = (lambda g,l: 100 - (100/(1+g/l.replace(0,np.nan))))(d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()).shift(1)
    df['z20'] = ((df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()).shift(1)
    df['trend'] = np.where((df['close']>df['close'].rolling(20).mean()) & (df['close'].rolling(20).mean()>df['close'].rolling(50).mean()), 1, np.where((df['close']<df['close'].rolling(20).mean()) & (df['close'].rolling(20).mean()<df['close'].rolling(50).mean()), -1, 0))
    for h in [1, 7]: df[f'fwd_{h}d'] = df['close'].shift(-h)/df['close']-1; df[f'fhi_{h}d'] = df['high'].rolling(h).max().shift(-h)/df['close']-1; df[f'flo_{h}d'] = df['low'].rolling(h).min().shift(-h)/df['close']-1
    # VIX
    vx = _load('vix').rename(columns={'close':'vix'})
    if not vx.empty:
        vx['vix_pct'] = vx['vix'].rolling(252).rank(pct=True).shift(1)
        vx['vix_spread'] = (_load('vterm')['vix_near'] - _load('vterm')['vix_far']).shift(1) if not _load('vterm').empty else 0
        df = df.merge(vx[['date','vix','vix_pct','vix_spread']], on='date', how='left')
    # Others
    sp = _load('sp500').rename(columns={'close':'sp'})
    if not sp.empty: df = df.merge(sp[['date','sp']], on='date', how='left'); df['sp_ret1'] = df['sp'].pct_change(1)
    bn = _load('bank_nifty').rename(columns={'close':'bn'})
    if not bn.empty: df = df.merge(bn[['date','bn']], on='date', how='left'); df['bn_vs_n'] = df['bn'].pct_change(1).shift(1) - df['close'].pct_change(1).shift(1)
    fii = _load('fii'); pcr = _load('pcr')
    if not fii.empty: df = df.merge(fii[['date','fii_net']], on='date', how='left'); df['fii_z'] = ((df['fii_net']-df['fii_net'].rolling(20).mean())/df['fii_net'].rolling(20).std()).shift(1)
    if not pcr.empty: df = df.merge(pcr[['date','pcr']], on='date', how='left'); df['pcr_z'] = ((df['pcr']-df['pcr'].rolling(20).mean())/df['pcr'].rolling(20).std()).shift(1)
    return df

def empirical_probs(df, h, row):
    hist = df.dropna(subset=[f'fwd_{h}d']).copy()
    cvp, crsi = row.get('vix_pct', 0.5), row.get('rsi', 50)
    sub = hist[(hist['vix_pct'] >= cvp-0.2) & (hist['vix_pct'] <= cvp+0.2) & (hist['rsi'] >= crsi-15) & (hist['rsi'] <= crsi+15)]
    if len(sub) < 30: sub = hist
    fwd, hi, lo = sub[f'fwd_{h}d'].values, sub[f'fhi_{h}d'].values, sub[f'flo_{h}d'].values
    return {'p_up': (fwd>0.01).mean(), 'p_flat': (np.abs(fwd)<=0.01).mean(), 'p_down': (fwd<-0.01).mean(), 'pct': np.percentile(fwd, [5, 10, 25, 50, 75, 90, 95])}

def monte_carlo(df, spot, h, n=50000):
    lr = df['logret'].dropna().values
    dft, mu_t, sig_t = stats.t.fit(lr)
    rets = student_t.rvs(df=dft, loc=mu_t, scale=sig_t, size=(n, h))
    paths = spot * np.exp(np.cumsum(rets, axis=1))
    finals = paths[:, -1]
    return {'p_up': (finals>spot*1.01).mean(), 'p_flat': (np.abs(finals/spot-1)<=0.01).mean(), 'p_down': (finals<spot*0.99).mean(), 'paths_sample': paths[:100]}

def bayesian_signals(row, df):
    hist = df.dropna(subset=['fwd_7d']).copy()
    if len(hist) < 50: return None
    pu, pd_ = (hist['fwd_7d']>0.01).mean(), (hist['fwd_7d']<-0.01).mean()
    lu, ld = np.log(max(pu,0.01)/(1-max(pu,0.01))), np.log(max(pd_,0.01)/(1-max(pd_,0.01)))
    adj = 0
    if row.get('vix_pct',0.5) > 0.7: adj -= 0.4
    if row.get('rsi',50) < 35: adj += 0.5
    if row.get('sp_ret1',0) < -0.015: adj -= 0.3
    res_u = 1/(1+np.exp(-(lu+adj))); res_d = 1/(1+np.exp(-(ld-adj)))
    s = res_u + res_d + 0.1; return {'p_up': res_u/s, 'p_flat': 0.1/s, 'p_down': res_d/s}

def compute_verdict(horizon=7):
    df = build_features()
    if df.empty: return None
    row = df.iloc[-1]; spot = float(row['close'])
    emp, mc, bay = empirical_probs(df, horizon, row), monte_carlo(df, spot, horizon), bayesian_signals(row, df)
    eu, ef, ed = (emp['p_up']*0.4 + mc['p_up']*0.35 + bay['p_up']*0.25), (emp['p_flat']*0.4 + mc['p_flat']*0.35 + bay['p_flat']*0.25), (emp['p_down']*0.4 + mc['p_down']*0.35 + bay['p_down']*0.25)
    mx = max(eu, ef, ed); v = 'UP' if mx==eu else 'DOWN' if mx==ed else 'SIDEWAYS'
    return { 'spot': spot, 'horizon': horizon, 'eu': eu, 'ef': ef, 'ed': ed, 'verdict': v, 'confidence': (mx-1/3)/(2/3)*100, 'expected_range': emp['pct'], 'mc': mc, 'empirical': emp, 'bayesian': bay }

if __name__ == "__main__":
    res = compute_verdict(7)
    if res:
        print(f"VERDICT: {res['verdict']} ({res['confidence']:.0f}% confidence)")
        print(f"UP: {res['eu']:.1%}, FLAT: {res['ef']:.1%}, DOWN: {res['ed']:.1%}")
