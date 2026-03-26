import os, warnings, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t

warnings.filterwarnings("ignore")
DATA_DIR = "data"
PATHS = {
    'nifty': os.path.join(DATA_DIR, "nifty_daily.csv"),
    'vix': os.path.join(DATA_DIR, "vix_daily.csv"),
    'bank_nifty': os.path.join(DATA_DIR, "bank_nifty_daily.csv"),
    'sp500': os.path.join(DATA_DIR, "sp500_daily.csv"),
    'vterm': os.path.join(DATA_DIR, "vix_term_daily.csv"),
    'fii': os.path.join(DATA_DIR, "fii_dii_daily.csv"),
    'pcr': os.path.join(DATA_DIR, "pcr_daily.csv"),
    'n15m': os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv"),
    'v15m': os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv")
}

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
    df['range'] = df['high'] - df['low']
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    df['atr10'] = df['range'].rolling(10).mean().shift(1)
    df['atr20'] = df['range'].rolling(20).mean().shift(1)
    d = df['close'].diff()
    df['rsi'] = (lambda g,l: 100 - (100/(1+g/l.replace(0,np.nan))))(d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()).shift(1)
    df['z20'] = ((df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()).shift(1)
    df['trend'] = np.where((df['close']>df['close'].rolling(20).mean()) & (df['close'].rolling(20).mean()>df['close'].rolling(50).mean()), 1, np.where((df['close']<df['close'].rolling(20).mean()) & (df['close'].rolling(20).mean()<df['close'].rolling(50).mean()), -1, 0))
    for h in [1, 7]: 
        df[f'fwd_{h}d'] = df['close'].shift(-h)/df['close']-1
        df[f'fhi_{h}d'] = df['high'].rolling(h).max().shift(-h)/df['close']-1
        df[f'flo_{h}d'] = df['low'].rolling(h).min().shift(-h)/df['close']-1
        
    vx = _load('vix').rename(columns={'close':'vix'})
    if not vx.empty:
        vx['vix_pct'] = vx['vix'].rolling(252).rank(pct=True).shift(1)
        vx['vix_spread'] = (_load('vterm')['vix_near'] - _load('vterm')['vix_far']).shift(1) if not _load('vterm').empty else 0
        df = pd.merge(df, vx[['date','vix_pct','vix_spread']], on='date', how='left')
        if 'vix' not in df.columns: df['vix'] = vx['vix']
    else: 
        if 'vix' not in df.columns: df['vix'] = 15
        df['vix_pct'] = 0.5; df['vix_spread'] = 0
        
    sp = _load('sp500').rename(columns={'close':'sp'})
    if not sp.empty: df = pd.merge(df, sp[['date','sp']], on='date', how='left'); df['sp_ret1'] = df['sp'].pct_change(1).shift(1)
    else: df['sp_ret1'] = 0
    
    bn = _load('bank_nifty').rename(columns={'close':'bn'})
    if not bn.empty: df = pd.merge(df, bn[['date','bn']], on='date', how='left'); df['bn_vs_n'] = df['bn'].pct_change(1).shift(1) - df['close'].pct_change(1).shift(1)
    else: df['bn_vs_n'] = 0
    
    fii = _load('fii'); pcr = _load('pcr')
    if not fii.empty: df = pd.merge(df, fii[['date','fii_net']], on='date', how='left'); df['fii_z'] = ((df['fii_net']-df['fii_net'].rolling(20).mean())/df['fii_net'].rolling(20).std()).shift(1)
    else: df['fii_net'] = 0; df['fii_z'] = 0
    if not pcr.empty: df = pd.merge(df, pcr[['date','pcr']], on='date', how='left'); df['pcr_z'] = ((df['pcr']-df['pcr'].rolling(20).mean())/df['pcr'].rolling(20).std()).shift(1)
    else: df['pcr'] = 0; df['pcr_z'] = 0
    
    n15m = _load('n15m'); v15m = _load('v15m')
    if not n15m.empty: df['n15_ret'] = df['close'].pct_change(1).fillna(0)
    else: df['n15_ret'] = 0
    if not v15m.empty: df['v15_ret'] = df['vix'] / df['vix'].shift(1) - 1
    else: df['v15_ret'] = 0

    return df

def get_verdict(u, f, d):
    mx = max(u, f, d)
    if mx == u: return 'UP'
    elif mx == d: return 'DOWN'
    else: return 'FLAT'

def empirical_probs(df, h, row, spot):
    hist = df.dropna(subset=[f'fwd_{h}d']).copy()
    cvp, crsi = row.get('vix_pct', 0.5), row.get('rsi', 50)
    sub = hist[(hist['vix_pct'] >= cvp-0.2) & (hist['vix_pct'] <= cvp+0.2) & (hist['rsi'] >= crsi-15) & (hist['rsi'] <= crsi+15)]
    if len(sub) < 30: sub = hist
    fwd, hi, lo = sub[f'fwd_{h}d'].values, sub[f'fhi_{h}d'].values, sub[f'flo_{h}d'].values
    
    u, f, d = (fwd>0.01).mean(), (np.abs(fwd)<=0.01).mean(), (fwd<-0.01).mean()
    s_sum = u+f+d
    if s_sum == 0: s_sum = 1
    u, f, d = u/s_sum, f/s_sum, d/s_sum
    
    upside = [
        {'label': f"+1% ({int(spot*1.01):,})", 'prob': float((hi >= 0.01).mean())},
        {'label': f"+2% ({int(spot*1.02):,})", 'prob': float((hi >= 0.02).mean())},
        {'label': f"+3% ({int(spot*1.03):,})", 'prob': float((hi >= 0.03).mean())},
        {'label': f"+5% ({int(spot*1.05):,})", 'prob': float((hi >= 0.05).mean())}
    ]
    downside = [
        {'label': f"-1% ({int(spot*0.99):,})", 'prob': float((lo <= -0.01).mean())},
        {'label': f"-2% ({int(spot*0.98):,})", 'prob': float((lo <= -0.02).mean())},
        {'label': f"-3% ({int(spot*0.97):,})", 'prob': float((lo <= -0.03).mean())},
        {'label': f"-5% ({int(spot*0.95):,})", 'prob': float((lo <= -0.05).mean())}
    ]
    
    pcts = np.percentile(fwd, [5, 25, 50, 75, 95])
    expected = { 'bull': int(spot * (1+pcts[4])), 'p75': int(spot * (1+pcts[3])), 'median': int(spot * (1+pcts[2])), 'p25': int(spot * (1+pcts[1])), 'bear': int(spot * (1+pcts[0])) }
    
    return { 'p_up': u, 'p_flat': f, 'p_down': d, 'n_samples': len(sub), 'verdict': get_verdict(u, f, d), 'upside': upside, 'downside': downside, 'expected': expected }

def monte_carlo(df, spot, h, n=50000):
    lr = df['logret'].dropna().values
    if len(lr) < 100: lr = np.random.normal(0, 0.01, 100)
    dft, mu_t, sig_t = stats.t.fit(lr)
    rets = student_t.rvs(df=dft, loc=mu_t, scale=sig_t, size=(n, h))
    paths = spot * np.exp(np.cumsum(rets, axis=1))
    finals = paths[:, -1]
    
    u, f, d = (finals>spot*1.01).mean(), (np.abs(finals/spot-1)<=0.01).mean(), (finals<spot*0.99).mean()
    s_sum = u+f+d
    if s_sum == 0: s_sum=1
    u, f, d = u/s_sum, f/s_sum, d/s_sum
    return { 'p_up': u, 'p_flat': f, 'p_down': d, 'paths_sample': paths[:10].tolist(), 'n_paths': n, 'verdict': get_verdict(u, f, d) }

def bayesian_signals(row, df):
    breakdown, adj_total = [], 0
    def add_sig(name, val, inte, adj): nonlocal adj_total; breakdown.append({'signal': name, 'value': val, 'interp': inte, 'adj': adj}); adj_total += adj
    
    vp = float(row.get('vix_pct', 0.5)); v,i,a = f"{vp:.0%}", "NEUTRAL", 0.0
    if vp > 0.8: i,a = "EXTREME FEAR → bear", -0.45
    elif vp < 0.2: i,a = "COMPLACENCY → bull", 0.30
    add_sig('India VIX percentile', v, i, a)
    
    vts = float(row.get('vix_spread', 0)); v,i,a = f"{vts:+.2f}", "NORMAL T/S", 0.0
    if vts > 0.5: i,a = "INVERTED → big move", -0.40
    elif vts < -2.0: i,a = "STEEP CONTANGO → bull", 0.20
    add_sig('VIX term spread', v, i, a)
    
    rsi14 = float(row.get('rsi', 50)); v,i,a = f"{rsi14:.1f}", "NEUTRAL", 0.0
    if rsi14 < 35: i,a = "DEEPLY OVERSOLD → bounce", 0.55
    elif rsi14 > 70: i,a = "OVERBOUGHT → resistance", -0.45
    add_sig('RSI-14', v, i, a)
    
    z20 = float(row.get('z20', 0)); v,i,a = f"{z20:+.2f}", "NEAR MEAN", 0.0
    if z20 < -1.5: i,a = "FAR BELOW MEAN", 0.45
    elif z20 > 1.5: i,a = "FAR ABOVE MEAN", -0.45
    add_sig('Z-score 20d', v, i, a)
    
    spr = float(row.get('sp_ret1', 0)); v,i,a = f"{spr:+.1%}", "IN LINE", 0.0
    if spr < -0.01: i,a = "WEAK → Nifty bear", -0.30
    elif spr > 0.01: i,a = "STRONG → Nifty bull", 0.30
    add_sig('SP500 yesterday', v, i, a)
    
    bnv = float(row.get('bn_vs_n', 0)); v,i,a = f"{bnv:+.1%}", "IN LINE", 0.0
    if bnv < -0.005: i,a = "BN LAGGING → bear", -0.20
    elif bnv > 0.005: i,a = "BN LEADING → bull", 0.20
    add_sig('BankNifty vs Nifty', v, i, a)
    
    v15 = float(row.get('v15_ret', 0)); v,i,a = f"{v15:+.1%}", "FLAT TODAY", 0.0
    if v15 > 0.01: i,a = "VIX RISING TODAY → bear", -0.20
    elif v15 < -0.01: i,a = "VIX FALLING TODAY → bull", 0.20
    add_sig('VIX 15m intraday', v, i, a)
    
    n15 = float(row.get('n15_ret', 0)); v,i,a = f"{n15:+.1%}", "FLAT TODAY", 0.0
    if n15 > 0.002: i,a = "NIFTY RISING TODAY", 0.15
    elif n15 < -0.002: i,a = "NIFTY FALLING TODAY", -0.15
    add_sig('Nifty 15m intraday', v, i, a)
    
    fz = float(row.get('fii_z', 0))
    if fz != 0: v,i,a = (f"{fz:+.1f}z", "STRONG INFLOW", 0.25) if fz>1.5 else (f"{fz:+.1f}z", "STRONG OUTFLOW", -0.25) if fz<-1.5 else (f"{fz:+.1f}z", "NEUTRAL", 0)
    else: v,i,a = "–", "building history", 0.0
    add_sig('FII/DII flow', v, i, a)
    
    pz = float(row.get('pcr_z', 0))
    if pz != 0: v,i,a = (f"{pz:+.1f}z", "OVERSOLD", 0.15) if pz>1.5 else (f"{pz:+.1f}z", "OVERBOUGHT", -0.15) if pz<-1.5 else (f"{pz:+.1f}z", "NEUTRAL", 0)
    else: v,i,a = "–", "building history", 0.0
    add_sig('PCR signal', v, i, a)
    
    hist = df.dropna(subset=['fwd_7d']).copy()
    if len(hist) < 50: return None
    pu, pd_ = max((hist['fwd_7d']>0.01).mean(), 0.01), max((hist['fwd_7d']<-0.01).mean(), 0.01)
    lu, ld = np.log(pu/(1-pu)), np.log(pd_/(1-pd_))
    
    res_u, res_d = 1/(1+np.exp(-(lu+adj_total))), 1/(1+np.exp(-(ld-adj_total)))
    s = res_u + res_d + 0.1
    u, f, d = res_u/s, 0.1/s, res_d/s
    return { 'p_up': u, 'p_flat': f, 'p_down': d, 'signals_count': len(breakdown), 'verdict': get_verdict(u, f, d), 'breakdown': breakdown }

def compute_verdict(horizon=7):
    df = build_features()
    if df.empty: return None
    row = df.iloc[-1]; spot = float(row['close'])
    
    emp = empirical_probs(df, horizon, row, spot)
    mc = monte_carlo(df, spot, horizon)
    bay = bayesian_signals(row, df)
    
    eu, ef, ed = (emp['p_up']*0.4 + mc['p_up']*0.35 + bay['p_up']*0.25), (emp['p_flat']*0.4 + mc['p_flat']*0.35 + bay['p_flat']*0.25), (emp['p_down']*0.4 + mc['p_down']*0.35 + bay['p_down']*0.25)
    mx = max(eu, ef, ed)
    verdict = 'UP' if mx==eu else 'DOWN' if mx==ed else 'FLAT'
    methods_agree = sum([emp['verdict'] == verdict, mc['verdict'] == verdict, bay['verdict'] == verdict])
    
    active_sources = [
        {'name': 'Nifty daily (11yr)', 'active': os.path.exists(PATHS['nifty'])},
        {'name': 'India VIX daily', 'active': os.path.exists(PATHS['vix'])},
        {'name': 'VIX term structure', 'active': os.path.exists(PATHS['vterm'])},
        {'name': 'BankNifty daily', 'active': os.path.exists(PATHS['bank_nifty'])},
        {'name': 'SP500 daily', 'active': os.path.exists(PATHS['sp500'])},
        {'name': 'Nifty 15m intraday', 'active': os.path.exists(PATHS['n15m'])},
        {'name': 'VIX 15m intraday', 'active': os.path.exists(PATHS['v15m'])},
        {'name': 'FII/DII (building)', 'active': os.path.exists(PATHS['fii']) and not _load('fii').empty},
        {'name': 'PCR (building)', 'active': os.path.exists(PATHS['pcr']) and not _load('pcr').empty}
    ]
    return { 'spot': spot, 'horizon': horizon, 'eu': eu, 'ef': ef, 'ed': ed, 'verdict': verdict, 'confidence': (mx-1/3)/(2/3)*100, 'methods_agree': methods_agree, 'empirical': emp, 'mc': mc, 'bayesian': bay, 'active_sources': active_sources }
