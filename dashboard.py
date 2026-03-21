import os, sys, warnings, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="CALEB", page_icon="⬡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
*,[class*="css"]{font-family:'IBM Plex Mono',monospace!important}
.block-container{padding:0!important;max-width:100%!important}
.regime-block{border:1px solid;border-radius:3px;padding:14px;text-align:center;margin-bottom:12px}
.rb-g{border-color:#22c55e;background:rgba(34,197,94,.04)}
.rb-y{border-color:#f59e0b;background:rgba(245,158,11,.04)}
.rb-r{border-color:#ef4444;background:rgba(239,68,68,.04)}
.rscore{font-size:3.5rem;font-weight:600;line-height:1}
.sec{font-size:.5rem;letter-spacing:.2em;color:#1a2535;font-weight:600;text-transform:uppercase;margin:12px 0 5px;padding-bottom:3px;border-bottom:1px solid #0f1520}
.srow{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #080c14;font-size:.6rem}
.tc{border:1px solid #0f1520;border-radius:3px;padding:12px 14px;margin-bottom:8px;position:relative;overflow:hidden}
.tcc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:#22c55e}
.tch{font-size:.52rem;letter-spacing:.15em;color:#1a2535;margin-bottom:8px;font-weight:600}
.tcs{display:flex;gap:16px;align-items:flex-end}
.tsv{font-size:1.4rem;font-weight:600}
.ep{display:inline-block;border:1px solid;border-radius:2px;padding:2px 7px;font-size:.52rem;margin:2px}
.epd{border-color:#ef444433;color:#ef4444;background:rgba(239,68,68,.03)}
.metric-row{background:#000;padding:15px;display:flex;justify-content:space-around;border-bottom:1px solid #0f1520}
.metric-card{text-align:center}
.metric-label{font-size:.5rem;color:#1a2535;letter-spacing:.2em}
.metric-value{font-size:1.8rem;font-weight:600;color:#fff}
</style>""", unsafe_allow_html=True)

try:
    from nifty_regime import build_regime_table, compute_score, classify, generate_signals, run_backtest
    from event_fetcher import load_events
    from probability_engine import compute_verdict
except ImportError as e: st.error(f"Import error: {e}"); st.stop()

@st.cache_data(ttl=60)
def load(): return build_regime_table()
df_all = load()
if df_all.empty: st.error("No data."); st.stop()

df = df_all.dropna(subset=['atr10'])
row = df.iloc[-1]; score, comps = compute_score(row); regime = classify(score); sigs = generate_signals(row, score)
close, vix, atr10 = float(row.get('close', 0)), float(row.get('vix', 0)), float(row.get('atr10', 1))
rsi, z = float(row.get('rsi', 50)), float(row.get('z_score', 0))
events = load_events(30); nxt = events[0] if events else None; ndelta = (nxt['date']-date.today()).days if nxt else None

# Metric row for visibility
st.markdown(f"""<div class="metric-row">
    <div class="metric-card"><div class="metric-label">SPOT</div><div class="metric-value">{close:,.2f}</div></div>
    <div class="metric-card"><div class="metric-label">VIX</div><div class="metric-value">{vix:.2f}</div></div>
    <div class="metric-card"><div class="metric-label">ATR-10</div><div class="metric-value">{atr10:.1f}</div></div>
    <div class="metric-card"><div class="metric-label">NEXT EVENT</div><div class="metric-value" style="font-size:1rem;color:#f5a623">{nxt['label'] if nxt else 'CLEAR'}</div></div>
</div>""", unsafe_allow_html=True)

lc, r_col = st.columns([1,3], gap="small")
with lc:
    st.markdown(f'<div class="regime-block rb-{"g" if regime=="GREEN" else "y" if regime=="YELLOW" else "r"}"><div class="rscore" style="color:{"#22c55e" if regime=="GREEN" else "#f59e0b" if regime=="YELLOW" else "#ef4444"}">{score}</div><div>{regime}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">MARKET STATS</div>', unsafe_allow_html=True)
    for k,v in [("Z-SCORE",f"{z:+.2f}"), ("RSI",f"{rsi:.1f}"), ("TREND", f"{row.get('trend',0)}"), ("VIX %ILE", f"{row.get('vix_pct',0):.0%}"), ("VIX SPREAD", f"{row.get('vix_spread',0):+.2f}")]:
        st.markdown(f'<div class="srow"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">EVENT CALENDAR</div>', unsafe_allow_html=True)
    for e in events[:10]: st.markdown(f'<div class="ep epd">{e["date"]} {e["label"]}</div>', unsafe_allow_html=True)
    if st.button("⟳ SYNC DATA"): st.cache_data.clear(); st.rerun()

with r_col:
    ta1, ta2, ta3, ta4, ta5 = st.tabs(["TRADE SIGNALS", "PROBABILITY ENGINE", "CHARTS", "BACKTEST", "RULES"])
    with ta1:
        if 'condor' in sigs:
            c = sigs['condor']
            st.markdown(f'<div class="tc tcc"><div class="tch">IRON CONDOR</div><div class="tcs"><div><div style="font-size:.5rem;color:#1a2535">SELL PE</div><div class="tsv">{c["sell_pe"]:,}</div></div><div style="font-size:1.1rem;padding-bottom:2px">↔</div><div><div style="font-size:.5rem;color:#1a2535">SELL CE</div><div class="tsv">{c["sell_ce"]:,}</div></div></div><div style="font-size:.55rem;color:#1a2535;margin-top:8px">{c["logic"]}</div></div>', unsafe_allow_html=True)
        else: st.warning("NO TRADE TODAY — Regime is RED or Market Stress is too high.")
    with ta2:
        res = compute_verdict(7)
        if res:
            c1, c2 = st.columns(2)
            color = "#22c55e" if res["verdict"]=="UP" else "#ef4444" if res["verdict"]=="DOWN" else "#f5a623"
            c1.markdown(f'<div style="border:1px solid #0f1520;padding:20px;text-align:center"><div style="font-size:.5rem;letter-spacing:.2em">7-DAY VERDICT</div><div style="font-size:2.8rem;font-weight:600;color:{color}">{res["verdict"]}</div><div style="font-size:.6rem;color:#1a2535;letter-spacing:.1em">{res["confidence"]:.0f}% CONFIDENCE</div></div>', unsafe_allow_html=True)
            with c2:
                st.write("CONSENSUS PROBABILITIES")
                st.progress(res['eu'], text=f"UP: {res['eu']:.1%}")
                st.progress(res['ef'], text=f"FLAT: {res['ef']:.1%}")
                st.progress(res['ed'], text=f"DOWN: {res['ed']:.1%}")
            
            st.markdown('<div class="sec">3-METHOD BREAKDOWN</div>', unsafe_allow_html=True)
            col_e, col_m, col_b = st.columns(3)
            col_e.metric("Empirical", f"{res['empirical']['p_up']:.0%}")
            col_m.metric("Monte Carlo", f"{res['mc']['p_up']:.0%}")
            col_b.metric("Bayesian", f"{res['bayesian']['p_up']:.0%}")

            st.markdown('<div class="sec">MONTE CARLO PATHS</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for p in res['mc']['paths_sample']: fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(width=0.4), opacity=0.3, hoverinfo='skip'))
            fig.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=250, margin=dict(l=5,r=5,t=5,b=5), showlegend=False, xaxis=dict(visible=False), yaxis=dict(gridcolor='#0f1520', tickfont=dict(size=7)))
            st.plotly_chart(fig, use_container_width=True)
    with ta3:
        st.write("MARKET REGIME VISUALIZATION")
        hist = df.tail(180).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['close'], mode='lines', line=dict(color='#fff', width=1)))
        for i in range(len(hist)-1):
            sc, _ = compute_score(hist.iloc[i]); reg = classify(sc)
            color = "rgba(34,197,94,0.1)" if reg=="GREEN" else "rgba(239,68,68,0.1)" if reg=="RED" else "rgba(245,158,11,0.1)"
            fig.add_vrect(x0=hist.iloc[i]['date'], x1=hist.iloc[i+1]['date'], fillcolor=color, layer="below", line_width=0)
        fig.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=400, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(gridcolor='#0f1520'), yaxis=dict(gridcolor='#0f1520'))
        st.plotly_chart(fig, use_container_width=True)
    with ta4:
        st.write("BACKTEST PERFORMANCE (Since 2015)")
        res_bt, yearly = run_backtest(df_all)
        c1, c2, c3 = st.columns(3)
        c1.metric("GREEN SAFE", f"{np.mean(res_bt['GREEN']):.1%}")
        c2.metric("ALL DAYS SAFE", f"{np.mean(res_bt['GREEN']+res_bt['YELLOW']+res_bt['RED']):.1%}")
        c3.metric("FILTER LIFT", f"{np.mean(res_bt['GREEN']) - np.mean(res_bt['GREEN']+res_bt['YELLOW']+res_bt['RED']):+.1%}")
        st.write("YEARLY BREAKOUT")
        y_df = pd.DataFrame([{'Year': y, 'Safe %': f"{v*100:.1f}%"} for y,v in yearly.items()])
        st.table(y_df)
    with ta5:
        st.markdown("""### Trading Rules
1. **Capital First**: If Regime is RED, do nothing. No exceptions.
2. **Event Guard**: Never sell options on RBI/FOMC decision days.
3. **Expiry Rule**: Close all positions by Tuesday morning of expiry week.
4. **Take Profit**: Exit Iron Condors at 50% max profit.
5. **Stop Loss**: Exit immediately if Spot breaches CE or PE strike.
6. **Size Scaling**: Trade full size in GREEN, half size in YELLOW.
7. **Whipsaw Awareness**: In downtrends, place Bear Call strikes above the predicted bounce peak.""")
