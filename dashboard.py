import os, sys, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import norm

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
</style>""", unsafe_allow_html=True)

try:
    from nifty_regime import build_regime_table, compute_score, classify, generate_signals
    from event_fetcher import load_events, is_event_day
    from probability_engine import compute_verdict
except ImportError as e: st.error(f"Import error: {e}"); st.stop()

@st.cache_data(ttl=300)
def load(): return build_regime_table()
df = load()
if df.empty: st.error("No data."); st.stop()

row = df.iloc[-1]; score, comps = compute_score(row); regime = classify(score); sigs = generate_signals(row, score)
close, vix, atr10 = float(row.get('close', 0)), float(row.get('vix', 0)), float(row.get('atr10', 1))
rsi, z = float(row.get('rsi', 50)), float(row.get('z_score', 0))
events = load_events(30); nxt = events[0] if events else None; ndelta = (nxt['date']-date.today()).days if nxt else None

# Header
hc1, hc2, hc3 = st.columns([2,5,1])
with hc1: st.markdown('<div style="padding:10px 20px"><div style="color:#f5a623;font-weight:600;letter-spacing:.3em;font-size:.7rem">CALEB</div></div>', unsafe_allow_html=True)
with hc2:
    t1,t2,t3,t4 = st.columns(4)
    t1.metric("NIFTY", f"{close:,.0f}"); t2.metric("VIX", f"{vix:.2f}"); t3.metric("ATR-10", f"{atr10:.0f}")
    t4.metric("NEXT EVENT", f"{nxt['label']} (+{ndelta}d)" if nxt else "CLEAR")
with hc3:
    if st.button("⟳ SYNC"): st.cache_data.clear(); st.rerun()

lc, r_col = st.columns([1,3], gap="small")
with lc:
    st.markdown(f'<div class="regime-block rb-{"g" if regime=="GREEN" else "y" if regime=="YELLOW" else "r"}"><div class="rscore" style="color:{"#22c55e" if regime=="GREEN" else "#f59e0b" if regime=="YELLOW" else "#ef4444"}">{score}</div><div>{regime}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">MARKET STATS</div>', unsafe_allow_html=True)
    for k,v in [("Z-SCORE",f"{z:+.2f}"), ("RSI",f"{rsi:.1f}"), ("TREND", f"{row.get('trend',0)}"), ("VIX %ILE", f"{row.get('vix_pct',0):.0%}"), ("VIX SPREAD", f"{row.get('vix_spread',0):+.2f}")]:
        st.markdown(f'<div class="srow"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">EVENT CALENDAR</div>', unsafe_allow_html=True)
    for e in events[:8]: st.markdown(f'<div class="ep epd">{e["date"]} {e["label"]}</div>', unsafe_allow_html=True)

with r_col:
    ta1, ta2, ta3, ta4, ta5 = st.tabs(["TRADE SIGNALS", "PROBABILITY ENGINE", "CHARTS", "BACKTEST", "RULES"])
    with ta1:
        if 'condor' in sigs:
            c = sigs['condor']
            st.markdown(f'<div class="tc tcc"><div class="tch">IRON CONDOR</div><div class="tcs"><div><div style="font-size:.5rem;color:#1a2535">SELL PE</div><div class="tsv">{c["sell_pe"]:,}</div></div><div style="font-size:1.1rem;padding-bottom:2px">↔</div><div><div style="font-size:.5rem;color:#1a2535">SELL CE</div><div class="tsv">{c["sell_ce"]:,}</div></div></div><div style="font-size:.55rem;color:#1a2535;margin-top:8px">{c["logic"]}</div></div>', unsafe_allow_html=True)
        else: st.warning("NO TRADE TODAY — Regime or Volatility out of range.")
    with ta2:
        res = compute_verdict(7)
        if res:
            c1, c2 = st.columns(2)
            c1.markdown(f'<div style="border:1px solid #0f1520;padding:20px;text-align:center"><div style="font-size:.5rem;letter-spacing:.2em">7-DAY VERDICT</div><div style="font-size:2.8rem;font-weight:600;color:{"#22c55e" if res["verdict"]=="UP" else "#ef4444" if res["verdict"]=="DOWN" else "#f5a623"}">{res["verdict"]}</div><div style="font-size:.6rem;color:#1a2535;letter-spacing:.1em">{res["confidence"]:.0f}% CONFIDENCE · {res["horizon"]} DAY HORIZON</div></div>', unsafe_allow_html=True)
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
