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
    from nifty_regime import build_regime_table, compute_score, classify, generate_signals, run_backtest, CONDOR_ATR_MULT
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
        if not res:
            st.warning("Data incomplete")
        else:
            st.markdown('<div class="sec" style="margin-top:0;">PROBABILITY ENGINE — 7-DAY HORIZON — 3 METHODS CROSS-VALIDATED</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1.2, 2.5, 1], gap="medium")
            with c1:
                v = res['verdict']
                col = '#ef4444' if v == 'DOWN' else '#22c55e' if v == 'UP' else '#f5a623'
                conf = res['confidence']
                conf_text = "HIGH" if conf > 60 else "MODERATE" if conf > 30 else "LOW"
                st.markdown(f'''
                <div style="border:1px solid {col}33; padding:20px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:0.55rem; letter-spacing:0.2em; color:#1a2535; margin-bottom:15px;">ENSEMBLE VERDICT</div>
                    <div style="font-size:4rem; font-weight:700; color:{col}; line-height:1;">{v}</div>
                    <div style="font-size:0.65rem; color:{col}; margin:15px 0; font-weight:600; letter-spacing:0.1em;">{conf_text} CONFIDENCE</div>
                    <div style="width:100%; height:4px; background:#0f1520; margin-bottom:10px;">
                        <div style="width:{conf}%; height:100%; background:{col};"></div>
                    </div>
                    <div style="font-size:0.55rem; color:#1a2535;">{conf:.0f}% confidence · {res["methods_agree"]}/3 methods agree</div>
                </div>
                ''', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div style="display:flex; justify-content:space-between; font-size:0.45rem; letter-spacing:0.15em; color:#1a2535; margin-bottom:5px; padding-bottom:5px; border-bottom:1px solid #0f1520;"><span>METHODVERDICT</span><span>UP / FLAT / DOWN</span><span>AGREE</span></div>', unsafe_allow_html=True)
                def pbar(label, data):
                    pu, pf, pd_ = data['p_up']*100, data['p_flat']*100, data['p_down']*100
                    vd = data['verdict']
                    vcol = '#ef4444' if vd == 'DOWN' else '#22c55e' if vd == 'UP' else '#f5a623'
                    return f'''
                    <div style="border:1px solid #0f1520; padding:12px; margin-bottom:10px; border-radius:3px;">
                        <div style="font-size:0.6rem; color:#1a2535; letter-spacing:0.05em; display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>{label}</span>
                            <span style="color:{vcol}; font-weight:600;">{vd}</span>
                        </div>
                        <div style="display:flex; height:4px; width:100%; margin-bottom:6px; background:#0f1520;">
                            <div style="width:{pu}%; background:#22c55e;"></div>
                            <div style="width:{pf}%; background:#f5a623;"></div>
                            <div style="width:{pd_}%; background:#ef4444;"></div>
                        </div>
                        <div style="font-size:0.55rem; display:flex; justify-content:space-between; color:#1a2535;">
                            <span style="color:#22c55e">↑ {pu:.0f}%</span>
                            <span style="color:#f5a623">↔ {pf:.0f}%</span>
                            <span style="color:#ef4444">↓ {pd_:.0f}%</span>
                        </div>
                    </div>
                    '''
                st.markdown(pbar(f"Empirical — 11yr history (n={res['empirical']['n_samples']})", res['empirical']), unsafe_allow_html=True)
                st.markdown(pbar(f"Monte Carlo — {res['mc']['n_paths']:,} paths (fat tails)", res['mc']), unsafe_allow_html=True)
                st.markdown(pbar(f"Bayesian — {res['bayesian']['signals_count']} signals updated", res['bayesian']), unsafe_allow_html=True)
                
            with c3:
                st.markdown('<div style="font-size:0.5rem; letter-spacing:0.15em; color:#1a2535; margin-bottom:10px;">EXPECTED PRICE AT DAY 7</div>', unsafe_allow_html=True)
                exp = res['expected']
                st.markdown(f'''
                <div style="border:1px solid #0f1520; padding:15px; border-radius:3px; font-size:0.65rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px; color:#1a2535;"><span>Bull (p95)</span><span style="color:#22c55e">{exp['bull']:,}</span></div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px; color:#1a2535;"><span>p75</span><span style="color:#22c55e">{exp['p75']:,}</span></div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px; color:#1a2535;"><span>Median</span><span style="color:#f5a623">{exp['median']:,}</span></div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px; color:#1a2535;"><span>p25</span><span style="color:#ef4444">{exp['p25']:,}</span></div>
                    <div style="display:flex; justify-content:space-between; color:#1a2535;"><span>Bear (p5)</span><span style="color:#ef4444">{exp['bear']:,}</span></div>
                </div>
                ''', unsafe_allow_html=True)

            st.write("")
            c_u, c_d = st.columns(2, gap="medium")
            with c_u:
                st.markdown('<div style="border:1px solid #0f1520; padding:15px; border-radius:3px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.55rem; color:#22c55e; letter-spacing:0.1em; margin-bottom:15px; font-weight:600;">UPSIDE — P(Nifty touches this within 7d)</div>', unsafe_allow_html=True)
                for item in res['empirical']['upside']:
                    pct = item['prob'] * 100
                    st.markdown(f'''<div style="display:flex; align-items:center; margin-bottom:10px; font-size:0.6rem; color:#1a2535;"><div style="width:100px;">{item['label']}</div><div style="flex-grow:1; height:6px; background:#0f1520; margin:0 15px;"><div style="width:{pct}%; height:100%; background:#22c55e;"></div></div><div style="width:30px; text-align:right; color:#22c55e; font-weight:600;">{pct:.0f}%</div></div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c_d:
                st.markdown('<div style="border:1px solid #0f1520; padding:15px; border-radius:3px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.55rem; color:#ef4444; letter-spacing:0.1em; margin-bottom:15px; font-weight:600;">DOWNSIDE — P(Nifty touches this within 7d)</div>', unsafe_allow_html=True)
                for item in res['empirical']['downside']:
                    pct = item['prob'] * 100
                    st.markdown(f'''<div style="display:flex; align-items:center; margin-bottom:10px; font-size:0.6rem; color:#1a2535;"><div style="width:100px;">{item['label']}</div><div style="flex-grow:1; height:6px; background:#0f1520; margin:0 15px;"><div style="width:{pct}%; height:100%; background:#ef4444;"></div></div><div style="width:30px; text-align:right; color:#ef4444; font-weight:600;">{pct:.0f}%</div></div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="sec">BAYESIAN SIGNAL BREAKDOWN — WHAT PUSHED THE VERDICT</div>', unsafe_allow_html=True)
            trs = []
            for sig in res['bayesian']['breakdown']:
                valcol = '#22c55e' if sig['adj']>0 else '#ef4444' if sig['adj']<0 else '#1a2535'
                aj = f"{sig['adj']:+.2f}" if sig['adj']!=0 else "0.00"
                trs.append(f'''<tr style="border-bottom:1px solid #0f1520; font-size:0.65rem; color:#1a2535;"><td style="padding:10px 0;">{sig['signal']}</td><td style="padding:10px 0;">{sig['value']}</td><td style="padding:10px 0; color:{valcol}; text-align:center;">{sig['interp']}</td><td style="padding:10px 0; color:{valcol}; text-align:right; font-family:monospace;">{aj}</td></tr>''')
            st.markdown(f'''<table style="width:100%; border-collapse:collapse; margin-bottom:20px;"><tr style="border-bottom:1px solid #0f1520; font-size:0.45rem; letter-spacing:0.15em; color:#1a2535; text-align:left;"><th style="padding-bottom:10px;">SIGNAL</th><th style="padding-bottom:10px;">VALUE</th><th style="padding-bottom:10px; text-align:center;">INTERPRETATION</th><th style="padding-bottom:10px; text-align:right;">ADJ</th></tr>{"".join(trs)}</table>''', unsafe_allow_html=True)
            
            st.markdown('<div class="sec">DATA SOURCES ACTIVE</div>', unsafe_allow_html=True)
            chips = []
            for src in res['active_sources']:
                if src['active']: chips.append(f'<span style="display:inline-block; border:1px solid #22c55e66; color:#22c55e; padding:4px 12px; border-radius:3px; font-size:0.55rem; margin:0 8px 8px 0; background:rgba(34,197,94,0.05);">✓ {src["name"]}</span>')
                else: chips.append(f'<span style="display:inline-block; border:1px solid #1a2535; color:#1a2535; padding:4px 12px; border-radius:3px; font-size:0.55rem; margin:0 8px 8px 0;">X {src["name"]}</span>')
            st.markdown(f'<div>{"".join(chips)}</div>', unsafe_allow_html=True)
    with ta3:
        hist = df.tail(150).copy(); hist['score'] = hist.apply(lambda r: compute_score(r)[0], axis=1)
        st.markdown('<div class="sec">REGIME PRICE TRACKER</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=hist['date'], y=hist['close'], name="Nifty Close", line=dict(color='#fff', width=1.5)))
        for i in range(len(hist)-1):
            r = classify(hist.iloc[i]['score'])
            color = "rgba(34,197,94,0.15)" if r=="GREEN" else "rgba(239,68,68,0.15)" if r=="RED" else "rgba(245,158,11,0.1)"
            fig1.add_vrect(x0=hist.iloc[i]['date'], x1=hist.iloc[i+1]['date'], fillcolor=color, layer="below", line_width=0)
        fig1.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=300, margin=dict(l=5,r=5,t=5,b=5), yaxis=dict(gridcolor='#0f1520'), xaxis=dict(visible=False))
        st.plotly_chart(fig1, width='stretch')

        st.markdown('<div class="sec">REGIME SCORE HISTORY</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist['date'], y=hist['score'], line=dict(color='#f5a623', width=1.5), fill='tozeroy', fillcolor='rgba(245,166,35,0.05)'))
        fig2.add_hline(y=65, line_dash="dash", line_color="#22c55e", opacity=0.5)
        fig2.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=200, margin=dict(l=5,r=5,t=5,b=5), yaxis=dict(gridcolor='#0f1520'), xaxis=dict(visible=False))
        st.plotly_chart(fig2, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec">ATR & CONDOR GAP</div>', unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=hist['date'], y=hist['atr10'], line=dict(color='#3b82f6')))
            fig3.add_hline(y=150, line_dash="dot", line_color="#ef4444")
            fig3.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=180, margin=dict(l=5,r=5,t=5,b=5), yaxis=dict(gridcolor='#0f1520'))
            st.plotly_chart(fig3, width='stretch')
        with c2:
            st.markdown('<div class="sec">INDIA VIX LEVELS</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=hist['date'], y=hist['vix'], line=dict(color='#a855f7')))
            fig4.add_hline(y=14, line_color="#22c55e", opacity=0.5); fig4.add_hline(y=20, line_color="#ef4444", opacity=0.5)
            fig4.update_layout(plot_bgcolor='#060810', paper_bgcolor='#060810', height=180, margin=dict(l=5,r=5,t=5,b=5), yaxis=dict(gridcolor='#0f1520'))
            st.plotly_chart(fig4, width='stretch')

    with ta4:
        st.write("BACKTEST PERFORMANCE (Since 2015)")
        res_bt, yearly = run_backtest(df_all)
        if res_bt and yearly:
            c1, c2, c3 = st.columns(3)
            tg = res_bt.get('GREEN', []) or [0]
            ta = (res_bt.get('GREEN',[]) + res_bt.get('YELLOW',[]) + res_bt.get('RED',[])) or [0]
            c1.metric("GREEN SAFE", f"{np.mean(tg):.1%}")
            c2.metric("ALL DAYS SAFE", f"{np.mean(ta):.1%}")
            c3.metric("FILTER LIFT", f"{np.mean(tg) - np.mean(ta):+.1%}")
            st.write("YEARLY PERFORMANCE (GREEN DAYS ONLY)")
            y_df = pd.DataFrame([{'Year': y, 'Success Rate': f"{v*100:.1f}%"} for y,v in sorted(yearly.items(), reverse=True)])
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
