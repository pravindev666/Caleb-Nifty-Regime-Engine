"""
dashboard.py — Nifty Regime Engine Dashboard
=============================================
Run:  streamlit run dashboard.py

Install:
  pip install streamlit plotly pandas numpy scipy yfinance requests
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Regime Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

.regime-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    font-family: 'DM Sans', sans-serif;
}
.regime-green  { background: #0d2b1a; border: 2px solid #1a7a3c; }
.regime-yellow { background: #2b2205; border: 2px solid #a07800; }
.regime-red    { background: #2b0d0d; border: 2px solid #8b1a1a; }

.score-number {
    font-size: 4.5rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin: 0.5rem 0;
}
.score-green  { color: #4ade80; }
.score-yellow { color: #fbbf24; }
.score-red    { color: #f87171; }

.signal-label {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.trade-box {
    background: #0f1117;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 4px solid;
    margin-bottom: 1rem;
}
.trade-condor { border-color: #4ade80; }
.trade-bull   { border-color: #60a5fa; }
.trade-bear   { border-color: #f87171; }
.trade-wait   { border-color: #6b7280; }

.trade-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}
.strike-big {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #f8f8f2;
}
.strike-label {
    font-size: 0.72rem;
    opacity: 0.5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-pill {
    background: #1a1d27;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #f8f8f2;
}
.metric-label {
    font-size: 0.68rem;
    opacity: 0.45;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

.bar-outer {
    background: #1a1d27;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin: 4px 0;
}
.bar-inner { height: 100%; border-radius: 6px; }

.rule-chip {
    display: inline-block;
    background: #1a1d27;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    margin: 3px;
    color: #aaa;
    border: 1px solid #2a2d3a;
}

.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.4;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

div[data-testid="stHorizontalBlock"] { gap: 1rem; }
div[data-testid="column"] { padding: 0; }
.stSpinner > div { border-top-color: #4ade80 !important; }
</style>
""", unsafe_allow_html=True)


# ── Import engine ─────────────────────────────────────────────────────────────
try:
    from nifty_regime import (
        build_regime_table, compute_score, classify,
        generate_signals, run_backtest,
        WEIGHTS, GREEN_MIN, YELLOW_MIN, CONDOR_ATR_MULT
    )
    ENGINE_OK = True
except ImportError as e:
    ENGINE_OK = False
    ENGINE_ERR = str(e)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_data():
    if not ENGINE_OK:
        return pd.DataFrame()
    try:
        return build_regime_table()
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()


def get_today(df):
    if df.empty:
        return None, None, None, None, None
    row   = df.dropna(subset=['atr10']).iloc[-1]
    score, components = compute_score(row)
    regime = classify(score)
    sigs   = generate_signals(row, score)
    return row, score, components, sigs, regime


# ── Helpers ───────────────────────────────────────────────────────────────────
def regime_color(r):
    return {'GREEN': '#4ade80', 'YELLOW': '#fbbf24', 'RED': '#f87171'}.get(r, '#888')

def regime_bg(r):
    return {'GREEN': 'regime-green', 'YELLOW': 'regime-yellow', 'RED': 'regime-red'}.get(r, '')

def score_class(r):
    return {'GREEN': 'score-green', 'YELLOW': 'score-yellow', 'RED': 'score-red'}.get(r, '')

def fmt(v, dec=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:,.{dec}f}'

def score_bar(val, color):
    pct = max(0, min(100, val))
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:2px 0;">
      <div class="bar-outer" style="flex:1;">
        <div class="bar-inner" style="width:{pct}%;background:{color};"></div>
      </div>
      <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#888;min-width:32px;text-align:right">{val}</span>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

if not ENGINE_OK:
    st.error(f"Cannot import nifty_regime.py. Make sure it is in the same folder.\n\n{ENGINE_ERR}")
    st.stop()

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## Nifty Regime Engine")
    st.markdown('<p style="opacity:0.4;margin:-8px 0 0;font-size:0.85rem;">Iron Condor · Bull Spread · Bear Spread · Capital Protection</p>', unsafe_allow_html=True)
with col_h2:
    if st.button("⟳  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Load
with st.spinner("Loading market data..."):
    df = load_data()

if df.empty:
    st.error("No data loaded. Run `python data_updater.py` first then refresh.")
    st.stop()

row, score, components, sigs, regime = get_today(df)
color = regime_color(regime)
date_str = str(row['date'])[:10]

close  = float(row.get('close', 0))
vix    = float(row.get('vix', 0))
atr10  = float(row.get('atr10', 0))
atr20  = float(row.get('atr20', 0))
rsi    = float(row.get('rsi', 50))
z      = float(row.get('z_score', 0))
trend  = int(row.get('trend', 0))
vix_pct = float(row.get('vix_pct', 0.5))
spread = float(row.get('vix_spread', 0)) if not np.isnan(row.get('vix_spread', np.nan)) else 0

atr_ratio = atr10 / max(atr20, 0.001)
trend_label = {1: '↑ Uptrend', 0: '→ Sideways', -1: '↓ Downtrend'}.get(trend, '?')
trend_color = {1: '#4ade80', 0: '#fbbf24', -1: '#f87171'}.get(trend, '#888')

# ── ROW 1: Regime score + Trade signals ──────────────────────────────────────
col_score, col_trades = st.columns([1, 2], gap="large")

with col_score:
    st.markdown(f"""
    <div class="regime-card {regime_bg(regime)}">
      <div style="font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;opacity:0.5;margin-bottom:0.25rem">{date_str}</div>
      <div class="score-number {score_class(regime)}">{score}</div>
      <div style="font-size:0.75rem;opacity:0.4;margin:-4px 0 8px">out of 100</div>
      <div class="signal-label" style="color:{color}">{'✅ ' if regime=='GREEN' else '⚠️ ' if regime=='YELLOW' else '🛑 '}{regime}</div>
      <div style="margin-top:1.2rem;font-size:0.8rem;opacity:0.6;">
        {'Good conditions to sell premium' if regime=='GREEN' else 'Caution — trade smaller' if regime=='YELLOW' else 'Do not trade — protect capital'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Score breakdown bars
    st.markdown('<div class="section-header">Score breakdown</div>', unsafe_allow_html=True)
    comp_labels = {
        'vix_level': 'VIX level',
        'vix_term':  'VIX term structure',
        'atr_ratio': 'ATR expansion',
        'vol_score': 'Vol composite',
        'global':    'Global stress',
    }
    comp_colors = {
        'vix_level': '#a78bfa',
        'vix_term':  '#818cf8',
        'atr_ratio': '#38bdf8',
        'vol_score': '#34d399',
        'global':    '#fb923c',
    }
    for k, label in comp_labels.items():
        v = components.get(k, 0)
        c2 = comp_colors[k]
        pct = max(0, min(100, v))
        st.markdown(f"""
        <div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span style="font-size:0.74rem;opacity:0.6">{label}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{c2}">{v}/100</span>
          </div>
          <div class="bar-outer">
            <div class="bar-inner" style="width:{pct}%;background:{c2};opacity:0.85;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

with col_trades:
    st.markdown('<div class="section-header">Trade signals</div>', unsafe_allow_html=True)

    # Iron Condor
    if 'condor' in sigs:
        c = sigs['condor']
        action_color = '#4ade80' if regime == 'GREEN' else '#fbbf24'
        st.markdown(f"""
        <div class="trade-box trade-condor">
          <div class="trade-title">Iron Condor — {'Sell Premium' if regime=='GREEN' else 'Half size — caution'}</div>
          <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin:0.5rem 0;">
            <div>
              <div class="strike-label">Sell PE (put)</div>
              <div class="strike-big" style="color:#f87171">{c['sell_pe']:,}</div>
            </div>
            <div style="opacity:0.3;font-size:1.5rem">↔</div>
            <div>
              <div class="strike-label">Sell CE (call)</div>
              <div class="strike-big" style="color:#4ade80">{c['sell_ce']:,}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div class="strike-label">Safe width</div>
              <div class="strike-big" style="color:#fbbf24">{c['width']:,} pts</div>
            </div>
          </div>
          <div style="font-size:0.74rem;opacity:0.5;margin-top:4px">{c['logic']}</div>
          <div style="font-size:0.74rem;color:#fbbf24;margin-top:6px">Exit: 50% profit target · Exit immediately on breach · Never hold through event days</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="trade-box trade-wait">
          <div class="trade-title">Iron Condor</div>
          <div style="font-size:1rem;opacity:0.5;margin:0.5rem 0">No trade — regime not safe for selling premium</div>
          <div style="font-size:0.74rem;opacity:0.4;margin-top:6px">
            {'Need VIX percentile < 40%  (now ' + f'{vix_pct:.0%})' if components.get('vix_level', 0) < 60 else ''}
            {'  ·  Need ATR to compress  (ratio=' + f'{atr_ratio:.2f})' if components.get('atr_ratio', 0) < 60 else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Bull Call Spread
    if 'bull_spread' in sigs:
        b = sigs['bull_spread']
        st.markdown(f"""
        <div class="trade-box trade-bull">
          <div class="trade-title">Bull Call Spread — Buy the bounce</div>
          <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin:0.5rem 0;">
            <div>
              <div class="strike-label">Buy CE at</div>
              <div class="strike-big" style="color:#60a5fa">{b['buy_strike']:,}</div>
            </div>
            <div style="opacity:0.3;font-size:1.5rem">→</div>
            <div>
              <div class="strike-label">Sell CE at</div>
              <div class="strike-big" style="color:#93c5fd">{b['sell_strike']:,}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div class="strike-label">Width</div>
              <div class="strike-big" style="color:#fbbf24">{b['width']} pts</div>
            </div>
          </div>
          <div style="font-size:0.74rem;opacity:0.5;margin-top:4px">{b['logic']}</div>
          <div style="font-size:0.74rem;color:#60a5fa;margin-top:6px">Hold 3–5 days · Max loss = premium paid · Profit if Nifty rises</div>
        </div>
        """, unsafe_allow_html=True)

    # Bear Call Spread
    if 'bear_spread' in sigs:
        b = sigs['bear_spread']
        st.markdown(f"""
        <div class="trade-box trade-bear">
          <div class="trade-title">Bear Call Spread — Sell the bounce</div>
          <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin:0.5rem 0;">
            <div>
              <div class="strike-label">Sell CE at</div>
              <div class="strike-big" style="color:#f87171">{b['sell_ce']:,}</div>
            </div>
            <div style="opacity:0.3;font-size:1.5rem">→</div>
            <div>
              <div class="strike-label">Buy CE at</div>
              <div class="strike-big" style="color:#fca5a5">{b['buy_ce']:,}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div class="strike-label">Width</div>
              <div class="strike-big" style="color:#fbbf24">{b['width']} pts</div>
            </div>
          </div>
          <div style="font-size:0.74rem;opacity:0.5;margin-top:4px">{b['logic']}</div>
          {f'<div style="font-size:0.74rem;color:#fbbf24;margin-top:6px">⚠️ {b["warning"]}</div>' if b.get('warning') else ''}
          <div style="font-size:0.74rem;color:#f87171;margin-top:6px">Profit if Nifty stays below {b['sell_ce']:,} · Exit at 50% profit</div>
        </div>
        """, unsafe_allow_html=True)

    # If no trades at all
    if not sigs:
        st.markdown(f"""
        <div class="trade-box trade-wait" style="padding:2rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:0.5rem">🛑</div>
          <div style="font-size:1.1rem;font-weight:600;color:#f87171">No trade today</div>
          <div style="font-size:0.82rem;opacity:0.5;margin-top:0.5rem">Protect your capital. Wait for GREEN.</div>
        </div>
        """, unsafe_allow_html=True)

    # What to watch for GREEN
    if regime == 'RED':
        st.markdown('<div class="section-header" style="margin-top:1rem">Waiting for GREEN — what needs to change</div>', unsafe_allow_html=True)
        needs = []
        if components.get('vix_level', 0) < 60:
            needs.append(f"VIX needs to drop below ~14–16  (now {vix:.1f}, {vix_pct:.0%})")
        if components.get('vix_term', 0) < 60:
            needs.append(f"VIX term structure needs contango  (spread={spread:.2f}, need < 0.5)")
        if components.get('atr_ratio', 0) < 60:
            needs.append(f"ATR needs 5–7 days of compression  (ratio={atr_ratio:.2f}, need < 1.10)")
        for n in needs:
            st.markdown(f'<div style="font-size:0.78rem;opacity:0.6;padding:3px 0">→ {n}</div>', unsafe_allow_html=True)

# ── ROW 2: Market metrics ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Market metrics</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics = [
    (m1, "Nifty close",    f"{close:,.0f}",     ""),
    (m2, "India VIX",      f"{vix:.2f}",         "🔴 High" if vix > 20 else "🟡 Elevated" if vix > 14 else "🟢 Low"),
    (m3, "ATR-10 (pts/day)", f"{atr10:.0f}",    "Avg daily range"),
    (m4, "RSI-14",         f"{rsi:.1f}",          "🔵 Oversold" if rsi < 35 else "🔴 Overbought" if rsi > 70 else "Neutral"),
    (m5, "Z-score",        f"{z:+.2f}",           "Stretched" if abs(z) > 1.5 else "Normal"),
    (m6, "Trend",          trend_label,           ""),
]
for col, label, val, sub in metrics:
    with col:
        tc = trend_color if label == "Trend" else "#f8f8f2"
        st.markdown(f"""
        <div class="metric-pill">
          <div class="metric-value" style="color:{tc}">{val}</div>
          <div class="metric-label">{label}</div>
          <div style="font-size:0.65rem;opacity:0.35;margin-top:2px">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── ROW 3: Charts ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:1.5rem">Historical context — last 180 days</div>', unsafe_allow_html=True)

chart_df = df.dropna(subset=['close','atr10','vix_pct']).tail(180).copy()
chart_df['score'] = chart_df.apply(lambda r: compute_score(r)[0], axis=1)
chart_df['regime'] = chart_df['score'].apply(classify)
chart_df['color']  = chart_df['regime'].map({'GREEN':'#4ade80','YELLOW':'#fbbf24','RED':'#f87171'})

col_c1, col_c2 = st.columns(2, gap="large")

with col_c1:
    # Nifty price chart with regime coloring
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df['date'], y=chart_df['close'],
        mode='lines', name='Nifty',
        line=dict(color='#94a3b8', width=1.5),
        hovertemplate='%{x|%d %b}<br>Close: %{y:,.0f}<extra></extra>'
    ))
    # Regime background bands
    for regime_name, col_hex in [('GREEN','#4ade80'),('YELLOW','#fbbf24'),('RED','#f87171')]:
        mask = chart_df['regime'] == regime_name
        if mask.any():
            fig.add_trace(go.Scatter(
                x=chart_df.loc[mask,'date'],
                y=chart_df.loc[mask,'close'],
                mode='markers',
                marker=dict(color=col_hex, size=5, opacity=0.6),
                name=regime_name,
                hovertemplate=f'{regime_name}<br>%{{x|%d %b}}<br>%{{y:,.0f}}<extra></extra>'
            ))
    fig.update_layout(
        title=dict(text='Nifty — regime coloring', font=dict(size=13, color='#888'), x=0),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#888', family='DM Sans'),
        xaxis=dict(showgrid=False, color='#333'),
        yaxis=dict(showgrid=True, gridcolor='#1a1d27', color='#555'),
        legend=dict(orientation='h', y=-0.15, x=0, font=dict(size=11)),
        height=280, margin=dict(l=0, r=0, t=36, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_c2:
    # Regime score over time
    fig2 = go.Figure()
    fig2.add_hrect(y0=GREEN_MIN, y1=100, fillcolor='#4ade80', opacity=0.05, line_width=0)
    fig2.add_hrect(y0=YELLOW_MIN, y1=GREEN_MIN, fillcolor='#fbbf24', opacity=0.05, line_width=0)
    fig2.add_hrect(y0=0, y1=YELLOW_MIN, fillcolor='#f87171', opacity=0.05, line_width=0)
    fig2.add_hline(y=GREEN_MIN,  line_dash='dash', line_color='#4ade80', opacity=0.4, line_width=1)
    fig2.add_hline(y=YELLOW_MIN, line_dash='dash', line_color='#fbbf24', opacity=0.4, line_width=1)
    fig2.add_trace(go.Scatter(
        x=chart_df['date'], y=chart_df['score'],
        mode='lines', name='Regime score',
        line=dict(color='#818cf8', width=2),
        fill='tozeroy', fillcolor='rgba(129,140,248,0.08)',
        hovertemplate='%{x|%d %b}<br>Score: %{y}<extra></extra>'
    ))
    # Mark today
    fig2.add_vline(x=str(chart_df['date'].iloc[-1])[:10],
                   line_dash='dot', line_color=color, opacity=0.6, line_width=1.5)
    fig2.update_layout(
        title=dict(text='Regime score — 65+ = GREEN zone', font=dict(size=13, color='#888'), x=0),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#888', family='DM Sans'),
        xaxis=dict(showgrid=False, color='#333'),
        yaxis=dict(showgrid=True, gridcolor='#1a1d27', color='#555', range=[0, 105]),
        showlegend=False,
        height=280, margin=dict(l=0, r=0, t=36, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── ROW 4: ATR + VIX charts ───────────────────────────────────────────────────
col_c3, col_c4 = st.columns(2, gap="large")

with col_c3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=chart_df['date'], y=chart_df['atr10'],
        name='ATR-10', line=dict(color='#fb923c', width=2),
        hovertemplate='%{x|%d %b}<br>ATR-10: %{y:.0f} pts<extra></extra>'
    ))
    if 'atr20' in chart_df.columns:
        fig3.add_trace(go.Scatter(
            x=chart_df['date'], y=chart_df['atr20'],
            name='ATR-20', line=dict(color='#fb923c', width=1, dash='dot'),
            opacity=0.5,
            hovertemplate='%{x|%d %b}<br>ATR-20: %{y:.0f} pts<extra></extra>'
        ))
    fig3.add_hline(y=150, line_dash='dash', line_color='#4ade80',
                   opacity=0.4, annotation_text='150 = condor zone',
                   annotation_font=dict(color='#4ade80', size=10))
    fig3.update_layout(
        title=dict(text='ATR — daily range (below 150 = good for condor)', font=dict(size=13, color='#888'), x=0),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#888', family='DM Sans'),
        xaxis=dict(showgrid=False, color='#333'),
        yaxis=dict(showgrid=True, gridcolor='#1a1d27', color='#555'),
        legend=dict(orientation='h', y=-0.15, x=0, font=dict(size=11)),
        height=250, margin=dict(l=0, r=0, t=36, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_c4:
    if 'vix' in chart_df.columns:
        fig4 = go.Figure()
        fig4.add_hrect(y0=0, y1=14, fillcolor='#4ade80', opacity=0.05, line_width=0)
        fig4.add_hrect(y0=14, y1=20, fillcolor='#fbbf24', opacity=0.04, line_width=0)
        fig4.add_trace(go.Scatter(
            x=chart_df['date'], y=chart_df['vix'],
            name='India VIX', line=dict(color='#a78bfa', width=2),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.06)',
            hovertemplate='%{x|%d %b}<br>VIX: %{y:.2f}<extra></extra>'
        ))
        fig4.add_hline(y=14, line_dash='dash', line_color='#4ade80',
                       opacity=0.4, annotation_text='14 = target',
                       annotation_font=dict(color='#4ade80', size=10))
        fig4.add_hline(y=20, line_dash='dash', line_color='#f87171',
                       opacity=0.4, annotation_text='20 = danger',
                       annotation_font=dict(color='#f87171', size=10))
        fig4.update_layout(
            title=dict(text='India VIX — below 14 = ideal for condor', font=dict(size=13, color='#888'), x=0),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#888', family='DM Sans'),
            xaxis=dict(showgrid=False, color='#333'),
            yaxis=dict(showgrid=True, gridcolor='#1a1d27', color='#555'),
            showlegend=False,
            height=250, margin=dict(l=0, r=0, t=36, b=0),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── TABS: Whipsaw + Backtest + Rules ─────────────────────────────────────────
tab_ws, tab_bt, tab_rules = st.tabs([
    "⚡  Whipsaw risk — path projection",
    "📊  Backtest — no lookahead",
    "📋  Trading rules",
])

# ── TAB 1: Whipsaw chart ──────────────────────────────────────────────────────
with tab_ws:
    try:
        from whipsaw_analyser import analyse_whipsaw, build_whipsaw_data

        ws_df = build_whipsaw_data()
        if not ws_df.empty:
            today_ws = ws_df.dropna(subset=['atr10','rsi']).iloc[-1]
            ws = analyse_whipsaw(
                close   = float(today_ws['close']),
                rsi     = float(today_ws['rsi']),
                z_score = float(today_ws.get('z_score', -1)),
                trend   = int(today_ws.get('trend', -1)),
                atr10   = float(today_ws['atr10']),
            )
        else:
            ws = None
    except ImportError:
        ws = None
        st.warning("whipsaw_analyser.py not found. Place it in the same folder as dashboard.py.")

    if ws:
        st.markdown(f"""
        <div style="background:#1a1205;border-radius:12px;padding:14px 18px;
             border-left:4px solid #fbbf24;margin-bottom:1rem;">
          <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
               opacity:0.5;margin-bottom:6px">Whipsaw warning — based on {ws['n_similar']} similar historical days</div>
          <div style="font-size:0.9rem;line-height:1.8;opacity:0.85">
            In <b style="color:#fbbf24">{ws['bounce_then_fall_prob']:.0%}</b> of similar setups,
            Nifty bounced <b style="color:#4ade80">+{ws['bounce_5d_pts']} pts</b> first
            (peak ~{ws['bounce_peak_est']:,}), then continued lower.
            Your stop gets hit on the bounce, then the original thesis plays out —
            but without you.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric row
        w1_c, w2_c, w3_c, w4_c = st.columns(4)
        w1_c.metric("Bounce probability",    f"{ws['bounce_then_fall_prob']:.0%}",
                  help="% of similar days with >1.5% bounce before continued fall")
        w2_c.metric("Typical bounce",        f"+{ws['bounce_5d_pts']} pts",
                  help="Median bounce in first 5 days")
        w3_c.metric("20-day drift",          f"{ws['drift_20d_pts']:+} pts",
                  help="Median 20-day outcome from similar conditions")
        w4_c.metric("Down at 20 days",       f"{ws['down_20d_prob']:.0%}",
                  help="% of similar days where price was lower after 20 days")

        st.markdown("")

        # The main whipsaw chart
        fig_ws = go.Figure()

        days      = ws['days']
        close_val = ws['close']

        # Shaded zones
        fig_ws.add_hrect(
            y0=ws['bounce_peak_est'] * 0.995,
            y1=ws['safe_sell_strike'] * 1.005,
            fillcolor='#f87171', opacity=0.08, line_width=0,
        )

        # Bounce path (green — the whipsaw that traps you)
        fig_ws.add_trace(go.Scatter(
            x=days, y=ws['bounce_path'],
            mode='lines', name='Bounce path (whipsaw)',
            line=dict(color='#4ade80', width=2.5),
            hovertemplate='Day %{x}: %{y:,.0f}<extra>Bounce path</extra>'
        ))

        # Bear path (red — direct continuation)
        fig_ws.add_trace(go.Scatter(
            x=days, y=ws['bear_path'],
            mode='lines', name='Bear path (direct fall)',
            line=dict(color='#f87171', width=2, dash='dot'),
            hovertemplate='Day %{x}: %{y:,.0f}<extra>Bear path</extra>'
        ))

        # Structural trend line
        fig_ws.add_trace(go.Scatter(
            x=days, y=ws['trend_line'],
            mode='lines', name='20-day structural trend',
            line=dict(color='#f87171', width=1.5, dash='dash'),
            hovertemplate='Day %{x}: %{y:,.0f}<extra>Trend line</extra>'
        ))

        # Today dot
        fig_ws.add_trace(go.Scatter(
            x=[0], y=[close_val],
            mode='markers+text',
            marker=dict(color='#60a5fa', size=10),
            text=[f"  Today {close_val:,.0f}"],
            textposition='middle right',
            textfont=dict(color='#60a5fa', size=11),
            name='Today', showlegend=False,
            hovertemplate=f'Today: {close_val:,.0f}<extra></extra>'
        ))

        # Bounce peak marker
        bounce_peak_day = 5
        fig_ws.add_trace(go.Scatter(
            x=[bounce_peak_day], y=[ws['bounce_peak_est']],
            mode='markers+text',
            marker=dict(color='#4ade80', size=9, symbol='diamond'),
            text=[f"  +{ws['bounce_5d_pts']} pts<br>  ~{ws['bounce_peak_est']:,}"],
            textposition='top right',
            textfont=dict(color='#4ade80', size=10),
            name='Bounce peak', showlegend=False,
            hovertemplate=f'Bounce peak ~{ws["bounce_peak_est"]:,}<extra></extra>'
        ))

        # 20-day endpoint
        fig_ws.add_trace(go.Scatter(
            x=[20], y=[ws['bounce_path'][-1]],
            mode='markers+text',
            marker=dict(color='#fb923c', size=9),
            text=[f"  {ws['drift_20d_pts']:+} pts<br>  ~{ws['bounce_path'][-1]:,}"],
            textposition='middle right',
            textfont=dict(color='#fb923c', size=10),
            name='20d outcome', showlegend=False,
        ))

        # Safe strike line
        fig_ws.add_hline(
            y=ws['safe_sell_strike'],
            line_dash='dash', line_color='#fbbf24', line_width=1.5, opacity=0.8,
            annotation_text=f"  Safe sell strike {ws['safe_sell_strike']:,}  (above bounce)",
            annotation_position='right',
            annotation_font=dict(color='#fbbf24', size=10),
        )

        # Labels for bounce phases
        fig_ws.add_annotation(
            x=2.5, y=ws['bounce_path'][3],
            text="Models see this<br>5-day bounce UP",
            showarrow=False,
            font=dict(color='#4ade80', size=10),
            align='center',
            bgcolor='rgba(0,0,0,0.5)',
        )
        fig_ws.add_annotation(
            x=15, y=ws['trend_line'][15],
            text="Structural trend sees this<br>20-day downtrend",
            showarrow=False,
            font=dict(color='#f87171', size=10),
            align='center',
            bgcolor='rgba(0,0,0,0.5)',
        )
        fig_ws.add_annotation(
            x=3, y=close_val * 0.985,
            text="Your stop gets hit here<br>if whipsaw",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#fbbf24',
            ax=0, ay=40,
            font=dict(color='#fbbf24', size=10),
            bgcolor='rgba(40,30,0,0.8)',
            bordercolor='#fbbf24',
            borderwidth=1,
        )

        fig_ws.update_layout(
            title=dict(
                text=f'Whipsaw risk — bounce then fall  (based on {ws["n_similar"]} similar historical setups)',
                font=dict(size=13, color='#888'), x=0
            ),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#888', family='DM Sans'),
            xaxis=dict(
                showgrid=False, color='#555',
                title='Trading days from today',
                tickvals=[0,3,5,10,15,20],
                ticktext=['Today','Day 3','Day 5','Day 10','Day 15','Day 20'],
            ),
            yaxis=dict(showgrid=True, gridcolor='#1a1d27', color='#555',
                       tickformat=',.0f'),
            legend=dict(orientation='h', y=-0.15, font=dict(size=11)),
            height=420, margin=dict(l=0, r=120, t=44, b=0),
        )
        st.plotly_chart(fig_ws, use_container_width=True)

        # Safe strikes from whipsaw analysis
        st.markdown(f"""
        <div style="background:#0f1117;border-radius:10px;padding:14px 18px;
             border-left:4px solid #fbbf24;margin-top:0.5rem">
          <div style="font-size:0.68rem;opacity:0.45;letter-spacing:0.1em;
               text-transform:uppercase;margin-bottom:8px">
            Whipsaw-adjusted bear call spread strikes
          </div>
          <div style="display:flex;gap:2rem;align-items:center">
            <div>
              <div style="font-size:0.6rem;opacity:0.4;text-transform:uppercase">Sell CE above bounce</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.8rem;color:#fbbf24">
                {ws['safe_sell_strike']:,}
              </div>
            </div>
            <div style="opacity:0.3;font-size:1.2rem">→</div>
            <div>
              <div style="font-size:0.6rem;opacity:0.4;text-transform:uppercase">Buy CE (loss cap)</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.8rem;color:#fb923c">
                {ws['safe_buy_strike']:,}
              </div>
            </div>
            <div style="margin-left:auto;font-size:0.75rem;opacity:0.5;max-width:220px;line-height:1.6">
              These strikes sit above the expected bounce peak ({ws['bounce_peak_est']:,}).
              Even if a whipsaw occurs, your short strike should not be touched.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Whipsaw backtest table
        st.markdown('<div class="section-header" style="margin-top:1.5rem">Whipsaw backtest — no lookahead — by year</div>', unsafe_allow_html=True)

        ws_df2 = build_whipsaw_data()
        if not ws_df2.empty:
            ws_df2['year'] = ws_df2['date'].dt.year
            bear_cond = (
                (ws_df2['trend'] == -1) &
                (ws_df2['rsi'] > 32) & (ws_df2['rsi'] < 60) &
                (ws_df2['bounce_5d'].notna()) &
                (ws_df2['drift_20d'].notna())
            )
            bear_days_hist = ws_df2[bear_cond].copy()

            if not bear_days_hist.empty:
                bear_days_hist['whipsaw'] = (
                    (bear_days_hist['bounce_5d'] > 0.015) &
                    (bear_days_hist['drift_20d'] < -0.02)
                )
                bear_days_hist['down_20d'] = bear_days_hist['drift_20d'] < 0

                yr_ws = bear_days_hist.groupby('year').agg(
                    bear_days=('whipsaw', 'count'),
                    whipsaw_pct=('whipsaw', 'mean'),
                    down_20d_pct=('down_20d', 'mean'),
                    avg_bounce=('bounce_5d', 'mean'),
                ).reset_index()
                yr_ws['whipsaw_pct']  = (yr_ws['whipsaw_pct'] * 100).round(1)
                yr_ws['down_20d_pct'] = (yr_ws['down_20d_pct'] * 100).round(1)
                yr_ws['avg_bounce']   = (yr_ws['avg_bounce'] * 100).round(1)
                yr_ws.columns = ['Year','Bear days','Whipsaw %','Down at 20d %','Avg bounce %']

                st.dataframe(
                    yr_ws.set_index('Year'),
                    use_container_width=True,
                    height=min(35 * len(yr_ws) + 40, 400),
                )

                # Overall stats
                overall_ws  = bear_days_hist['whipsaw'].mean()
                overall_d20 = bear_days_hist['down_20d'].mean()
                o1_c, o2_c, o3_c = st.columns(3)
                o1_c.metric("Overall whipsaw rate",  f"{overall_ws:.1%}",
                          help="% of bear signal days where bounce preceded continued fall")
                o2_c.metric("Down at 20 days",       f"{overall_d20:.1%}",
                          help="% of bear signal days where price was lower after 20 days")
                o3_c.metric("Total bear signal days", str(len(bear_days_hist)))
    else:
        st.info("No similar historical patterns found. Need more data.")

# ── TAB 2: No-lookahead backtest ──────────────────────────────────────────────
with tab_bt:
    st.markdown('<div class="section-header" style="margin-top:0">Condor coverage — GREEN vs all days — no lookahead</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;opacity:0.55;margin-bottom:1rem;line-height:1.7">
    All features are lagged 1 day — the backtest never uses today's data to score today.
    Target: did next-day's actual range stay within the condor strikes?
    </div>
    """, unsafe_allow_html=True)

    bt_df = df.dropna(subset=['atr10']).copy()
    bt_df['fwd_range'] = bt_df['high'].shift(-1) - bt_df['low'].shift(-1)
    bt_df = bt_df.dropna(subset=['fwd_range'])
    bt_df['score']  = bt_df.apply(lambda r: compute_score(r)[0], axis=1)
    bt_df['regime'] = bt_df['score'].apply(classify)
    bt_df['safe']   = (bt_df['fwd_range'] <= bt_df['atr10'] * CONDOR_ATR_MULT).astype(int)
    bt_df['year']   = bt_df['date'].dt.year

    yearly_bt = bt_df.groupby(['year','regime'])['safe'].agg(['mean','count']).reset_index()
    yearly_bt.columns = ['year','regime','safe_rate','count']
    green_yr = yearly_bt[yearly_bt['regime']=='GREEN'].copy()
    all_yr   = bt_df.groupby('year')['safe'].agg(['mean','count']).reset_index()
    all_yr.columns = ['year','all_rate','all_count']

    if not green_yr.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=all_yr['year'], y=all_yr['all_rate']*100,
            name='All days', marker_color='#374151', opacity=0.75,
            hovertemplate='%{x}: %{y:.1f}%<extra>All days</extra>'
        ))
        fig_bt.add_trace(go.Bar(
            x=green_yr['year'], y=green_yr['safe_rate']*100,
            name='GREEN days only', marker_color='#4ade80', opacity=0.85,
            hovertemplate='%{x}: %{y:.1f}%<extra>GREEN days</extra>'
        ))
        fig_bt.add_hline(y=65, line_dash='dash', line_color='#fbbf24', opacity=0.6,
                         annotation_text='65% = profitable threshold',
                         annotation_font=dict(color='#fbbf24', size=10))
        fig_bt.add_hline(y=50, line_dash='dot', line_color='#6b7280', opacity=0.4,
                         annotation_text='50% = coin flip',
                         annotation_font=dict(color='#6b7280', size=10))
        fig_bt.update_layout(
            barmode='group',
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#888', family='DM Sans'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#1a1d27', range=[0, 105], ticksuffix='%'),
            legend=dict(orientation='h', y=-0.12, font=dict(size=12)),
            height=340, margin=dict(l=0, r=0, t=16, b=0),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        g_all = bt_df[bt_df['regime']=='GREEN']['safe'].mean()
        a_all = bt_df['safe'].mean()
        n_g   = (bt_df['regime']=='GREEN').sum()

        from scipy.stats import binomtest
        res  = binomtest(int(bt_df[bt_df['regime']=='GREEN']['safe'].sum()),
                         int(n_g), p=0.5, alternative='greater')
        pval = res.pvalue

        r1_c, r2_c, r3_c, r4_c, r5_c = st.columns(5)
        r1_c.metric("GREEN day coverage", f"{g_all:.1%}")
        r2_c.metric("All-day coverage",   f"{a_all:.1%}")
        r3_c.metric("Filter lift",        f"+{(g_all-a_all)*100:.1f}pp")
        r4_c.metric("GREEN days / year",  f"{n_g/max(bt_df['year'].nunique(),1):.0f}")
        r5_c.metric("p-value",            f"{pval:.4f}",
                    delta="REAL EDGE" if pval < 0.05 else "not significant",
                    delta_color="normal" if pval < 0.05 else "inverse")

        # Year-by-year table
        st.markdown('<div class="section-header" style="margin-top:1rem">Year-by-year breakdown</div>', unsafe_allow_html=True)
        yr_table = []
        for yr in sorted(bt_df['year'].unique()):
            y = bt_df[bt_df['year']==yr]
            yg = y[y['regime']=='GREEN']
            ya = y
            yr_table.append({
                'Year': yr,
                'GREEN days':   len(yg),
                'GREEN safe %': f"{yg['safe'].mean()*100:.0f}%" if len(yg)>0 else '—',
                'All safe %':   f"{ya['safe'].mean()*100:.0f}%",
                'ATR avg':      f"{y['atr10'].mean():.0f} pts",
                'Regime days':  f"G:{len(yg)} Y:{len(y[y['regime']=='YELLOW'])} R:{len(y[y['regime']=='RED'])}",
            })
        st.dataframe(pd.DataFrame(yr_table).set_index('Year'),
                     use_container_width=True, height=min(38*len(yr_table)+40, 450))

# ── TAB 3: Rules ──────────────────────────────────────────────────────────────
with tab_rules:
    st.markdown('<div class="section-header" style="margin-top:0.5rem">Rules — never break these</div>', unsafe_allow_html=True)
    rule_items = [
        ("🚫", "Never trade expiry week",           "IV distortion makes premiums unreliable"),
        ("🚫", "Never trade RBI / FOMC / Budget day","Event risk creates gap moves that breach any strike"),
        ("🔴", "Exit immediately on breach",         "If either strike is touched, exit. No holding. No hoping."),
        ("🔴", "No reversion guarantee",             "A breach is a real loss. The market does not owe you a bounce back."),
        ("🟡", "GREEN = good conditions, not profit","The filter improves odds. It does not guarantee anything."),
        ("🟡", "Half size on YELLOW",                "Reduce normal lot count by 50% when signal is YELLOW."),
        ("⚪", "Do nothing on RED",                  "Waiting is a trade. Capital preserved = capital available for GREEN."),
        ("⚪", "Whipsaw = place strike above bounce","In downtrends, set bear call sell strike above expected bounce peak."),
    ]
    for icon, title, desc in rule_items:
        st.markdown(f"""
        <div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #1a1d27;align-items:flex-start">
          <span style="font-size:16px;min-width:24px">{icon}</span>
          <div>
            <div style="font-weight:500;font-size:0.85rem;color:#f8f8f2">{title}</div>
            <div style="font-size:0.74rem;opacity:0.45;margin-top:2px">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1a1d27;
     display:flex;justify-content:space-between;font-size:0.7rem;opacity:0.3;">
  <span>Nifty Regime Engine · Data: yfinance + NSE</span>
  <span>Last computed: {date_str} · Score: {score}/100 · {regime}</span>
</div>
""", unsafe_allow_html=True)
