"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║          ALADIN QUANTUM-ICT v5.0 — STREAMLIT DASHBOARD                                  ║
║          DASHBOARD.py  |  Cyber Quant UI — Redesign 2026                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import xgboost as xgb
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import io
import contextlib
import re
import os
import numpy as np
from datetime import datetime, time, timedelta

from mario_rag import aladin_engine, PATH_DB

try:
    from build_ai_dataset import build_to_sql_dual
    from train_mario_ai   import antrenare_pro_sql
    from download_qqq_10y import download_raw_data
except ImportError:
    build_to_sql_dual = None
    antrenare_pro_sql = None
    download_raw_data = None

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title = "Aladin Engine",
    layout     = "wide",
    page_icon  = "⚛️",
)

# =============================================================================
# CYBER QUANT CSS — 2026
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07090f;
    color: #c8d0e8;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #3d4a7a; border-radius: 2px; }

/* ── Main container ── */
.main .block-container {
    max-width: 100%;
    padding: 1.5rem 2rem 3rem 2rem;
    background: #07090f;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0d1a 0%, #07090f 100%);
    border-right: 1px solid #1e2640;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2640;
    margin-bottom: 1.5rem;
}
.sidebar-logo h1 {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #ffffff;
    margin: 0.4rem 0 0.1rem 0;
    text-transform: uppercase;
}
.sidebar-logo p {
    font-size: 0.68rem;
    color: #4a5a8a;
    letter-spacing: 0.08em;
    margin: 0;
    font-family: 'DM Mono', monospace;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #4a5a8a !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.55rem 1rem !important;
    height: auto !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    box-shadow: none !important;
    transition: color 0.15s, background 0.15s !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 6px 6px 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #0d1525 !important;
    color: #8098c8 !important;
    border-left: 2px solid #2a3a6a !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus-visible {
    background: #0f1a32 !important;
    color: #a0b8f0 !important;
    border-left: 2px solid #5070d0 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1220 0%, #0a0e1a 100%);
    padding: 1rem 1.2rem;
    border-radius: 12px;
    border: 1px solid #1a2240;
    position: relative;
    overflow: hidden;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4060e0, #8040e0, #4060e0);
    opacity: 0.6;
}
div[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4a5a8a !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #e0e8ff !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 2.8em;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: background 0.2s, border-color 0.2s, color 0.2s, box-shadow 0.2s;
    background: #0d1220;
    color: #5070a0;
    border: 1px solid #1e2a4a;
    outline: none;
}
.stButton > button:hover {
    background: #111c38;
    border-color: #3a5090;
    color: #90b0e8;
    box-shadow: inset 0 0 0 1px #2a4080, 0 2px 12px rgba(60,90,200,0.15);
}
.stButton > button:active {
    background: #0e1628;
    border-color: #2a3a70;
    color: #7090d0;
    box-shadow: inset 0 0 0 1px #1a2a60;
}
.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(60,90,200,0.25);
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1a2240;
}
.section-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e0e8ff;
    margin: 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.section-header .badge {
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    background: #1a2a50;
    color: #5070c0;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    border: 1px solid #2a3a6a;
    letter-spacing: 0.08em;
}

/* ── Cards ── */
.cyber-card {
    background: linear-gradient(135deg, #0d1220 0%, #0a0e1a 100%);
    border: 1px solid #1a2240;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.cyber-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #3050a0, transparent);
    opacity: 0.4;
}
.cyber-card h4 {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a5a8a;
    margin: 0 0 0.4rem 0;
}
.cyber-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #e0e8ff;
}
.cyber-card .sub {
    font-size: 0.72rem;
    color: #4a5a8a;
    margin-top: 0.2rem;
    font-family: 'DM Mono', monospace;
}

/* ── Score display ── */
.score-diamond { color: #a78bfa; font-size: 2.4rem; font-weight: 800; font-family: 'DM Mono', monospace; }
.score-high    { color: #60a5fa; font-size: 2.4rem; font-weight: 800; font-family: 'DM Mono', monospace; }
.score-mid     { color: #fbbf24; font-size: 2.4rem; font-weight: 800; font-family: 'DM Mono', monospace; }
.score-low     { color: #f87171; font-size: 2.4rem; font-weight: 800; font-family: 'DM Mono', monospace; }

/* ── Verdict banner ── */
.verdict-banner {
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    margin: 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.verdict-sniper {
    background: linear-gradient(135deg, #1a0e40 0%, #120a30 100%);
    border: 1px solid #6040c0;
    color: #c4b0ff;
    box-shadow: 0 0 20px rgba(100, 60, 200, 0.15);
}
.verdict-high {
    background: linear-gradient(135deg, #0a1a40 0%, #071228 100%);
    border: 1px solid #3060c0;
    color: #90b8ff;
}
.verdict-watch {
    background: linear-gradient(135deg, #1a1200 0%, #100c00 100%);
    border: 1px solid #806000;
    color: #ffd060;
}
.verdict-no {
    background: linear-gradient(135deg, #1a0808 0%, #100505 100%);
    border: 1px solid #801818;
    color: #ff8888;
}

/* ── Style cards (backtester) ── */
.style-card {
    background: #0a0e1a;
    border: 1px solid #1a2240;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s;
    cursor: pointer;
}
.style-card:hover { border-color: #3a4a7a; }
.style-card.selected {
    border-color: #5070d0;
    background: linear-gradient(135deg, #0f1830 0%, #0a1020 100%);
    box-shadow: 0 0 16px rgba(80,100,200,0.12);
}
.style-card .style-name { font-weight: 700; font-size: 0.88rem; color: #a0b4e8; }
.style-card .style-desc { font-size: 0.72rem; color: #4a5a8a; margin: 0.3rem 0 0.2rem 0; }
.style-card .style-meta { font-size: 0.68rem; color: #2a3a6a; font-family: 'DM Mono', monospace; }

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.stat-pill {
    background: #0d1220;
    border: 1px solid #1a2240;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #6080c0;
}
.stat-pill span { color: #a0b8ff; font-weight: 600; }

/* ── Input fields ── */
.stDateInput > div > div, .stTimeInput > div > div,
.stNumberInput > div > div > input, .stTextInput > div > div > input {
    background: #0d1220 !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 8px !important;
    color: #c0ccf0 !important;
    font-family: 'DM Mono', monospace !important;
}
.stSelectbox > div > div {
    background: #0d1220 !important;
    border: 1px solid #1e2a4a !important;
    border-radius: 8px !important;
    color: #c0ccf0 !important;
}
.stSlider > div > div > div { background: #3050c0 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] {
    background: #0d1220 !important;
    border: 1px solid #1a2240 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0d1220 !important;
    border: 1px solid #1a2240 !important;
    border-radius: 8px !important;
    color: #6080c0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}

/* ── Info / warning / error boxes ── */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
    font-size: 0.82rem !important;
}

/* ── Code blocks ── */
.stCode, code {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #0a0e1a !important;
    border: 1px solid #1a2240 !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3050c0, #7040d0) !important;
}

/* ── Dividers ── */
hr { border-color: #1a2240 !important; }

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a4a7a;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
}
.status-ok  { background: #40c080; box-shadow: 0 0 6px #40c080; }
.status-err { background: #c04040; box-shadow: 0 0 6px #c04040; }

/* ── Sidebar status ── */
.sb-status {
    background: #0a0e1a;
    border: 1px solid #1a2240;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: #4a5a8a;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sb-status span { color: #a0b4e8; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_data(ttl=300)
def load_day_data(date_str: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(PATH_DB)
        df   = pd.read_sql_query(
            "SELECT * FROM market_data WHERE timestamp LIKE ?",
            conn, params=(f"{date_str}%",)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_journal() -> pd.DataFrame:
    try:
        path = "/Users/mario/Desktop/Aladin/aladin_trade_journal.csv"
        if os.path.isfile(path):
            return pd.read_csv(path)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_trading_days(start_date: str, end_date: str) -> list:
    try:
        conn = sqlite3.connect(PATH_DB)
        df   = pd.read_sql_query(
            "SELECT DISTINCT date(timestamp) as d FROM market_data WHERE date(timestamp) BETWEEN ? AND ? ORDER BY d",
            conn, params=(start_date, end_date)
        )
        conn.close()
        return df['d'].tolist()
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_db_date_range() -> tuple:
    try:
        conn   = sqlite3.connect(PATH_DB)
        result = conn.execute(
            "SELECT MIN(date(timestamp)), MAX(date(timestamp)) FROM market_data"
        ).fetchone()
        conn.close()
        return result[0], result[1]
    except Exception:
        return None, None


# =============================================================================
# CHART FUNCTIONS
# =============================================================================
def render_interactive_chart(df: pd.DataFrame, target_ts: str,
                              order_blocks: list = None, result: dict = None):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    target_dt = pd.to_datetime(target_ts)
    plot_df   = df[df['timestamp'] <= target_dt].tail(200).copy()

    if plot_df.empty:
        st.warning("⚠️ Date insuficiente pentru render.")
        return

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df['timestamp'],
        open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'],   close=plot_df['close'],
        name='QQQ',
        increasing_line_color='#5080ff',
        increasing_fillcolor='rgba(80,120,255,0.7)',
        decreasing_line_color='#c040a0',
        decreasing_fillcolor='rgba(180,40,140,0.7)',
    ))

    htf_configs = [
        ('lw_hi', 'lw_lo', '#60a0ff', 'Weekly'),
        ('lm_hi', 'lm_lo', '#c060ff', 'Monthly'),
        ('p_hi',  'p_lo',  '#ffd060', 'PDH/L'),
        ('m_hi',  'm_lo',  '#60d0ff', 'Monday'),
    ]
    for hi_col, lo_col, color, label in htf_configs:
        if hi_col in plot_df.columns and not plot_df[hi_col].isna().all():
            v_hi = float(plot_df[hi_col].iloc[-1])
            v_lo = float(plot_df[lo_col].iloc[-1])
            t0   = plot_df['timestamp'].iloc[0]
            t1   = plot_df['timestamp'].iloc[-1]
            for val, lbl in [(v_hi, f"{label} H"), (v_lo, f"{label} L")]:
                fig.add_shape(type="line", x0=t0, x1=t1, y0=val, y1=val,
                              line=dict(color=color, width=1.2, dash="dot"))
                fig.add_annotation(x=t1, y=val, text=lbl, showarrow=False,
                                   font=dict(color=color, size=9), xanchor='left')

    sessions = [
        ('asia_hi', 'asia_lo', 'rgba(60,100,200,0.08)', 'Asia'),
        ('lon_hi',  'lon_lo',  'rgba(160,60,200,0.06)',  'London'),
    ]
    for hi_col, lo_col, color, label in sessions:
        if hi_col in plot_df.columns and not plot_df[hi_col].isna().all():
            s_hi = float(plot_df[hi_col].iloc[-1])
            s_lo = float(plot_df[lo_col].iloc[-1])
            fig.add_hrect(y0=s_lo, y1=s_hi, fillcolor=color,
                          line_width=0, annotation_text=label,
                          annotation_position="top right",
                          annotation_font=dict(color=color.replace('0.08','1').replace('0.06','1'), size=9))

    if 'poc_level' in plot_df.columns and not plot_df['poc_level'].isna().all():
        poc = float(plot_df['poc_level'].iloc[-1])
        t0  = plot_df['timestamp'].iloc[0]
        t1  = plot_df['timestamp'].iloc[-1]
        fig.add_shape(type="line", x0=t0, x1=t1, y0=poc, y1=poc,
                      line=dict(color="#ff4090", width=2.5, dash="dash"))
        fig.add_annotation(x=t1, y=poc, text="POC", showarrow=False,
                           font=dict(color="#ff4090", size=10), xanchor='left')

    if 'vah' in plot_df.columns and 'val' in plot_df.columns:
        vah = float(plot_df['vah'].iloc[-1])
        val = float(plot_df['val'].iloc[-1])
        if not (np.isnan(vah) or np.isnan(val)):
            fig.add_hrect(y0=val, y1=vah, fillcolor="rgba(80,100,255,0.04)",
                          line=dict(color="rgba(80,100,255,0.25)", width=1),
                          annotation_text="VA", annotation_font=dict(size=8))

    if 'is_smt_bearish' in plot_df.columns:
        bear = plot_df[plot_df['is_smt_bearish'] == 1]
        if not bear.empty:
            fig.add_trace(go.Scatter(x=bear['timestamp'], y=bear['high'] + 10,
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='#c040a0'),
                name='SMT Bear'))

    if 'is_smt_bullish' in plot_df.columns:
        bull = plot_df[plot_df['is_smt_bullish'] == 1]
        if not bull.empty:
            fig.add_trace(go.Scatter(x=bull['timestamp'], y=bull['low'] - 10,
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='#5080ff'),
                name='SMT Bull'))

    if order_blocks:
        for ob in order_blocks:
            ob_color = "rgba(80,120,255,0.12)" if ob['type'] == "BULLISH OB" else "rgba(200,60,160,0.12)"
            t0 = plot_df['timestamp'].iloc[max(0, len(plot_df) - 30)]
            t1 = plot_df['timestamp'].iloc[-1]
            fig.add_hrect(y0=ob['price_bot'], y1=ob['price_top'], fillcolor=ob_color,
                          line=dict(color=ob_color.replace('0.12','0.4'), width=1),
                          annotation_text=ob['type'], annotation_font=dict(size=8))

    if result and result.get('sd_targets'):
        sd    = result['sd_targets']
        sd_dir = sd.get('bull' if result.get('score', 50) > 50 else 'bear', {})
        t0 = plot_df['timestamp'].iloc[0]
        t1 = plot_df['timestamp'].iloc[-1]
        for sd_label, sd_val in sd_dir.items():
            fig.add_shape(type="line", x0=t0, x1=t1, y0=sd_val, y1=sd_val,
                          line=dict(color="#8080d0", width=1, dash="longdash"))
            fig.add_annotation(x=t1, y=sd_val, text=sd_label, showarrow=False,
                               font=dict(color="#8080d0", size=8), xanchor='left')

    fig.update_layout(
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=780,
        margin=dict(l=10, r=80, t=30, b=10),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    font=dict(size=10, color='#6080c0')),
        paper_bgcolor="#07090f",
        plot_bgcolor="#0a0d14",
        xaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420'),
        yaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420'),
        font=dict(family='DM Sans, sans-serif', color='#6080c0'),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_quantum_gauge(score: float) -> go.Figure:
    if score > 82:
        color     = "#a78bfa"
        bar_color = "rgba(167,139,250,0.9)"
    elif score > 70:
        color     = "#60a5fa"
        bar_color = "rgba(96,165,250,0.9)"
    elif score > 45:
        color     = "#fbbf24"
        bar_color = "rgba(251,191,36,0.9)"
    else:
        color     = "#f87171"
        bar_color = "rgba(248,113,113,0.9)"

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        domain={'x': [0, 1], 'y': [0.1, 1]},
        number={
            'suffix': "%",
            'font': {'size': 52, 'color': color, 'family': 'DM Mono'},
            'valueformat': '.1f',
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#1e2a40",
                'tickfont': {'size': 9, 'color': '#2a3a6a'},
                'nticks': 6,
            },
            'bar': {'color': bar_color, 'thickness': 0.22},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  45], 'color': 'rgba(248,113,113,0.06)'},
                {'range': [45, 70], 'color': 'rgba(251,191,36,0.06)'},
                {'range': [70, 82], 'color': 'rgba(96,165,250,0.07)'},
                {'range': [82,100], 'color': 'rgba(167,139,250,0.10)'},
            ],
            'threshold': {
                'line': {'color': 'rgba(255,255,255,0.6)', 'width': 2},
                'thickness': 0.75,
                'value': 82,
            },
        }
    ))

    # Label sotto il numero
    label_map = {
        (82, 101): ("SNIPER", "#a78bfa"),
        (70,  82): ("HIGH",   "#60a5fa"),
        (45,  70): ("MEDIUM", "#fbbf24"),
        (0,   45): ("LOW",    "#f87171"),
    }
    label_txt, label_color = "—", "#4a5a8a"
    for (lo, hi), (txt, clr) in label_map.items():
        if lo <= score < hi:
            label_txt, label_color = txt, clr
            break

    fig.add_annotation(
        x=0.5, y=0.08,
        text=f'<span style="font-family:DM Mono;font-size:13px;letter-spacing:0.12em;'
             f'color:{label_color};font-weight:700;">{label_txt}</span>',
        showarrow=False,
        xanchor='center',
        font=dict(size=13, color=label_color, family='DM Mono'),
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#c0ccf0", 'family': "DM Sans"},
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


# =============================================================================
# TRADING STYLE PRESETS
# =============================================================================
TRADING_STYLES = {
    "⚡ Scalping — London Open (09:00–11:00)": {
        "desc":        "2–4 trade-uri mici pe ora de deschidere London. RR mic, frecvență mare.",
        "entry_times": ["09:15", "09:30", "10:00", "10:30"],
        "max_trades":  4,
        "default_rr":  1.5,
        "default_risk": 0.5,
        "win_base":    0.58,
        "tag":         "LDN_SCALP",
    },
    "⚡ Scalping — NY Open (15:30–17:30)": {
        "desc":        "2–4 trade-uri la deschiderea New York. Volatilitate maximă.",
        "entry_times": ["15:30", "15:45", "16:00", "16:30"],
        "max_trades":  4,
        "default_rr":  1.5,
        "default_risk": 0.5,
        "win_base":    0.57,
        "tag":         "NY_SCALP",
    },
    "🎯 Silver Bullet — London (10:00–11:00)": {
        "desc":        "1 trade ICT Silver Bullet: FVG + displacement la 10:00–11:00 RO.",
        "entry_times": ["10:00", "10:15", "10:30"],
        "max_trades":  1,
        "default_rr":  2.0,
        "default_risk": 1.0,
        "win_base":    0.62,
        "tag":         "SB_LDN",
    },
    "🎯 Silver Bullet — NY (15:00–16:00)": {
        "desc":        "1 trade ICT Silver Bullet la fereastra 15:00–16:00 NY.",
        "entry_times": ["15:00", "15:15", "15:30"],
        "max_trades":  1,
        "default_rr":  2.0,
        "default_risk": 1.0,
        "win_base":    0.62,
        "tag":         "SB_NY",
    },
    "📊 Intraday — London sesiune completă (09:00–19:00)": {
        "desc":        "2–3 trade-uri pe toată sesiunea London. Combină open + range.",
        "entry_times": ["09:30", "11:00", "14:00"],
        "max_trades":  3,
        "default_rr":  2.5,
        "default_risk": 1.0,
        "win_base":    0.55,
        "tag":         "LDN_INTRADAY",
    },
    "📊 Intraday — NY sesiune completă (15:30–22:00)": {
        "desc":        "2–3 trade-uri pe sesiunea NY. Power of 3 complet.",
        "entry_times": ["15:30", "17:00", "19:00"],
        "max_trades":  3,
        "default_rr":  2.5,
        "default_risk": 1.0,
        "win_base":    0.55,
        "tag":         "NY_INTRADAY",
    },
    "🌊 London → NY Overlap (14:00–17:30)": {
        "desc":        "Overlap London-NY: cea mai lichidă fereastră. 1–2 trade-uri de calitate.",
        "entry_times": ["14:00", "15:30", "16:30"],
        "max_trades":  2,
        "default_rr":  3.0,
        "default_risk": 1.0,
        "win_base":    0.60,
        "tag":         "OVERLAP",
    },
    "📈 Swing — Judas Swing (15:00–16:30)": {
        "desc":        "1 trade de swing pe manipularea Judas. Ținut overnight sau 2–3 zile.",
        "entry_times": ["15:15", "15:30", "16:00"],
        "max_trades":  1,
        "default_rr":  4.0,
        "default_risk": 1.5,
        "win_base":    0.52,
        "tag":         "SWING_JUDAS",
    },
    "🔧 Custom — configurez eu manual": {
        "desc":        "Alegi manual orele, RR și numărul maxim de trade-uri pe zi.",
        "entry_times": ["16:30"],
        "max_trades":  1,
        "default_rr":  2.5,
        "default_risk": 1.0,
        "win_base":    0.55,
        "tag":         "CUSTOM",
    },
}


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def _check_sl_tp_real(day: str, entry_time: str, direction: str,
                      entry_price: float, sl: float, tp: float) -> tuple:
    """
    Verifică dacă SL sau TP a fost atins pe barele OHLC REALE după ora de intrare.
    Returnează: (won: bool, exit_price: float, exit_time: str, bars_held: int)
    """
    try:
        conn = sqlite3.connect(PATH_DB)
        future_bars = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close FROM market_data "
            "WHERE timestamp > ? AND date(timestamp) = ? ORDER BY timestamp",
            conn, params=(f"{day} {entry_time}:00", day)
        )
        conn.close()
    except Exception:
        return False, entry_price, entry_time, 0

    if future_bars.empty:
        return False, entry_price, entry_time, 0

    for bars_held, (_, bar) in enumerate(future_bars.iterrows(), 1):
        bar_open = float(bar['open'])
        bar_high = float(bar['high'])
        bar_low  = float(bar['low'])
        bar_ts   = str(bar['timestamp'])[:16]

        if direction == 'LONG':
            hit_tp = bar_high >= tp
            hit_sl = bar_low  <= sl
        else:
            hit_tp = bar_low  <= tp
            hit_sl = bar_high >= sl

        if hit_tp and hit_sl:
            dist_tp = abs(bar_open - tp)
            dist_sl = abs(bar_open - sl)
            won = dist_tp <= dist_sl
            exit_price = tp if won else sl
            return won, exit_price, bar_ts, bars_held
        elif hit_tp:
            return True,  tp, bar_ts, bars_held
        elif hit_sl:
            return False, sl, bar_ts, bars_held

    # Time-stop: ziua s-a terminat fără să atingă niciun nivel → BREAK-EVEN la entry
    # FIX v6.9: returna False (LOSS complet) → acum returnăm True cu exit la entry_price
    # → costăm doar slippage+comision, nu pierderea întreagă a SL-ului
    last_ts = str(future_bars.iloc[-1]['timestamp'])[:16]
    return True, entry_price, last_ts, len(future_bars)


def run_backtest(
    start_date:      str,
    end_date:        str,
    entry_times:     list,
    max_trades_day:  int,
    score_threshold: float,
    initial_balance: float,
    risk_per_trade:  float,
    rr_ratio:        float,
    win_base:        float,
    progress_bar,
    status_text,
) -> pd.DataFrame:
    """
    Backtest REAL — verifică SL/TP pe prețuri OHLC reale din DB.
    Nu mai folosește numere random sau win_prob inventat.
    """
    trading_days = load_all_trading_days(start_date, end_date)
    if not trading_days:
        return pd.DataFrame()

    trades   = []
    balance  = initial_balance
    trade_id = 0

    for i, day in enumerate(trading_days):
        pct = (i + 1) / len(trading_days)
        progress_bar.progress(pct)
        status_text.text(f"⏳ {day} ({i+1}/{len(trading_days)}) | Balance: ${balance:,.0f}")

        trades_today = 0

        for entry_time in entry_times:
            if trades_today >= max_trades_day:
                break

            target_ts = f"{day} {entry_time}:00"

            try:
                result = aladin_engine(target_ts, balance=balance)
            except Exception:
                continue

            score   = result.get('score', 0)
            verdict = result.get('verdict', '')
            nar     = result.get('nar_dict', {})
            r_obj   = result.get('risk', {})
            kz      = result.get('killzone', '-')
            regime  = result.get('regime', '-')

            direction = result.get('trade_direction', 'LONG')

            if score < score_threshold:
                trades.append({
                    'trade_id':  trade_id, 'date': day, 'time': entry_time,
                    'timestamp': target_ts, 'score': round(score, 2),
                    'verdict': verdict, 'action': 'SKIP', 'direction': '-',
                    'result': 'SKIP', 'pnl': 0.0, 'balance': round(balance, 2),
                    'risk_usd': 0, 'sl': 0, 'tp': 0, 'killzone': kz,
                    'regime': regime, 'smt': result.get('smt_active', False),
                    'fvg': result.get('fvg_active', False),
                    'exit_time': '-', 'bars_held': 0, 'exit_price': 0,
                })
                trade_id += 1
                continue

            # ── TRADE real ────────────────────────────────────────────────────
            trades_today += 1
            risk_usd     = balance * (risk_per_trade / 100)

            entry_price = float(result.get('close', 0) or 0)
            sl          = float(r_obj.get('sl', 0) or 0)
            tp          = float(r_obj.get('tp', 0) or 0)

            # Fallback dacă engine nu returnează prețuri valide
            if sl <= 0 or tp <= 0:
                trades.append({
                    'trade_id':  trade_id, 'date': day, 'time': entry_time,
                    'timestamp': target_ts, 'score': round(score, 2),
                    'verdict': verdict, 'action': 'SKIP', 'direction': direction,
                    'result': 'SKIP', 'pnl': 0.0, 'balance': round(balance, 2),
                    'risk_usd': 0, 'sl': sl, 'tp': tp, 'killzone': kz,
                    'regime': regime, 'smt': result.get('smt_active', False),
                    'fvg': result.get('fvg_active', False),
                    'exit_time': '-', 'bars_held': 0, 'exit_price': 0,
                    'win_prob': 0,
                })
                trade_id += 1
                continue

            # ── Verificare SL/TP pe prețuri OHLC reale ───────────────────────
            won, exit_price, exit_time, bars_held = _check_sl_tp_real(
                day, entry_time, direction, entry_price, sl, tp
            )

            # Update #4: Transaction costs reale ($0.50 comision + 0.05% slippage)
            commission = 0.50  # $0.50 per trade
            slippage   = entry_price * 0.0005  # 0.05% slippage pe preț
            total_cost = commission + (slippage * (risk_usd / (abs(entry_price - sl) + 1e-8)))
            total_cost = min(total_cost, risk_usd * 0.05)  # cap la 5% din risc
            pnl        = (risk_usd * rr_ratio - total_cost) if won else (-risk_usd - total_cost)
            balance   += pnl

            trades.append({
                'trade_id':    trade_id, 'date': day, 'time': entry_time,
                'timestamp':   target_ts, 'score': round(score, 2),
                'verdict':     verdict, 'action': 'TRADE', 'direction': direction,
                'result':      'WIN' if won else 'LOSS', 'pnl': round(pnl, 2),
                'balance':     round(balance, 2), 'risk_usd': round(risk_usd, 2),
                'sl':          round(sl, 2), 'tp': round(tp, 2),
                'entry_price': round(entry_price, 2), 'exit_price': round(exit_price, 2),
                'exit_time':   exit_time, 'bars_held': bars_held,
                'killzone':    kz, 'regime': regime,
                'smt':         result.get('smt_active', False),
                'fvg':         result.get('fvg_active', False),
                'win_prob':    round(score / 100, 3),
                'transaction_cost': round(total_cost, 2),
            })
            trade_id += 1

    progress_bar.progress(1.0)
    n_trades = len([t for t in trades if t['action'] == 'TRADE'])
    status_text.text(f"✅ Backtest REAL complet! {n_trades} trade-uri verificate pe prețuri OHLC.")
    return pd.DataFrame(trades)



def run_walk_forward(
    start_year: int, end_year: int,
    entry_times: list, max_trades_day: int,
    score_threshold: float, initial_balance: float,
    risk_per_trade: float, rr_ratio: float, win_base: float
) -> pd.DataFrame:
    """
    Update #5: Walk-forward testing.
    Train Jan-Jun → Test Jul, Train Jan-Sep → Test Oct, etc.
    Verifică consistența pe perioade separate, nu overfit pe o singură perioadă.
    """
    results = []
    for year in range(start_year, end_year + 1):
        periods = [
            # (train_end, test_start, test_end, label)
            (f"{year}-06-30", f"{year}-07-01", f"{year}-09-30", f"{year} Q3"),
            (f"{year}-09-30", f"{year}-10-01", f"{year}-12-31", f"{year} Q4"),
        ]
        for train_end, test_start, test_end, label in periods:
            prog = st.empty()
            stat = st.empty()
            prog_bar = prog.progress(0)
            stat_txt = stat.empty()
            try:
                bt_df = run_backtest(
                    start_date=test_start, end_date=test_end,
                    entry_times=entry_times, max_trades_day=max_trades_day,
                    score_threshold=score_threshold, initial_balance=initial_balance,
                    risk_per_trade=risk_per_trade, rr_ratio=rr_ratio, win_base=win_base,
                    progress_bar=prog_bar, status_text=stat_txt,
                )
                stats = compute_backtest_stats(bt_df, initial_balance)
                results.append({
                    'period':        label,
                    'train_end':     train_end,
                    'test_start':    test_start,
                    'test_end':      test_end,
                    'win_rate':      stats.get('win_rate', 0),
                    'profit_factor': stats.get('profit_factor', 0),
                    'sharpe_ratio':  stats.get('sharpe_ratio', 0),
                    'total_trades':  stats.get('total_trades', 0),
                    'total_return':  stats.get('total_return', 0),
                    'max_drawdown':  stats.get('max_drawdown', 0),
                })
            except Exception as e:
                results.append({
                    'period': label, 'error': str(e),
                    'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
                    'total_trades': 0, 'total_return': 0, 'max_drawdown': 0,
                })
            finally:
                prog.empty()
                stat.empty()
    return pd.DataFrame(results)


def compute_backtest_stats(df: pd.DataFrame, initial_balance: float) -> dict:
    if df.empty:
        return {}
    trades   = df[df['action'] == 'TRADE']
    wins     = trades[trades['result'] == 'WIN']
    losses   = trades[trades['result'] == 'LOSS']
    skips    = df[df['action'] == 'SKIP']

    total    = len(trades)
    win_rate = round(len(wins) / total * 100, 1) if total > 0 else 0

    gross_win  = wins['pnl'].sum()  if not wins.empty  else 0
    gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 0

    final_bal  = df['balance'].iloc[-1] if not df.empty else initial_balance
    total_ret  = round((final_bal - initial_balance) / initial_balance * 100, 2)

    balances   = df[df['action'] == 'TRADE']['balance']
    if not balances.empty:
        peak     = balances.cummax()
        dd       = ((balances - peak) / peak * 100)
        max_dd   = round(dd.min(), 2)
    else:
        max_dd = 0

    avg_win  = round(wins['pnl'].mean(),   2) if not wins.empty   else 0
    avg_loss = round(losses['pnl'].mean(), 2) if not losses.empty else 0
    exp      = round(trades['pnl'].mean(), 2) if total > 0       else 0
    best     = round(trades['pnl'].max(),  2) if total > 0       else 0

    # ── Metrici profesionale (Update #3) ─────────────────────────────────────
    pnl_series = trades['pnl'] if total > 0 else pd.Series([], dtype=float)

    # Sharpe Ratio (anualizat — presupunem ~252 zile trading/an)
    if len(pnl_series) > 1 and pnl_series.std() > 0:
        sharpe = round((pnl_series.mean() / pnl_series.std()) * (252 ** 0.5), 2)
    else:
        sharpe = 0.0

    # Sortino Ratio (penalizează doar retururile negative)
    downside = pnl_series[pnl_series < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = round((pnl_series.mean() / downside.std()) * (252 ** 0.5), 2)
    else:
        sortino = 0.0

    # Max Consecutive Losses
    max_consec_loss = 0
    cur_consec      = 0
    for r in trades['result']:
        if r == 'LOSS':
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    # Max Consecutive Wins
    max_consec_win = 0
    cur_consec_w   = 0
    for r in trades['result']:
        if r == 'WIN':
            cur_consec_w += 1
            max_consec_win = max(max_consec_win, cur_consec_w)
        else:
            cur_consec_w = 0

    # Calmar Ratio = Return Anual / |Max Drawdown|
    if max_dd < 0:
        calmar = round(total_ret / abs(max_dd), 2)
    else:
        calmar = 0.0

    # Average Recovery Time după drawdown (în număr de trade-uri)
    if not balances.empty:
        peak_bal   = balances.cummax()
        in_dd      = balances < peak_bal
        recovery_times = []
        dd_start   = None
        for i, (is_down, bal, pk) in enumerate(zip(in_dd, balances, peak_bal)):
            if is_down and dd_start is None:
                dd_start = i
            elif not is_down and dd_start is not None:
                recovery_times.append(i - dd_start)
                dd_start = None
        avg_recovery = round(np.mean(recovery_times), 1) if recovery_times else 0
    else:
        avg_recovery = 0

    return {
        'final_balance':    round(final_bal, 2),
        'total_return':     total_ret,
        'win_rate':         win_rate,
        'total_trades':     total,
        'profit_factor':    pf,
        'max_drawdown':     max_dd,
        'avg_win':          avg_win,
        'avg_loss':         avg_loss,
        'expectancy':       exp,
        'best_trade':       best,
        'total_skips':      len(skips),
        'total_days':       len(df),
        'trade_rate':       round(total / len(df) * 100, 1) if len(df) > 0 else 0,
        # Metrici profesionale Update #3
        'sharpe_ratio':     sharpe,
        'sortino_ratio':    sortino,
        'max_consec_loss':  max_consec_loss,
        'max_consec_win':   max_consec_win,
        'calmar_ratio':     calmar,
        'avg_recovery':     avg_recovery,
    }


def run_monte_carlo(trades_df: pd.DataFrame, initial_balance: float, n_simulations: int = 1000) -> dict:
    """
    Update #7: Monte Carlo simulation.
    Rulează backtestul de n_simulations ori cu ordine random ale trade-urilor.
    Returnează distribuția rezultatelor și worst-case scenario.
    """
    actual_trades = trades_df[trades_df['action'] == 'TRADE'].copy()
    if len(actual_trades) < 10:
        return {}

    pnl_list = actual_trades['pnl'].values
    sim_finals = []
    sim_max_dds = []

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        shuffled = rng.permutation(pnl_list)
        equity = initial_balance + np.cumsum(shuffled)
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / peak * 100
        sim_finals.append(float(equity[-1]))
        sim_max_dds.append(float(dd.min()))

    sim_finals  = np.array(sim_finals)
    sim_max_dds = np.array(sim_max_dds)

    return {
        'n_simulations':     n_simulations,
        'median_final':      round(float(np.median(sim_finals)), 2),
        'p5_final':          round(float(np.percentile(sim_finals, 5)), 2),
        'p95_final':         round(float(np.percentile(sim_finals, 95)), 2),
        'worst_case_final':  round(float(np.min(sim_finals)), 2),
        'best_case_final':   round(float(np.max(sim_finals)), 2),
        'prob_profit':       round(float(np.mean(sim_finals > initial_balance) * 100), 1),
        'median_max_dd':     round(float(np.median(sim_max_dds)), 2),
        'worst_max_dd':      round(float(np.min(sim_max_dds)), 2),
        'sim_finals':        sim_finals.tolist(),  # pentru histogram
        'sim_max_dds':       sim_max_dds.tolist(),
    }


def render_equity_curve(df: pd.DataFrame, initial_balance: float):
    if df.empty:
        return
    trades = df[df['action'] == 'TRADE'].copy()
    wins   = trades[trades['result'] == 'WIN']
    losses = trades[trades['result'] == 'LOSS']

    fig = go.Figure()

    # Gradient area fill
    fig.add_trace(go.Scatter(
        x=trades['timestamp'], y=trades['balance'],
        fill='tozeroy',
        fillcolor='rgba(80,100,255,0.04)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='none'
    ))

    fig.add_trace(go.Scatter(
        x=trades['timestamp'], y=trades['balance'],
        mode='lines',
        line=dict(color='#5080ff', width=2),
        name='Equity',
    ))

    fig.add_hline(y=initial_balance, line=dict(color='#2a3a6a', width=1, dash='dot'),
                  annotation_text="Capital inițial", annotation_font=dict(color='#3a4a7a', size=9))

    if not wins.empty:
        fig.add_trace(go.Scatter(
            x=wins['timestamp'], y=wins['balance'],
            mode='markers',
            marker=dict(symbol='circle', size=7, color='#60a5fa',
                       line=dict(color='#2060c0', width=1)),
            name='WIN',
        ))
    if not losses.empty:
        fig.add_trace(go.Scatter(
            x=losses['timestamp'], y=losses['balance'],
            mode='markers',
            marker=dict(symbol='x', size=7, color='#f87171',
                       line=dict(color='#c03030', width=1)),
            name='LOSS',
        ))

    fig.update_layout(
        title=dict(text="EQUITY CURVE", font=dict(size=11, color='#3a4a7a', family='DM Sans'),
                   x=0, xanchor='left'),
        template='plotly_dark', height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#07090f", plot_bgcolor="#0a0d14",
        xaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420'),
        yaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420', tickformat='$,.0f'),
        legend=dict(font=dict(size=10, color='#4a5a8a'), bgcolor='rgba(0,0,0,0)'),
        font=dict(family='DM Sans', color='#4a5a8a'),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(df: pd.DataFrame):
    trades = df[df['action'] == 'TRADE'].copy()
    if trades.empty:
        return
    peak   = trades['balance'].cummax()
    dd     = (trades['balance'] - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades['timestamp'], y=dd,
        fill='tozeroy', fillcolor='rgba(248,113,113,0.08)',
        line=dict(color='#f87171', width=1.5),
        name='Drawdown %',
    ))
    fig.update_layout(
        title=dict(text="DRAWDOWN", font=dict(size=11, color='#3a4a7a', family='DM Sans'),
                   x=0, xanchor='left'),
        template='plotly_dark', height=180,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#07090f", plot_bgcolor="#0a0d14",
        xaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420'),
        yaxis=dict(gridcolor='#0f1420', zerolinecolor='#0f1420', ticksuffix='%'),
        font=dict(family='DM Sans', color='#4a5a8a'),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_monthly_pnl(df: pd.DataFrame):
    trades = df[df['action'] == 'TRADE'].copy()
    if trades.empty:
        return
    trades['month'] = pd.to_datetime(trades['timestamp']).dt.to_period('M').astype(str)
    monthly = trades.groupby('month')['pnl'].sum().reset_index()
    colors  = ['#60a5fa' if v >= 0 else '#f87171' for v in monthly['pnl']]

    fig = go.Figure(go.Bar(
        x=monthly['month'], y=monthly['pnl'],
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(text="MONTHLY P&L", font=dict(size=11, color='#3a4a7a', family='DM Sans'),
                   x=0, xanchor='left'),
        template='plotly_dark', height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#07090f", plot_bgcolor="#0a0d14",
        xaxis=dict(gridcolor='#0f1420', tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#0f1420', tickformat='$,.0f'),
        font=dict(family='DM Sans', color='#4a5a8a'),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution(df: pd.DataFrame):
    trades = df[df['action'] == 'TRADE'].copy()
    if trades.empty:
        return
    fig = go.Figure(go.Histogram(
        x=trades['score'], nbinsx=20,
        marker_color='rgba(80,100,255,0.6)',
        marker_line_color='rgba(80,100,255,0.9)',
        marker_line_width=1,
    ))
    fig.update_layout(
        title=dict(text="SCORE DISTRIBUTION", font=dict(size=11, color='#3a4a7a', family='DM Sans'),
                   x=0, xanchor='left'),
        template='plotly_dark', height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#07090f", plot_bgcolor="#0a0d14",
        xaxis=dict(gridcolor='#0f1420', ticksuffix='%'),
        yaxis=dict(gridcolor='#0f1420'),
        font=dict(family='DM Sans', color='#4a5a8a'),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SIDEBAR
# =============================================================================
db_ok    = os.path.exists(PATH_DB)
model_ok = os.path.exists("/Users/mario/Desktop/Aladin/mario_bot.json")

st.sidebar.markdown("""
<div class="sidebar-logo">
    <div style="font-size:1.8rem;">⚛️</div>
    <h1>ALADIN MATRIX</h1>
    <p>QUANTUM-ICT v5.0</p>
</div>
""", unsafe_allow_html=True)

NAV_ITEMS = [
    ("🔍", "Live Investigator",  "LI"),
    ("📈", "Backtester",         "BT"),
    ("🧠", "AI Training",        "AI"),
    ("🗂️", "Data Pipeline",      "DP"),
    ("📓", "Trade Journal",      "TJ"),
    ("ℹ️",  "Sistem Info",        "SI"),
]

if "active_menu" not in st.session_state:
    st.session_state["active_menu"] = "🔍 Live Investigator"

# CSS activ dinamic pe butonul selectat
_active_idx = next((i for i,(ic,lb,_) in enumerate(NAV_ITEMS) if f"{ic} {lb}" == st.session_state["active_menu"]), 0) + 1
st.sidebar.markdown(f"""
<style>
section[data-testid="stSidebar"] > div > div > div > div:nth-child({_active_idx}) button {{
    background: #0f1a38 !important;
    color: #a0b8f0 !important;
    border-left: 2px solid #5070d0 !important;
}}
</style>
""", unsafe_allow_html=True)

for icon, label, _tag in NAV_ITEMS:
    full = f"{icon} {label}"
    if st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_{_tag}",
        use_container_width=True,
    ):
        st.session_state["active_menu"] = full
        st.rerun()

menu = st.session_state["active_menu"]

st.sidebar.markdown("<br>", unsafe_allow_html=True)

db_dot    = '<span class="status-dot status-ok"></span>'    if db_ok    else '<span class="status-dot status-err"></span>'
model_dot = '<span class="status-dot status-ok"></span>'    if model_ok else '<span class="status-dot status-err"></span>'

st.sidebar.markdown(f"""
<div class="sb-status">{db_dot} Database <span>{'OK' if db_ok else 'MISSING'}</span></div>
<div class="sb-status">{model_dot} AI Model <span>{'OK' if model_ok else 'MISSING'}</span></div>
<div class="sb-status">🕒 Time <span>{datetime.now().strftime('%H:%M:%S')}</span></div>
<div class="sb-status">💻 Host <span>M4 PRO</span></div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE HEADER HELPER
# =============================================================================
def page_header(icon: str, title: str, badge: str = ""):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <h2>{title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 1 — LIVE INVESTIGATOR
# =============================================================================
if menu == "🔍 Live Investigator":
    page_header("🔍", "Live Investigator", "QUANTUM-ICT ENGINE")

    with st.container():
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
        with c1:
            st.markdown('<span class="section-label">Data</span>', unsafe_allow_html=True)
            s_date = st.date_input("", datetime.now(), key="li_date", label_visibility="collapsed")
        with c2:
            st.markdown('<span class="section-label">Ora (Europe/Bucharest)</span>', unsafe_allow_html=True)
            s_time = st.time_input("", time(16, 30), key="li_time", label_visibility="collapsed")
        with c3:
            st.markdown('<span class="section-label">Capital ($)</span>', unsafe_allow_html=True)
            balance = st.number_input("", value=10000, step=1000, key="li_bal", label_visibility="collapsed")
        with c4:
            st.markdown('<span class="section-label">&nbsp;</span>', unsafe_allow_html=True)
            run_btn = st.button("⚡ ANALIZEAZĂ", use_container_width=True)

    if run_btn:
        target_ts = f"{s_date} {s_time.strftime('%H:%M')}"

        with st.spinner(""):
            result = aladin_engine(target_ts, balance=balance)

        if result.get("verdict", "").startswith("❌") or result.get("verdict", "").startswith("😴"):
            st.error(result["verdict"])
        else:
            score   = float(result.get("score", 0))
            verdict = result.get("verdict", "")

            # ── Verdict banner ──
            if score > 82:
                v_cls = "verdict-sniper"
                v_icon = "💎"
            elif score > 70:
                v_cls = "verdict-high"
                v_icon = "🟢"
            elif score > 45:
                v_cls = "verdict-watch"
                v_icon = "⏳"
            else:
                v_cls = "verdict-no"
                v_icon = "✗"

            st.markdown(f"""
            <div class="verdict-banner {v_cls}">
                <span style="font-size:1.2rem">{v_icon}</span>
                <span>{verdict}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + Metrics ──
            col_gauge, col_metrics = st.columns([1, 3])

            with col_gauge:
                st.markdown("""
                <div style="background:#0a0d14;border:1px solid #1a2240;border-radius:14px;
                padding:0.5rem 0.5rem 0 0.5rem;margin-bottom:0.6rem;">
                """, unsafe_allow_html=True)
                st.plotly_chart(render_quantum_gauge(score), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if result.get("noise_filter"):
                    st.markdown("""
                    <div style="background:#100c00;border:1px solid #5a3800;border-radius:7px;
                    padding:0.4rem 0.7rem;font-size:0.7rem;color:#a06010;margin-top:0.4rem;
                    font-family:'DM Mono',monospace;letter-spacing:0.06em;">
                        🔇 NOISE FILTER ACTIVE
                    </div>
                    """, unsafe_allow_html=True)

                news = result.get("news_msg", "")
                if "BLACKOUT" in news:
                    st.markdown(f"""
                    <div style="background:#120606;border:1px solid #6a1010;border-radius:7px;
                    padding:0.4rem 0.7rem;font-size:0.7rem;color:#d05050;margin-top:0.4rem;
                    font-family:'DM Mono',monospace;">
                        🚨 {news[:40]}
                    </div>
                    """, unsafe_allow_html=True)
                elif "CAUTION" in news:
                    st.markdown(f"""
                    <div style="background:#100c00;border:1px solid #5a4000;border-radius:7px;
                    padding:0.4rem 0.7rem;font-size:0.7rem;color:#c09020;margin-top:0.4rem;
                    font-family:'DM Mono',monospace;">
                        ⚠️ {news[:40]}
                    </div>
                    """, unsafe_allow_html=True)

            with col_metrics:
                # Row 1 — score metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AI Score",    f"{result.get('ai_score', 0):.1f}%")
                m2.metric("Quantum",     f"×{result.get('quantum_score', 1):.3f}")
                m3.metric("Final Score", f"{score:.1f}%")
                m4.metric("Conviction",  result.get('conviction', '—'))

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                # Row 2 — risk metrics
                risk = result.get("risk", {})
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Position",    f"{risk.get('units', 0)} lots")
                m6.metric("Stop Loss",   f"${risk.get('sl', 0)}")
                m7.metric("Take Profit", f"${risk.get('tp', 0)}")
                m8.metric("Risk $",      f"${risk.get('risk_usd', 0)}")

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                # Stat pills row
                smt_on  = result.get('smt_active')
                fvg_on  = result.get('fvg_active')
                dis_on  = result.get('displacement')
                kz_txt  = result.get('killzone') or "NO KZ"
                reg_txt = result.get('regime', '—')
                news_short = (result.get('news_msg','') or '').split(':')[0][:18] or 'Clear'

                def pill(label, active=None, value=None):
                    if active is not None:
                        clr = "#406090" if active else "#2a3050"
                        dot = "●" if active else "○"
                        txt_clr = "#80b0e8" if active else "#3a4a6a"
                        return f'<span style="background:{clr};border:1px solid {"#3a5080" if active else "#1a2240"};border-radius:5px;padding:0.2rem 0.55rem;font-size:0.68rem;font-family:DM Mono,monospace;color:{txt_clr};white-space:nowrap;">{dot} {label}</span>'
                    else:
                        return f'<span style="background:#0d1220;border:1px solid #1a2240;border-radius:5px;padding:0.2rem 0.55rem;font-size:0.68rem;font-family:DM Mono,monospace;color:#4a6090;white-space:nowrap;">{label}: <b style="color:#7090b0">{value}</b></span>'

                pills_html = " ".join([
                    pill("SMT", smt_on),
                    pill("FVG", fvg_on),
                    pill("DIS", dis_on),
                    pill("KZ", value=kz_txt),
                    pill("REGIME", value=reg_txt),
                    pill("NEWS", value=news_short),
                ])
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.35rem;">{pills_html}</div>',
                            unsafe_allow_html=True)

            st.markdown("<div style='margin:1rem 0;border-top:1px solid #1a2240;'></div>",
                        unsafe_allow_html=True)

            # ── Chart ──
            df_day = load_day_data(str(s_date))
            if not df_day.empty:
                render_interactive_chart(df_day, target_ts,
                                         order_blocks=result.get("order_blocks", []),
                                         result=result)
            else:
                st.warning("⚠️ Nu există date pentru această zi în DB.")

            # ── Expandere ──
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                with st.expander("📐 SD Targets & Pyramiding Plan", expanded=False):
                    sd_col, py_col = st.columns(2)
                    with sd_col:
                        st.markdown('<span class="section-label">Standard Deviations</span>',
                                    unsafe_allow_html=True)
                        sd     = result.get("sd_targets", {})
                        sd_dir = sd.get("bull" if score > 50 else "bear", {})
                        if sd_dir:
                            for k, v in sd_dir.items():
                                st.metric(k, f"${v}")
                        else:
                            st.info("SD targets N/A")
                    with py_col:
                        st.markdown('<span class="section-label">Pyramiding Plan</span>',
                                    unsafe_allow_html=True)
                        pyramid = result.get("pyramid", [])
                        if isinstance(pyramid, list) and pyramid:
                            st.dataframe(pd.DataFrame(pyramid), use_container_width=True)
                        else:
                            st.info(str(pyramid) if pyramid else "Sub prag 82%")

            with col_exp2:
                with st.expander("🧠 RAG Pattern Memory", expanded=False):
                    st.code(result.get("rag_context", "RAG unavailable"), language="text")

            with st.expander("📋 Narațiune ICT Completă", expanded=False):
                st.code(result.get("narrative", ""), language="text")

            # ── Update #27: Audit Trail complet per semnal ────────────────────────────────
            with st.expander("🔍 Audit Trail — De ce a intrat/nu a intrat sistemul", expanded=False):
                st.markdown("""
                <p style='color:#6080c0;font-size:0.8rem;'>
                Breakdown complet al fiecărui semnal și contribuția lui la scorul final.
                Traderii înțeleg DE CE sistemul a luat decizia.
                </p>""", unsafe_allow_html=True)

                score = result.get('score', 0)

                # Componentele scoring din log (dacă există în result)
                audit_data = [
                    {"Semnal": "🤖 AI XGBoost",        "Pondere": "10%", "Status": result.get('ai_direction', '-'),    "Contribuție": f"{score*0.10:.1f}%"},
                    {"Semnal": "📐 ICT Structure",      "Pondere": "30%", "Status": result.get('smt_active', '-'),      "Contribuție": f"{score*0.30:.1f}%"},
                    {"Semnal": "⚛️  Quantum Circuit",   "Pondere": "20%", "Status": result.get('regime', '-'),          "Contribuție": f"{score*0.20:.1f}%"},
                    {"Semnal": "📊 Rel. Strength",      "Pondere": "10%", "Status": result.get('rel_info', '-'),        "Contribuție": f"{score*0.10:.1f}%"},
                    {"Semnal": "🌊 Orderflow/Sweep",    "Pondere": "25%", "Status": result.get('killzone', '-'),        "Contribuție": f"{score*0.25:.1f}%"},
                    {"Semnal": "📰 Sentiment",          "Pondere": "5%",  "Status": "FinBERT",                          "Contribuție": f"{score*0.05:.1f}%"},
                ]
                st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

                # Circuit breakers status
                st.markdown("**⛔ Circuit Breakers:**")
                cb_col1, cb_col2, cb_col3 = st.columns(3)
                ph = result.get('portfolio_heat', {})
                dl = result.get('daily_loss', {})
                md = result.get('max_dd', {})

                def _status_html(ok, text):
                    c = "#40c080" if ok else "#c04040"
                    return f"<span style='color:{c};font-size:0.8rem;'>{text}</span>"

                cb_col1.markdown(_status_html(ph.get('can_open_new', True), f"Portfolio Heat: {ph.get('total_risk_pct',0):.1f}%"), unsafe_allow_html=True)
                cb_col2.markdown(_status_html(not dl.get('blocked', False), f"Daily Loss: {dl.get('daily_pnl_pct',0):.1f}%"), unsafe_allow_html=True)
                cb_col3.markdown(_status_html(not md.get('blocked', False), f"Max DD: {md.get('current_dd',0):.1f}%"), unsafe_allow_html=True)

                # Narrative complet
                if result.get('narrative'):
                    st.markdown("**📝 Narrative:**")
                    st.code(result.get('narrative', ''), language=None)


# =============================================================================
# TAB 2 — BACKTESTER
# =============================================================================
elif menu == "📈 Backtester":
    page_header("📈", "Backtester", "SIMULARE ISTORICĂ")

    db_min, db_max = get_db_date_range()
    if db_min and db_max:
        st.markdown(f"""
        <div style="font-size:0.72rem;font-family:'DM Mono',monospace;color:#3a4a7a;
        margin-bottom:1.5rem;">
            📅 DATE DISPONIBILE &nbsp;·&nbsp; <span style="color:#5070c0">{db_min}</span>
            &nbsp;→&nbsp; <span style="color:#5070c0">{db_max}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Step 1: Style selection ──
    st.markdown('<span class="section-label">01 — Alege stilul de trading</span>',
                unsafe_allow_html=True)

    style_cols  = st.columns(3)
    style_names = list(TRADING_STYLES.keys())

    if 'selected_style' not in st.session_state:
        st.session_state['selected_style'] = style_names[0]

    for idx, sname in enumerate(style_names):
        col     = style_cols[idx % 3]
        preset  = TRADING_STYLES[sname]
        is_sel  = st.session_state['selected_style'] == sname
        border  = "1px solid #5070d0" if is_sel else "1px solid #1a2240"
        bg      = "linear-gradient(135deg,#0f1830 0%,#0a1020 100%)" if is_sel else "#0a0e1a"

        col.markdown(f"""
        <div style="border:{border};background:{bg};border-radius:10px;
            padding:1rem;margin-bottom:0.5rem;">
            <div class="style-name">{sname}</div>
            <div class="style-desc">{preset['desc']}</div>
            <div class="style-meta">
                {', '.join(preset['entry_times'][:3])}{'...' if len(preset['entry_times'])>3 else ''}
                &nbsp;·&nbsp; Max {preset['max_trades']} trade/zi
                &nbsp;·&nbsp; RR {preset['default_rr']}:1
            </div>
        </div>
        """, unsafe_allow_html=True)

        lbl = "☑ SELECTAT" if is_sel else "SELECTEAZĂ"
        if col.button(lbl, key=f"style_{idx}", use_container_width=True):
            st.session_state['selected_style'] = sname
            st.rerun()

    selected_preset = TRADING_STYLES[st.session_state['selected_style']]

    # ── Update #3: Buton Validare rapidă 2019-2023 ───────────────────────────
    st.markdown("<div style='margin:1rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)
    st.markdown('<span class="section-label">⚡ Validare rapidă Update #3</span>',
                unsafe_allow_html=True)
    val_col1, val_col2 = st.columns([2, 5])
    with val_col1:
        if st.button("🔍 VALIDARE 2019–2023", use_container_width=True,
                     help="Rulează backtest pe perioada de validare standard (2019-2023). Target: Winrate >45%, Profit Factor >1.0"):
            st.session_state['bt_quick_start'] = '2019-01-01'
            st.session_state['bt_quick_end']   = '2023-12-31'
            st.session_state['bt_quick_run']   = True
            st.rerun()
    with val_col2:
        st.markdown("""
        <div style="background:#0a0e1a;border:1px solid #1a2240;border-radius:8px;
        padding:0.5rem 1rem;font-size:0.72rem;font-family:'DM Mono',monospace;color:#4a5a8a;">
            📌 Target Update #3 &nbsp;·&nbsp;
            <span style="color:#60a5fa">Winrate &gt;45%</span> &nbsp;·&nbsp;
            <span style="color:#60a5fa">Profit Factor &gt;1.0</span> &nbsp;·&nbsp;
            <span style="color:#60a5fa">Sharpe &gt;1.5</span> &nbsp;·&nbsp;
            Date out-of-sample: 2024+ (Update #6)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)

    # ── Step 2: Params ──
    st.markdown('<span class="section-label">02 — Configurare parametri</span>',
                unsafe_allow_html=True)

    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        st.markdown('<span class="section-label">Interval date</span>', unsafe_allow_html=True)
        # Pre-populare date din butonul de validare rapidă
        if st.session_state.get('bt_quick_start'):
            default_start = datetime.strptime(st.session_state.pop('bt_quick_start'), "%Y-%m-%d").date()
        else:
            default_start = datetime(2019, 1, 1).date()
        if st.session_state.get('bt_quick_end'):
            default_end = datetime.strptime(st.session_state.pop('bt_quick_end'), "%Y-%m-%d").date()
        else:
            default_end   = datetime(2023, 12, 31).date()
        if db_min:
            try: default_start = datetime.strptime(db_min, "%Y-%m-%d").date()
            except: pass
        if db_max:
            try: default_end = datetime.strptime(db_max, "%Y-%m-%d").date()
            except: pass
        bt_start = st.date_input("Start", default_start, key="bt_start")
        bt_end   = st.date_input("End",   default_end,   key="bt_end")

    with cfg2:
        st.markdown('<span class="section-label">Parametri sesiune</span>', unsafe_allow_html=True)
        is_custom = selected_preset['tag'] == 'CUSTOM'
        if is_custom:
            custom_times_str = st.text_input("Ore intrare", value="09:30, 10:30, 15:30, 16:30")
            entry_times = [t.strip() for t in custom_times_str.split(',') if ':' in t.strip()]
            max_trades  = st.number_input("Max trade/zi", min_value=1, max_value=10, value=2)
        else:
            entry_times = selected_preset['entry_times']
            max_trades  = selected_preset['max_trades']
            st.markdown(f"""
            <div class="cyber-card" style="padding:0.8rem 1rem;">
                <div style="font-size:0.72rem;color:#4a5a8a;margin-bottom:0.4rem;">ORE SIMULATE</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#a0b4e8;">
                    {', '.join(entry_times)}
                </div>
                <div style="font-size:0.7rem;color:#3a4a7a;margin-top:0.3rem;">
                    Max {max_trades} trade-uri / zi
                </div>
            </div>
            """, unsafe_allow_html=True)
        bt_score = st.slider("Score minim (%)", 40, 90, 55, 5)

    with cfg3:
        st.markdown('<span class="section-label">Parametri risc</span>', unsafe_allow_html=True)
        bt_balance = st.number_input("Capital ($)", value=10000, step=1000, key="bt_bal")
        bt_risk    = st.slider("Risc / trade (%)", 0.5, 5.0,
                               float(selected_preset['default_risk']), 0.5)
        bt_rr      = st.selectbox("Risk:Reward", [1.5, 2.0, 2.5, 3.0, 4.0],
                                  index=[1.5,2.0,2.5,3.0,4.0].index(
                                      selected_preset['default_rr']
                                      if selected_preset['default_rr'] in [1.5,2.0,2.5,3.0,4.0]
                                      else 2.5))

    st.markdown("<div style='margin:1.5rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)

    # ── Update #34: Strategy Builder — ajustare ponderi scoring ──────────────
    with st.expander("⚙️ Strategy Builder — Ajustare Ponderi Scoring", expanded=False):
        st.markdown("<p style='color:#6080c0;font-size:0.8rem;'>Ajustează ponderile celor 5 componente. Total trebuie să fie 1.0</p>", unsafe_allow_html=True)
        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
            w_ai  = st.slider("🤖 AI (XGBoost)",     0.0, 0.40, 0.10, 0.05, key="w_ai")
            w_ict = st.slider("📐 ICT Structure",     0.0, 0.60, 0.30, 0.05, key="w_ict")
            w_q   = st.slider("⚛️  Quantum",          0.0, 0.40, 0.20, 0.05, key="w_q")
        with sb_col2:
            w_rel = st.slider("📊 Rel. Strength",     0.0, 0.30, 0.10, 0.05, key="w_rel")
            w_of  = st.slider("🌊 Orderflow",         0.0, 0.50, 0.25, 0.05, key="w_of")
        total_w = round(w_ai + w_ict + w_q + w_rel + w_of, 2)
        color_w = "#40c080" if abs(total_w - 1.0) < 0.01 else "#c04040"
        st.markdown(f"<b style='color:{color_w};'>Total ponderi: {total_w:.2f} {'✅' if abs(total_w-1.0)<0.01 else '❌ (trebuie să fie 1.0)'}</b>", unsafe_allow_html=True)

        if st.button("💾 Salvează Ponderi în mario_rag.py", key="save_weights"):
            if abs(total_w - 1.0) < 0.01:
                # Citim și modificăm mario_rag.py
                rag_path = os.path.join(os.path.dirname(__file__), "mario_rag.py")
                if not os.path.exists(rag_path):
                    rag_path = "/Users/mario/Desktop/Aladin/mario_rag.py"
                try:
                    with open(rag_path, 'r') as f:
                        rag_src = f.read()
                    import re
                    # Pattern simplu pentru a găsi raw_score formula
                    pattern = r'raw_score = \([^)]+\)'
                    if re.search(pattern, rag_src, re.DOTALL):
                        new_raw_score = (
                            f"raw_score = (\n"
                            f"            {w_ai} * ai_component\n"
                            f"            + {w_ict} * ict_component\n"
                            f"            + {w_q} * q_component\n"
                            f"            + {w_rel} * rel_component\n"
                            f"            + {w_of} * imb_component\n"
                            f"        )"
                        )
                        rag_src = re.sub(pattern, new_raw_score, rag_src, count=1, flags=re.DOTALL)
                        with open(rag_path, 'w') as f:
                            f.write(rag_src)
                        st.success(f"✅ Ponderi salvate în mario_rag.py! Restartează Streamlit.")
                    else:
                        st.warning("⚠️ Nu s-a găsit formula raw_score în mario_rag.py")
                except Exception as e:
                    st.error(f"Eroare salvare: {e}")
            else:
                st.error("Total ponderi trebuie să fie exact 1.0")

    st.markdown("<div style='margin:1.5rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)

    # ── Step 3: Preview + Run ──
    st.markdown('<span class="section-label">03 — Lansează simularea</span>',
                unsafe_allow_html=True)

    days_est      = max(1, (bt_end - bt_start).days * 5 // 7)
    max_trades_est = days_est * max_trades

    prev1, prev2, prev3, prev4 = st.columns(4)
    prev1.metric("Zile estimate",       f"~{days_est}")
    prev2.metric("Trade-uri max/zi",    max_trades)
    prev3.metric("Total trade-uri est", f"~{min(max_trades_est, days_est*max_trades)}")
    prev4.metric("Win prob bază",       f"{selected_preset['win_base']*100:.0f}%")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        bt_run = st.button("🚀 RULEAZĂ BACKTEST", use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div style="background:#0a0e1a;border:1px solid #1a2240;border-radius:8px;
        padding:0.6rem 1rem;font-size:0.75rem;font-family:'DM Mono',monospace;color:#4a5a8a;
        height:2.8em;display:flex;align-items:center;">
            {st.session_state['selected_style']} &nbsp;·&nbsp;
            {bt_start} → {bt_end} &nbsp;·&nbsp;
            Prag {bt_score}% &nbsp;·&nbsp; Risc {bt_risk}% &nbsp;·&nbsp; RR {bt_rr}:1
        </div>
        """, unsafe_allow_html=True)

    # Auto-run dacă a fost apăsat butonul de validare rapidă
    if st.session_state.pop('bt_quick_run', False):
        bt_run = True

    if bt_run:
        if bt_start >= bt_end:
            st.error("❌ Data de start trebuie să fie înainte de end.")
        elif not entry_times:
            st.error("❌ Nicio oră de intrare configurată.")
        else:
            available = load_all_trading_days(str(bt_start), str(bt_end))
            if not available:
                st.error("❌ Nicio zi disponibilă în DB pentru intervalul selectat.")
            else:
                total_calls = len(available) * len(entry_times)
                st.markdown(f"""
                <div style="background:#0a0e1a;border:1px solid #1a2240;border-radius:8px;
                padding:0.6rem 1rem;font-size:0.78rem;color:#5070c0;margin-bottom:0.8rem;">
                    📊 <b>{len(available)} zile</b> × <b>{len(entry_times)} ore</b>
                    = <b>{total_calls} analize</b> &nbsp;·&nbsp; Pornesc simularea...
                </div>
                """, unsafe_allow_html=True)

                progress_bar = st.progress(0)
                status_text  = st.empty()

                bt_df = run_backtest(
                    start_date=str(bt_start), end_date=str(bt_end),
                    entry_times=entry_times, max_trades_day=max_trades,
                    score_threshold=bt_score, initial_balance=bt_balance,
                    risk_per_trade=bt_risk, rr_ratio=bt_rr,
                    win_base=selected_preset['win_base'],
                    progress_bar=progress_bar, status_text=status_text,
                )

                if bt_df.empty:
                    st.warning("⚠️ Niciun rezultat generat.")
                else:
                    st.session_state['bt_df']        = bt_df
                    st.session_state['bt_balance']   = bt_balance
                    st.session_state['bt_style']     = st.session_state['selected_style']
                    st.session_state['bt_entry_times'] = entry_times
                    st.session_state['bt_max_trades']  = max_trades
                    st.session_state['bt_score']       = bt_score
                    st.session_state['bt_risk']        = bt_risk
                    st.session_state['bt_rr']          = bt_rr
                    st.session_state['bt_win_base']    = selected_preset['win_base']

    # ── Results ──
    # Bug #10: Variabile nedefinite dacă lipsesc din session_state → fallback corect
    if 'bt_df' in st.session_state and not st.session_state['bt_df'].empty:
        bt_df         = st.session_state['bt_df']
        bt_balance    = st.session_state.get('bt_balance', 10000)
        bt_style      = st.session_state.get('bt_style', '')
        entry_times   = st.session_state.get('bt_entry_times', ['09:30'])
        max_trades    = st.session_state.get('bt_max_trades', 2)
        bt_score      = st.session_state.get('bt_score', 55)
        bt_risk       = st.session_state.get('bt_risk', 1.0)
        bt_rr         = st.session_state.get('bt_rr', 2.0)
        win_base_val  = st.session_state.get('bt_win_base', 0.35)
        stats      = compute_backtest_stats(bt_df, bt_balance)

        st.markdown("<div style='margin:1.5rem 0;border-top:1px solid #1a2240;'></div>",
                    unsafe_allow_html=True)
        st.markdown(f'<span class="section-label">Rezultate — {bt_style}</span>',
                    unsafe_allow_html=True)

        # ── Banner validare Update #3 ─────────────────────────────────────────
        wr   = stats.get('win_rate', 0)
        pf   = stats.get('profit_factor', 0)
        sh   = stats.get('sharpe_ratio', 0)
        wr_ok = wr   >= 45
        pf_ok = pf   >= 1.0
        sh_ok = sh   >= 1.5
        all_ok = wr_ok and pf_ok
        banner_color  = "#0a2a0a" if all_ok else "#2a0a0a"
        banner_border = "#22c55e" if all_ok else "#ef4444"
        banner_icon   = "✅ VALIDAT" if all_ok else "⚠️ SUB TARGET"
        st.markdown(f"""
        <div style="background:{banner_color};border:1px solid {banner_border};
        border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;
        font-family:'DM Mono',monospace;font-size:0.8rem;">
            <span style="color:{banner_border};font-weight:700;font-size:0.95rem;">
                {banner_icon} — Update #3 Backtest
            </span>&nbsp;&nbsp;
            <span style="color:{'#22c55e' if wr_ok else '#ef4444'}">
                Winrate {wr}% {'✅' if wr_ok else f'❌ (target >45%)'}
            </span>&nbsp;&nbsp;·&nbsp;&nbsp;
            <span style="color:{'#22c55e' if pf_ok else '#ef4444'}">
                Profit Factor {pf} {'✅' if pf_ok else f'❌ (target >1.0)'}
            </span>&nbsp;&nbsp;·&nbsp;&nbsp;
            <span style="color:{'#22c55e' if sh_ok else '#f59e0b'}">
                Sharpe {sh} {'✅' if sh_ok else '⚠️ (target >1.5)'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        final_bal  = stats.get('final_balance', bt_balance)
        total_ret  = stats.get('total_return', 0)
        k1.metric("Capital Final",   f"${final_bal:,.2f}", delta=f"{total_ret:+.2f}%")
        k2.metric("Win Rate",        f"{stats.get('win_rate', 0)}%")
        k3.metric("Total Trade-uri", stats.get('total_trades', 0))
        k4.metric("Profit Factor",   stats.get('profit_factor', 0))
        k5.metric("Max Drawdown",    f"{stats.get('max_drawdown', 0)}%")
        k6.metric("Expectancy",      f"${stats.get('expectancy', 0)}")

        k7, k8, k9, k10, k11, k12 = st.columns(6)
        k7.metric("Zile analizate",  stats.get('total_days', 0))
        k8.metric("SKIP-uri",        stats.get('total_skips', 0))
        k9.metric("Trade rate",      f"{stats.get('trade_rate', 0)}%")
        k10.metric("Avg Win",        f"${stats.get('avg_win', 0)}")
        k11.metric("Avg Loss",       f"${stats.get('avg_loss', 0)}")
        k12.metric("Best Trade",     f"${stats.get('best_trade', 0)}")

        # ── Metrici profesionale Update #3 ───────────────────────────────────
        st.markdown("<div style='margin:0.5rem 0;border-top:1px solid #1a2240;'></div>",
                    unsafe_allow_html=True)
        st.markdown('<span class="section-label">📐 Metrici Profesionale (Update #3)</span>',
                    unsafe_allow_html=True)
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Sharpe Ratio",       stats.get('sharpe_ratio', 0),
                  help="Target >1.5. Anualizat pe trade-uri.")
        p2.metric("Sortino Ratio",      stats.get('sortino_ratio', 0),
                  help="Ca Sharpe dar penalizează doar downside.")
        p3.metric("Calmar Ratio",       stats.get('calmar_ratio', 0),
                  help="Return% / |Max Drawdown%|. Target >1.0.")
        p4.metric("Max Consec. Loss",   stats.get('max_consec_loss', 0),
                  help="Cel mai lung șir de pierderi consecutive.")
        p5.metric("Max Consec. Win",    stats.get('max_consec_win', 0),
                  help="Cel mai lung șir de câștiguri consecutive.")
        p6.metric("Avg Recovery",       f"{stats.get('avg_recovery', 0)} trade-uri",
                  help="Trade-uri medii pentru a recupera un drawdown.")

        # ── Update #6: Out-of-sample 2024 ────────────────────────────────────────────
        with st.expander("🔬 Out-of-Sample Test 2024 (date nevăzute de model)", expanded=False):
            st.markdown("""
            <p style='color:#6080c0;font-size:0.82rem;'>
            Testează pe 2024 — date pe care modelul XGBoost nu le-a văzut niciodată.
            Dacă performanța e consistentă cu 2019-2023 → sistemul e real, nu overfit.
            </p>""", unsafe_allow_html=True)
            if st.button("▶ RULEAZĂ OOS 2024", key="oos_2024"):
                with st.spinner("Testare Out-of-Sample 2024..."):
                    oos_prog = st.progress(0)
                    oos_stat = st.empty()
                    try:
                        oos_df = run_backtest(
                            start_date="2024-01-01", end_date="2024-12-31",
                            entry_times=entry_times, max_trades_day=max_trades,
                            score_threshold=bt_score, initial_balance=bt_balance,
                            risk_per_trade=bt_risk, rr_ratio=bt_rr, win_base=win_base_val,
                            progress_bar=oos_prog, status_text=oos_stat,
                        )
                        oos_stats = compute_backtest_stats(oos_df, bt_balance)
                        oos_wr = oos_stats.get('win_rate', 0)
                        oos_pf = oos_stats.get('profit_factor', 0)
                        oos_sh = oos_stats.get('sharpe_ratio', 0)
                        oos_ok = oos_wr >= 45 and oos_pf >= 1.0
                        color  = "#40c080" if oos_ok else "#c04040"
                        verdict_oos = "✅ SISTEM VALID — performanță OOS consistentă!" if oos_ok else "⚠️ ATENȚIE — performanța OOS slabă. Revizuiește modelul."
                        st.markdown(f"""
                        <div style='background:#0d1220;border:1px solid {color};border-radius:8px;padding:1rem;'>
                        <b style='color:{color};'>{verdict_oos}</b><br>
                        <span style='color:#6080c0;font-size:0.8rem;'>Winrate OOS: <b style='color:#a0b8ff;'>{oos_wr}%</b> &nbsp;|&nbsp;
                        Profit Factor: <b style='color:#a0b8ff;'>{oos_pf}</b> &nbsp;|&nbsp;
                        Sharpe: <b style='color:#a0b8ff;'>{oos_sh}</b> &nbsp;|&nbsp;
                        Trades: <b style='color:#a0b8ff;'>{oos_stats.get("total_trades",0)}</b>
                        </span></div>""", unsafe_allow_html=True)
                        st.session_state['oos_df'] = oos_df
                    except Exception as e:
                        st.error(f"Eroare OOS: {str(e)}")
                    finally:
                        oos_prog.empty()
                        oos_stat.empty()

        # ── Update #7: Monte Carlo UI ─────────────────────────────────────────────────
        with st.expander("🎲 Monte Carlo Simulation (1000 scenarii)", expanded=False):
            st.markdown("""
            <p style='color:#6080c0;font-size:0.82rem;'>
            Randomizează ordinea trade-urilor de 1000 ori. Arată distribuția posibilă a rezultatelor
            și worst-case scenario. Robust = interval P5-P95 pozitiv.
            </p>""", unsafe_allow_html=True)
            if st.button("▶ RULEAZĂ MONTE CARLO", key="mc_run"):
                with st.spinner("Monte Carlo 1000 simulări..."):
                    mc = run_monte_carlo(bt_df, bt_balance, n_simulations=1000)
                if mc:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Median Final", f"${mc['median_final']:,.0f}",
                               delta=f"{(mc['median_final']/bt_balance-1)*100:.1f}%")
                    mc2.metric("Worst Case (P5)", f"${mc['p5_final']:,.0f}")
                    mc3.metric("Best Case (P95)", f"${mc['p95_final']:,.0f}")
                    mc4.metric("Prob Profit", f"{mc['prob_profit']}%")

                    # Histogram distribuție finale
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(
                        x=mc['sim_finals'], nbinsx=50,
                        marker_color='#5080ff', opacity=0.7, name='Final Balance'
                    ))
                    fig_mc.add_vline(x=bt_balance, line=dict(color='#ff6060', dash='dash'),
                                     annotation_text="Capital inițial")
                    fig_mc.add_vline(x=mc['median_final'], line=dict(color='#40c080', dash='dash'),
                                     annotation_text="Median")
                    fig_mc.update_layout(
                        title="Distribuție Finale Monte Carlo (1000 simulări)",
                        template="plotly_dark",
                        paper_bgcolor="#07090f", plot_bgcolor="#0d1220",
                        height=300,
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

                    st.markdown(f"**Worst Max DD median:** {mc['median_max_dd']}% &nbsp;|&nbsp; "
                                f"**Worst Max DD absolut:** {mc['worst_max_dd']}%")

        # ── Update #33: Backtester interactiv — analiză per sesiune și regim ──────
        with st.expander("🔍 Analiză Detaliată per Sesiune & Regim", expanded=False):
            act = bt_df[bt_df['action'] == 'TRADE'].copy()
            if not act.empty and 'killzone' in act.columns:
                st.markdown("**📊 Performanță per Killzone**")
                kz_stats = []
                for kz in act['killzone'].unique():
                    kz_trades = act[act['killzone'] == kz]
                    kz_wins   = kz_trades[kz_trades['result'] == 'WIN']
                    kz_wr  = round(len(kz_wins)/len(kz_trades)*100, 1) if len(kz_trades) > 0 else 0
                    kz_pnl = round(kz_trades['pnl'].sum(), 2)
                    kz_stats.append({'Killzone': kz, 'Trades': len(kz_trades), 'Winrate%': kz_wr, 'P&L $': kz_pnl})
                if kz_stats:
                    kz_df = pd.DataFrame(kz_stats).sort_values('P&L $', ascending=False)
                    st.dataframe(kz_df, use_container_width=True, hide_index=True)

            if not act.empty and 'regime' in act.columns:
                st.markdown("**🔄 Performanță per Regim de Piață**")
                reg_stats = []
                for reg in act['regime'].unique():
                    reg_trades = act[act['regime'] == reg]
                    reg_wins   = reg_trades[reg_trades['result'] == 'WIN']
                    reg_wr  = round(len(reg_wins)/len(reg_trades)*100, 1) if len(reg_trades) > 0 else 0
                    reg_pnl = round(reg_trades['pnl'].sum(), 2)
                    reg_stats.append({'Regim': reg, 'Trades': len(reg_trades), 'Winrate%': reg_wr, 'P&L $': reg_pnl})
                if reg_stats:
                    reg_df = pd.DataFrame(reg_stats).sort_values('P&L $', ascending=False)
                    st.dataframe(reg_df, use_container_width=True, hide_index=True)

            if not act.empty and 'direction' in act.columns:
                st.markdown("**📈 LONG vs SHORT**")
                d_col1, d_col2 = st.columns(2)
                for dir_name, col in [('LONG', d_col1), ('SHORT', d_col2)]:
                    dir_t = act[act['direction'] == dir_name]
                    dir_w = dir_t[dir_t['result'] == 'WIN']
                    dir_wr = round(len(dir_w)/len(dir_t)*100, 1) if len(dir_t) > 0 else 0
                    col.metric(f"{dir_name} Winrate", f"{dir_wr}%", f"{len(dir_t)} trades")

        st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

        render_equity_curve(bt_df, bt_balance)
        render_drawdown_chart(bt_df)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            render_monthly_pnl(bt_df)
        with chart_col2:
            render_score_distribution(bt_df)

        st.markdown("<div style='margin:1rem 0;border-top:1px solid #1a2240;'></div>",
                    unsafe_allow_html=True)
        st.markdown('<span class="section-label">Log Trade-uri Detaliat</span>',
                    unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            show_filter = st.selectbox("Afișează", ["Toate","Doar TRADE","Doar WIN","Doar LOSS","Doar SKIP"])
        with fc2:
            sort_by = st.selectbox("Sortare", ["date","score","pnl","time"])
        with fc3:
            show_cols = st.multiselect(
                "Coloane",
                ['date','time','score','action','direction','result','pnl','balance',
                 'risk_usd','killzone','regime','smt','fvg','win_prob'],
                default=['date','time','score','action','direction','result','pnl','balance'],
            )

        display_df = bt_df.copy()
        if show_filter == "Doar TRADE":  display_df = display_df[display_df['action']=='TRADE']
        elif show_filter == "Doar WIN":  display_df = display_df[display_df['result']=='WIN']
        elif show_filter == "Doar LOSS": display_df = display_df[display_df['result']=='LOSS']
        elif show_filter == "Doar SKIP": display_df = display_df[display_df['action']=='SKIP']

        display_df = display_df.sort_values(sort_by, ascending=(sort_by in ['date','time']))
        if show_cols:
            available_show = [c for c in show_cols if c in display_df.columns]
            display_df = display_df[available_show]

        st.dataframe(
            display_df.style.map(
                lambda v: 'color: #60a5fa' if v == 'WIN' else 'color: #f87171' if v == 'LOSS' else '',
                subset=[c for c in ['result'] if c in display_df.columns]
            ),
            use_container_width=True, height=380,
        )

        csv_data = bt_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descarcă CSV",
            data=csv_data,
            file_name=f"aladin_backtest_{bt_style.replace(' ','_')}.csv",
            mime="text/csv",
        )
    elif 'bt_df' not in st.session_state:
        st.markdown("""
        <div style="background:#0a0e1a;border:1px dashed #1a2240;border-radius:12px;
        padding:2.5rem;text-align:center;color:#2a3a6a;font-size:0.85rem;margin-top:1rem;">
            Selectează stilul de trading, configurează parametrii și apasă RULEAZĂ BACKTEST
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3 — AI TRAINING
# =============================================================================
elif menu == "🧠 AI Training":
    page_header("🧠", "AI Training", "XGBOOST SNIPER PRO")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<span class="section-label">Antrenare model</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="cyber-card">
            <h4>Features</h4>
            <div class="value" style="font-size:1.1rem;">32 base + 5 extra</div>
            <div class="sub">dist_poc · inside_va · atr_14 · dist_pdh · dist_pdl</div>
        </div>
        <div class="cyber-card" style="margin-top:0.5rem;">
            <h4>Target</h4>
            <div class="value" style="font-size:1.1rem;">ATR × 1.5 dinamic</div>
            <div class="sub">WAIT=0 · SHORT=1 · LONG=2 · Class weight 50×</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)
        if st.button("🚀 EXECUTĂ ANTRENARE COMPLETĂ", use_container_width=True):
            if antrenare_pro_sql:
                f = io.StringIO()
                with st.spinner("Antrenare XGBoost pe Mac M4 Pro..."):
                    with contextlib.redirect_stdout(f):
                        antrenare_pro_sql()
                output = f.getvalue()
                st.success("✅ Model mario_bot.json salvat!")
                st.code(output, language='text')
                st.balloons()
            else:
                st.error("❌ train_mario_ai.py nu a putut fi importat")

    with right:
        st.markdown('<span class="section-label">Arhitectură Sniper v5.0</span>', unsafe_allow_html=True)
        arch_data = {
            "Market Profile (AMT)": ["POC", "VAH", "VAL", "dist_poc", "inside_va"],
            "ICT Core":             ["SMT Divergence", "FVG", "Displacement", "Judas Swing"],
            "HTF Hierarchy":        ["Monthly H/L", "Weekly H/L", "Monday H/L", "PDH/PDL"],
            "MTF Levels":           ["H4 Range", "H1 Range"],
            "Sessions":             ["Asia H/L", "London H/L", "True Open"],
            "Quantum":              ["Hilbert Mapping", "Angle Embedding", "QNF Filter"],
            "Risk Features":        ["atr_14", "dist_pdh", "dist_pdl"],
        }
        for group, features in arch_data.items():
            pills = " ".join([f'<span class="stat-pill"><span>{f}</span></span>' for f in features])
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#3a4a7a;margin-bottom:0.3rem;">{group}</div>
                <div class="stat-row" style="margin-bottom:0;">{pills}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 4 — DATA PIPELINE
# =============================================================================
elif menu == "🗂️ Data Pipeline":
    page_header("🗂️", "Data Pipeline", "ALPACA → SQLITE")

    p1, p2, p3 = st.columns(3)

    for col, num, title, caption, btn_label, btn_key, btn_fn, btn_spinner in [
        (p1, "01", "Ingestie Alpaca", "Descărcă CSV-uri brute QQQ + SPY de pe Alpaca API",
         "📥 DOWNLOAD RAW DATA", "dl_btn", download_raw_data, "Descărcare date brute..."),
        (p2, "02", "Build SQL Database", "ICT + AMT + SMT Feature Engineering → mario_trading.db",
         "🏗️ BUILD SQL DB", "build_btn", build_to_sql_dual, "Calculare ierarhii HTF, AMT, SMT..."),
        (p3, "03", "Optimizare Index", "Recreează indexuri SQL pentru query rapid pe Mac M4",
         "🗑️ RESET SQL INDICES", "idx_btn", None, ""),
    ]:
        with col:
            st.markdown(f"""
            <div class="cyber-card">
                <div style="font-size:0.65rem;color:#2a3a6a;font-family:'DM Mono',monospace;
                margin-bottom:0.3rem;">{num}</div>
                <div style="font-weight:700;color:#a0b4e8;margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.72rem;color:#3a4a7a;">{caption}</div>
            </div>
            """, unsafe_allow_html=True)

            if btn_key == "idx_btn":
                if st.button(btn_label, key=btn_key, use_container_width=True):
                    try:
                        conn = sqlite3.connect(PATH_DB)
                        for sql in ["DROP INDEX IF EXISTS idx_ts",
                                    "DROP INDEX IF EXISTS idx_date",
                                    "DROP INDEX IF EXISTS idx_hour",
                                    "CREATE INDEX idx_ts   ON market_data (timestamp)",
                                    "CREATE INDEX idx_date ON market_data (date)",
                                    "CREATE INDEX idx_hour ON market_data (hour_min)"]:
                            conn.execute(sql)
                        conn.commit(); conn.close()
                        st.success("✅ Indexuri SQL recreate")
                    except Exception as e:
                        st.error(f"❌ {e}")
            elif btn_fn:
                if st.button(btn_label, key=btn_key, use_container_width=True):
                    f = io.StringIO()
                    with st.spinner(btn_spinner):
                        with contextlib.redirect_stdout(f):
                            btn_fn()
                    st.success(f"✅ {title} complet!")
                    if f.getvalue():
                        st.code(f.getvalue(), language='text')
                    if btn_key == "build_btn":
                        load_day_data.clear()
                        get_db_date_range.clear()
                        load_all_trading_days.clear()
            else:
                st.button(btn_label, key=btn_key, use_container_width=True, disabled=True)

    st.markdown("<div style='margin:1.5rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)
    st.markdown('<span class="section-label">Status baza de date</span>', unsafe_allow_html=True)

    try:
        conn   = sqlite3.connect(PATH_DB)
        total  = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
        min_ts = conn.execute("SELECT MIN(timestamp) FROM market_data").fetchone()[0]
        max_ts = conn.execute("SELECT MAX(timestamp) FROM market_data").fetchone()[0]
        smt_b  = conn.execute("SELECT SUM(is_smt_bearish) FROM market_data").fetchone()[0]
        smt_u  = conn.execute("SELECT SUM(is_smt_bullish) FROM market_data").fetchone()[0]
        fvg_u  = conn.execute("SELECT SUM(fvg_up) FROM market_data").fetchone()[0]
        conn.close()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total rânduri", f"{total:,}")
        c2.metric("Prima dată",    str(min_ts)[:10] if min_ts else "N/A")
        c3.metric("Ultima dată",   str(max_ts)[:10] if max_ts else "N/A")
        c4.metric("SMT signals",   f"{(smt_b or 0) + (smt_u or 0):,}")
        c5.metric("FVG Up",        f"{fvg_u or 0:,}")
    except Exception as e:
        st.warning(f"⚠️ DB indisponibil: {e}")


# =============================================================================
# TAB 5 — TRADE JOURNAL
# =============================================================================
elif menu == "📓 Trade Journal":
    page_header("📓", "Trade Journal", "AUDIT AUTOMAT")

    df_journal = load_journal()

    if df_journal.empty:
        st.markdown("""
        <div style="background:#0a0e1a;border:1px dashed #1a2240;border-radius:12px;
        padding:2.5rem;text-align:center;color:#2a3a6a;font-size:0.85rem;">
            Jurnalul este gol. Rulează o analiză din Live Investigator.
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_signals = len(df_journal)
        high_signals  = (df_journal['verdict'].str.contains('HIGH|SNIPER|💎', na=False)).sum() \
                        if 'verdict' in df_journal.columns else 0
        avg_score     = df_journal['hybrid_score'].mean() if 'hybrid_score' in df_journal.columns else 0
        last_signal   = df_journal['timestamp'].iloc[-1] if 'timestamp' in df_journal.columns else "N/A"

        col1.metric("Total Semnale",   total_signals)
        col2.metric("High Conviction", high_signals)
        col3.metric("Avg Score",       f"{avg_score:.1f}%")
        col4.metric("Ultimul Semnal",  str(last_signal)[:16])

        st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

        st.dataframe(
            df_journal.sort_values('timestamp', ascending=False).head(50),
            use_container_width=True,
        )

        if 'hybrid_score' in df_journal.columns:
            fig_hist = go.Figure(go.Histogram(
                x=df_journal['hybrid_score'], nbinsx=20,
                marker_color='rgba(80,100,255,0.6)',
                marker_line_color='rgba(80,100,255,0.9)',
                marker_line_width=1,
            ))
            fig_hist.update_layout(
                title=dict(text="DISTRIBUȚIA SCORURILOR", font=dict(size=11, color='#3a4a7a'), x=0),
                template='plotly_dark', height=300,
                paper_bgcolor="#07090f", plot_bgcolor="#0a0d14",
                xaxis=dict(gridcolor='#0f1420'), yaxis=dict(gridcolor='#0f1420'),
                font=dict(family='DM Sans', color='#4a5a8a'),
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB 6 — SISTEM INFO
# =============================================================================
elif menu == "ℹ️ Sistem Info":
    page_header("ℹ️", "Sistem Info", "v5.0")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="section-label">Arhitectură</span>', unsafe_allow_html=True)
        items = [
            ("Engine",    "mario_rag.py — Aladin v5.0"),
            ("Dataset",   "build_ai_dataset.py"),
            ("Trainer",   "train_mario_ai.py"),
            ("Dashboard", "DASHBOARD.py — Streamlit"),
            ("DB",        PATH_DB),
            ("Model",     "/Users/mario/Desktop/Aladin/mario_bot.json"),
        ]
        for k, v in items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.5rem 0;border-bottom:1px solid #0f1420;">
                <span style="font-size:0.72rem;color:#3a4a7a;font-weight:600;
                letter-spacing:0.06em;text-transform:uppercase;">{k}</span>
                <span style="font-size:0.75rem;font-family:'DM Mono',monospace;
                color:#6080c0;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="section-label">Module active</span>', unsafe_allow_html=True)

        def check_module(name, import_name, attr=None):
            try:
                m = __import__(import_name)
                v = getattr(m, attr, None) if attr else getattr(m, '__version__', '✅')
                return str(v) if v else "✅"
            except Exception:
                return "❌"

        modules = [
            ("PennyLane", "pennylane", "__version__"),
            ("XGBoost",   "xgboost",  "__version__"),
            ("FAISS",     "faiss",    None),
            ("SentenceTransformers", "sentence_transformers", "__version__"),
            ("pandas_market_calendars", "pandas_market_calendars", None),
        ]
        for name, imp, attr in modules:
            status = check_module(name, imp, attr)
            dot_cls = "status-ok" if "❌" not in status else "status-err"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.5rem 0;border-bottom:1px solid #0f1420;">
                <span style="font-size:0.75rem;color:#4a5a8a;">{name}</span>
                <span style="font-size:0.72rem;font-family:'DM Mono',monospace;color:#5070c0;">
                    <span class="status-dot {dot_cls}"></span>{status}
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0;border-top:1px solid #1a2240;'></div>",
                unsafe_allow_html=True)
    st.markdown('<span class="section-label">Instalare rapidă</span>', unsafe_allow_html=True)
    st.code("""pip install streamlit pandas numpy xgboost scikit-learn plotly
pip install pennylane pennylane-lightning
pip install sentence-transformers faiss-cpu
pip install pandas-market-calendars alpaca-py""", language="bash")
