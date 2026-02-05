# app.py — QOB (quarters-on-book) version with:
# - Chart: X axis in MONTHS (QOB*3), Y axis in %
# - Vintage table: nicer aesthetics, PD as %
# - Fast, vectorized pipeline + optional Numba speed-up
# - Progress bar while building curves
# - Integrity checks + summary-only PDF + Excel samples

import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"

import math
import hashlib
import textwrap
import re
from io import BytesIO
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative, sequential
import plotly.io as pio

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Smart cache decorator: avoids "missing ScriptRunContext … bare mode" spam
# ──────────────────────────────────────────────────────────────────────────────
def cache_data_smart(**kwargs):
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            def passthrough(func): return func
            return passthrough
    except Exception:
        pass
    return st.cache_data(**kwargs)

# Optional Numba acceleration for per-loan cummax
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

st.set_page_config(
    page_title='Vintage Default-Rate Analytics | Corporate Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ──────────────────────────────────────────────────────────────────────────────
# Corporate Color Palette (keeping user's colors)
# ──────────────────────────────────────────────────────────────────────────────
WB_PRIMARY = "#1E3A8A"       # Primary blue - buttons, accents
WB_SECONDARY = "#1E40AF"     # Secondary blue - sidebar
WB_BG = "#FFFFFF"            # Background white
WB_TEXT = "#002244"          # Text color
WB_LIGHT = "#F8FAFC"         # Light background for cards
WB_BORDER = "#E2E8F0"        # Subtle border color
WB_ACCENT = "#3B82F6"        # Lighter accent blue
WB_SUCCESS = "#10B981"       # Success green
WB_MUTED = "#64748B"         # Muted text

# Density selector in sidebar
density_mode = st.sidebar.selectbox('Display Density', ['Comfortable', 'Compact'], index=0)

# ──────────────────────────────────────────────────────────────────────────────
# Corporate CSS Styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* ═══════════════════════════════════════════════════════════════════
           CSS VARIABLES & BASE STYLES
           ═══════════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {{
            --primary: {WB_PRIMARY};
            --secondary: {WB_SECONDARY};
            --bg: {WB_BG};
            --text: {WB_TEXT};
            --light: {WB_LIGHT};
            --border: {WB_BORDER};
            --accent: {WB_ACCENT};
            --success: {WB_SUCCESS};
            --muted: {WB_MUTED};
            --space-1: 8px;
            --space-2: 16px;
            --space-3: 24px;
            --space-4: 32px;
            --space-5: 48px;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           GLOBAL STYLES
           ═══════════════════════════════════════════════════════════════════ */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: linear-gradient(135deg, {WB_BG} 0%, {WB_LIGHT} 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}

        [data-testid="stAppViewContainer"] {{
            padding: var(--space-4);
        }}

        *, *::placeholder {{
            color: var(--text) !important;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           SIDEBAR STYLES - Light background with dark text for contrast
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {{
            background: {WB_LIGHT};
            padding: 0;
            box-shadow: var(--shadow-lg);
            border-right: 1px solid {WB_BORDER};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding: var(--space-3);
        }}

        [data-testid="stSidebar"] * {{
            color: {WB_TEXT} !important;
        }}

        [data-testid="stSidebar"] .stSelectbox *,
        [data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: {WB_TEXT} !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {WB_PRIMARY} !important;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}

        [data-testid="stSidebar"] .stMarkdown p {{
            color: {WB_TEXT} !important;
            font-size: 0.9rem;
        }}

        /* Sidebar divider */
        [data-testid="stSidebar"] hr {{
            border: none;
            height: 1px;
            background: {WB_BORDER};
            margin: var(--space-3) 0;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           MAIN CONTENT AREA
           ═══════════════════════════════════════════════════════════════════ */
        .block-container {{
            padding: var(--space-4) var(--space-5);
            max-width: 1400px;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           HEADERS & TYPOGRAPHY
           ═══════════════════════════════════════════════════════════════════ */
        h1 {{
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text) !important;
        }}

        h2 {{
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--text) !important;
            border-bottom: 2px solid var(--border);
            padding-bottom: var(--space-2);
            margin-bottom: var(--space-3);
        }}

        h3 {{
            font-weight: 600;
            color: var(--primary) !important;
            letter-spacing: -0.01em;
        }}

        p {{
            max-width: 75ch;
            color: var(--text);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           BUTTONS
           ═══════════════════════════════════════════════════════════════════ */
        .stButton > button {{
            background: linear-gradient(135deg, {WB_PRIMARY} 0%, {WB_SECONDARY} 100%);
            color: white !important;
            border: none;
            border-radius: var(--radius-md);
            padding: 12px 24px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            box-shadow: var(--shadow-md);
            transition: var(--transition);
            margin: var(--space-2) 0;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            filter: brightness(1.05);
        }}

        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: var(--shadow-sm);
        }}

        .stButton > button:focus {{
            outline: 3px solid {WB_ACCENT};
            outline-offset: 2px;
        }}

        .stButton > button, .stButton > button * {{
            color: white !important;
        }}

        /* Secondary/Download buttons */
        .stDownloadButton > button {{
            background: var(--light);
            color: var(--primary) !important;
            border: 2px solid var(--border);
            border-radius: var(--radius-md);
            padding: 10px 20px;
            font-weight: 500;
            transition: var(--transition);
        }}

        .stDownloadButton > button:hover {{
            background: var(--primary);
            color: white !important;
            border-color: var(--primary);
        }}

        .stDownloadButton > button * {{
            transition: var(--transition);
        }}

        .stDownloadButton > button:hover * {{
            color: white !important;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           FORM ELEMENTS
           ═══════════════════════════════════════════════════════════════════ */
        /* Text inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            border: 2px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 10px 14px;
            font-size: 0.95rem;
            transition: var(--transition);
            background: white;
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1);
        }}

        /* Select boxes */
        [data-baseweb="select"] {{
            border-radius: var(--radius-sm) !important;
        }}

        [data-baseweb="select"] > div {{
            border: 2px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            background: white !important;
            transition: var(--transition);
        }}

        [data-baseweb="select"] > div:hover {{
            border-color: var(--primary) !important;
        }}

        /* Checkboxes and radios */
        input[type="checkbox"], input[type="radio"] {{
            accent-color: var(--primary);
            width: 18px;
            height: 18px;
        }}

        input[type="range"] {{
            accent-color: var(--primary);
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed var(--border);
            border-radius: var(--radius-lg);
            padding: var(--space-3);
            background: var(--light);
            transition: var(--transition);
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: var(--primary);
            background: white;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           TABS
           ═══════════════════════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {{
            background: var(--light);
            border-radius: var(--radius-lg);
            padding: 6px;
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: var(--radius-md);
            padding: 12px 24px;
            font-weight: 500;
            color: var(--muted) !important;
            transition: var(--transition);
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: white;
            color: var(--text) !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: white !important;
            color: var(--primary) !important;
            box-shadow: var(--shadow-sm);
        }}

        .stTabs [data-baseweb="tab-highlight"] {{
            display: none;
        }}

        .stTabs [data-baseweb="tab-border"] {{
            display: none;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           DATA TABLES
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        [data-testid="stDataFrame"] table {{
            font-size: 0.9rem;
        }}

        [data-testid="stDataFrame"] th {{
            background: var(--light) !important;
            color: var(--text) !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            padding: 14px 16px !important;
        }}

        [data-testid="stDataFrame"] td {{
            padding: 12px 16px !important;
            border-bottom: 1px solid var(--border);
        }}

        [data-testid="stDataFrame"] tr:hover td {{
            background: var(--light) !important;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           CARDS & CONTAINERS
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {{
            background: white;
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }}

        [data-testid="stExpander"] summary {{
            font-weight: 600;
            padding: var(--space-2) var(--space-3);
        }}

        /* Status container */
        [data-testid="stStatusWidget"] {{
            background: white;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           ALERTS & MESSAGES
           ═══════════════════════════════════════════════════════════════════ */
        .stAlert {{
            border-radius: var(--radius-md);
            border: none;
            box-shadow: var(--shadow-sm);
        }}

        [data-testid="stAlert"] {{
            padding: var(--space-2) var(--space-3);
        }}

        /* Success message */
        .stSuccess {{
            background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
            border-left: 4px solid {WB_SUCCESS};
        }}

        /* Info message */
        .stInfo {{
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
            border-left: 4px solid var(--primary);
        }}

        /* Warning message */
        .stWarning {{
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border-left: 4px solid #F59E0B;
        }}

        /* Error message */
        .stError {{
            background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
            border-left: 4px solid #EF4444;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           TAGS & BADGES
           ═══════════════════════════════════════════════════════════════════ */
        [data-baseweb="tag"] {{
            background: linear-gradient(135deg, {WB_PRIMARY} 0%, {WB_SECONDARY} 100%);
            border-radius: var(--radius-sm);
            font-weight: 500;
            padding: 4px 12px;
        }}

        [data-baseweb="tag"], [data-baseweb="tag"] * {{
            color: white !important;
        }}

        [data-baseweb="tag"] [data-baseweb="close"] svg {{
            fill: white !important;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           LINKS
           ═══════════════════════════════════════════════════════════════════ */
        a {{
            color: var(--primary) !important;
            text-decoration: none;
            font-weight: 500;
            transition: var(--transition);
        }}

        a:hover {{
            color: var(--accent) !important;
            text-decoration: underline;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           DIVIDERS
           ═══════════════════════════════════════════════════════════════════ */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            margin: var(--space-4) 0;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           PROGRESS BAR
           ═══════════════════════════════════════════════════════════════════ */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {WB_PRIMARY} 0%, {WB_ACCENT} 100%);
            border-radius: var(--radius-sm);
        }}

        .stProgress > div > div {{
            background: var(--light);
            border-radius: var(--radius-sm);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           METRIC CARDS
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {{
            background: white;
            padding: var(--space-3);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }}

        [data-testid="stMetric"] label {{
            color: var(--muted) !important;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--text) !important;
            font-weight: 700;
            font-size: 1.8rem;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           PLOTLY CHARTS
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stPlotlyChart"] {{
            background: white;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            padding: var(--space-2);
            border: 1px solid var(--border);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           JSON DISPLAY
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stJson"] {{
            background: var(--light);
            border-radius: var(--radius-md);
            padding: var(--space-2);
            border: 1px solid var(--border);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           CAPTION TEXT
           ═══════════════════════════════════════════════════════════════════ */
        .stCaption {{
            color: var(--muted) !important;
            font-size: 0.85rem;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           MULTISELECT
           ═══════════════════════════════════════════════════════════════════ */
        [data-baseweb="multi-select"] {{
            border-radius: var(--radius-sm);
        }}

        [data-baseweb="multi-select"] > div {{
            border: 2px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            min-height: 42px;
        }}

        [data-baseweb="multi-select"] > div:hover {{
            border-color: var(--primary) !important;
        }}

        /* ═══════════════════════════════════════════════════════════════════
           SLIDER
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
            background: var(--light) !important;
        }}

        [data-testid="stSlider"] [role="slider"] {{
            background: var(--primary) !important;
            border: 3px solid white !important;
            box-shadow: var(--shadow-md);
        }}

        /* ═══════════════════════════════════════════════════════════════════
           COLOR PICKER
           ═══════════════════════════════════════════════════════════════════ */
        [data-testid="stColorPicker"] > div {{
            border-radius: var(--radius-sm);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

    </style>
    """,
    unsafe_allow_html=True,
)

# Compact density adjustments
if density_mode == 'Compact':
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                padding: var(--space-2);
            }
            .stButton > button {
                padding: 8px 16px;
                margin: var(--space-1) 0;
            }
            .block-container {
                padding: var(--space-2) var(--space-3);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CORPORATE HEADER - Light background with dark text for contrast
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background: white;
        padding: 32px 40px;
        border-radius: 16px;
        margin-bottom: 32px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid {WB_BORDER};
        border-left: 5px solid {WB_PRIMARY};
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
            <div>
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
                    <div style="
                        background: {WB_PRIMARY};
                        border-radius: 12px;
                        padding: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M3 3v18h18"/>
                            <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                        </svg>
                    </div>
                    <h1 style="color: {WB_TEXT} !important; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">
                        Vintage Default-Rate Analytics
                    </h1>
                </div>
                <p style="color: {WB_MUTED} !important; margin: 0; font-size: 1rem; max-width: 500px;">
                    Enterprise-grade loan portfolio analysis with integrity validation and performance tracking
                </p>
            </div>
            <div style="display: flex; gap: 24px;">
                <div style="text-align: center;">
                    <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Platform</div>
                    <div style="color: {WB_PRIMARY}; font-weight: 600; font-size: 1rem;">QOB Analysis</div>
                </div>
                <div style="width: 1px; background: {WB_BORDER};"></div>
                <div style="text-align: center;">
                    <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">Version</div>
                    <div style="color: {WB_PRIMARY}; font-weight: 600; font-size: 1rem;">2.0</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

MAX_MB = 50
RESERVED_COLS = {
    'Loan ID','Origination date','Maturity date','Observation date',
    'Days past due','Origination amount','Current amount',
    'Vintage','MOB','QOB','is_def','is_def_cum'
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: schema, typing, ageing
# ──────────────────────────────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    key = {str(c).strip().lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            c = key.get(n.lower())
            if c is not None:
                return c
        return None
    col_loan = find('loan id','loan_id','loan number','loan no','id')
    col_dpd  = find('days past due','dpd','days_past_due')
    col_orig = find('origination date','origination_date','orig date','orig_date')
    col_obs  = find('observation date','observation_date','obs date','obs_date')
    col_mat  = find('maturity date','maturity_date','mat date','mat_date')
    col_orig_amt = find('origination amount','origination_amount','original amount','principal','orig amt','orig_amt')
    col_curr_amt = find('current amount','current_amount','balance','outstanding','current bal','current_bal')
    missing = []
    if not col_loan:     missing.append('Loan ID')
    if not col_dpd:      missing.append('Days past due')
    if not col_orig:     missing.append('Origination date')
    if not col_obs:      missing.append('Observation date')
    if not col_mat:      missing.append('Maturity date')
    if not col_orig_amt: missing.append('Origination amount')
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")
    rename_map = {
        col_loan:     'Loan ID',
        col_dpd:      'Days past due',
        col_orig:     'Origination date',
        col_obs:      'Observation date',
        col_mat:      'Maturity date',
        col_orig_amt: 'Origination amount',
    }
    if col_curr_amt:
        rename_map[col_curr_amt] = 'Current amount'
    df = df.rename(columns=rename_map)
    if 'Current amount' not in df.columns:
        df['Current amount'] = np.nan
    return df

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['Origination date','Observation date','Maturity date','Days past due',
              'Origination amount','Current amount']:
        df[f'__orig_{c}'] = df[c]
    for c in ['Origination date','Observation date','Maturity date']:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for c in ['Days past due','Origination amount','Current amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def add_vintage_mob(df: pd.DataFrame) -> pd.DataFrame:
    """Add quarterly Vintage (YYYYQx) and monthly MOB (keep for other calcs)."""
    df = df.copy()
    df['Vintage'] = df['Origination date'].dt.to_period('Q').astype(str)
    mob = ((df['Observation date'].dt.year - df['Origination date'].dt.year) * 12
          + (df['Observation date'].dt.month - df['Origination date'].dt.month)) + 1
    df['MOB'] = mob
    df = df[df['MOB'] > 0]
    return df

def add_qob(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter-on-book (QOB) as small int: stable quarterly ageing."""
    df = df.copy()
    oy = df['Origination date'].dt.year
    oq = df['Origination date'].dt.quarter
    sy = df['Observation date'].dt.year
    sq = df['Observation date'].dt.quarter
    o_code = oy * 4 + (oq - 1)
    s_code = sy * 4 + (sq - 1)
    qob = (s_code - o_code + 1)
    df['QOB'] = pd.to_numeric(qob, errors='coerce')
    df = df[df['QOB'].gt(0).fillna(False)]
    df['QOB'] = df['QOB'].astype('int16', copy=False)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────────────────
def _df_key(df: pd.DataFrame) -> str:
    cols = [c for c in ['Loan ID','Origination date','Observation date','Maturity date',
                        'Days past due','Origination amount','Current amount'] if c in df.columns]
    if not cols:
        cols = list(df.columns)
    parts = [df[cols].head(100000), df[cols].tail(1000)]
    sample = pd.concat(parts).drop_duplicates()
    h = pd.util.hash_pandas_object(sample, index=True).values
    m = hashlib.blake2b(digest_size=16)
    m.update(h.tobytes()); m.update(str(df.shape).encode()); m.update(",".join(map(str, sample.columns)).encode())
    return m.hexdigest()

@cache_data_smart(show_spinner=False, hash_funcs={pd.DataFrame: _df_key})
def prepare_base_cached(raw_df: pd.DataFrame) -> pd.DataFrame:
    dfn = normalize_columns(raw_df)
    dfn = ensure_types(dfn)
    dfn = add_vintage_mob(dfn)
    dfn['Vintage'] = dfn['Vintage'].astype('category')
    dfn['Loan ID'] = dfn['Loan ID'].astype('category')
    for c in ['Days past due','Origination amount','Current amount']:
        if c in dfn.columns:
            dfn[c] = dfn[c].astype('float32')
    dfn = dfn.drop_duplicates(subset=['Loan ID','Observation date'], keep='last')
    dfn = dfn.sort_values(['Loan ID','Observation date'], kind='mergesort')
    return dfn

@cache_data_smart(show_spinner=False)
def load_full(file_bytes: bytes, sheet: str, header: int):
    bio = BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet, header=header, engine='openpyxl')

# ──────────────────────────────────────────────────────────────────────────────
# Fast per-loan cumulative OR (Numba or pandas fallback)
# ──────────────────────────────────────────────────────────────────────────────
if NUMBA_OK:
    from numba import njit
    @njit(cache=True, fastmath=True)
    def _cum_or_by_group(codes: np.ndarray, flags: np.ndarray) -> np.ndarray:
        n = flags.size
        out = np.empty(n, np.uint8)
        if n == 0: return out
        prev = codes[0]; acc = flags[0] != 0; out[0] = 1 if acc else 0
        for i in range(1, n):
            c = codes[i]
            if c != prev:
                prev = c; acc = flags[i] != 0
            else:
                if flags[i] != 0: acc = True
            out[i] = 1 if acc else 0
        return out
else:
    def _cum_or_by_group(codes: np.ndarray, flags: np.ndarray) -> np.ndarray:
        s_codes = pd.Series(codes, copy=False)
        s_flags = pd.Series(flags, copy=False)
        return (s_flags.groupby(s_codes, sort=False).cummax().astype(np.uint8).to_numpy(copy=False))

# ──────────────────────────────────────────────────────────────────────────────
# Progress helper
# ──────────────────────────────────────────────────────────────────────────────
def mk_progress_updater(bar, steps: int = 5) -> Callable[[str], None]:
    ctr = {"i": 0, "steps": max(1, int(steps))}
    def _update(msg: str):
        ctr["i"] += 1
        bar.progress(min(ctr["i"]/ctr["steps"], 1.0), text=msg)
    return _update

# ──────────────────────────────────────────────────────────────────────────────
# Chart pipeline (supports QOB and MOB, smoothing toggles, cure-adjusted mode)
# ──────────────────────────────────────────────────────────────────────────────
def build_chart_data_fast(raw_df: pd.DataFrame, dpd_threshold: int,
                          max_periods: int = 20,
                          granularity: str = "QOB",
                          smooth: bool = True,
                          force_monotone: bool = True,
                          cure_adjusted: bool = False,
                          exclude_indices: Optional[set] = None,
                          prog: Optional[Callable[[str], None]] = None) -> tuple[pd.DataFrame, dict]:
    if prog: prog("Preparing base dataset …")
    base = prepare_base_cached(raw_df)

    if exclude_indices:
        base = base.loc[~base.index.isin(exclude_indices)]

    period_col = granularity.upper()
    if period_col == "QOB":
        base = add_qob(base)
    else:
        period_col = "MOB"

    if prog: prog("Computing default flags …")
    flags = (base['Days past due'].to_numpy(np.float32, copy=False) >= dpd_threshold).astype(np.uint8, copy=False)

    if cure_adjusted:
        is_def = flags
    else:
        codes = base['Loan ID'].cat.codes.to_numpy(np.int64, copy=False)
        if prog: prog("Applying per-loan cummax …")
        is_def = _cum_or_by_group(codes, flags)

    if prog: prog(f"Aggregating cohorts ({period_col}) …")
    agg = (pd.DataFrame({
                'Vintage': base['Vintage'].to_numpy(),
                period_col: base[period_col].to_numpy(),
                'is_def': is_def,
                'LoanID': base['Loan ID'].to_numpy()
           })
           .groupby(['Vintage', period_col], sort=False)
           .agg(total_loans=('LoanID', 'nunique'),
                total_default=('is_def', 'sum'))
           .reset_index())

    if agg.empty:
        if prog: prog("No data to plot.")
        return pd.DataFrame(), {}

    agg[period_col] = pd.to_numeric(agg[period_col], errors='coerce').astype('int16')
    agg['default_rate'] = (agg['total_default'] / agg['total_loans']).astype('float32')
    max_p = min(int(agg[period_col].max()), max_periods)

    # Cohort sizes for legend labels
    cohort_sizes = (agg.groupby('Vintage')['total_loans'].max()
                       .to_dict())

    wide = (agg.pivot(index=period_col, columns='Vintage', values='default_rate')
               .sort_index()
               .reindex(range(1, max_p + 1)))
    if smooth:
        wide = wide.rolling(2, 1, center=True).mean()
    if force_monotone:
        wide = wide.cummax()
    wide = wide.astype('float32')
    wide.index.name = period_col
    return wide, cohort_sizes

# ──────────────────────────────────────────────────────────────────────────────
# Plotting: % y-axis, months x-axis (QOB*3), optional legend
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves_percent_with_months(df_wide: pd.DataFrame,
                                    title: str,
                                    show_legend: bool = True,
                                    legend_limit: int = 40,
                                    palette: str = "Gradient",
                                    base_color: Optional[str] = None,
                                    line_width: int = 1,
                                    cohort_sizes: Optional[dict] = None):
    if df_wide.empty:
        st.info('Not enough data to plot.')
        return None

    idx = df_wide.index.to_numpy()
    name = (df_wide.index.name or "").upper()
    x_months = idx * 3 if name == "QOB" else idx  # MOB already months

    Y = df_wide.to_numpy(dtype='float32')
    M, N = Y.shape
    target_points = 200_000
    step = max(1, int(np.ceil((M * N) / target_points)))
    if step > 1:
        x_months = x_months[::step]
        Y = Y[::step, :]

    def _generate_palette(base: str, n: int) -> list[str]:
        r, g, b = hex_to_rgb(base)
        palette = []
        for i in range(n):
            ratio = 0.2 + 0.8 * (i / max(n - 1, 1))
            ri = int(r + (255 - r) * ratio)
            gi = int(g + (255 - g) * ratio)
            bi = int(b + (255 - b) * ratio)
            palette.append(f'rgb({ri},{gi},{bi})')
        return palette

    palette_choice = palette.lower()
    base_color = base_color or st.get_option("theme.primaryColor") or "#1f77b4"
    if palette_choice == "gradient":
        palette_colors = _generate_palette(base_color, N)
    elif palette_choice == "plotly":
        palette_colors = qualitative.Plotly
    elif palette_choice == "viridis":
        palette_colors = sequential.Viridis
    else:
        palette_colors = qualitative.Plotly
    if len(palette_colors) < N:
        times = int(np.ceil(N / len(palette_colors)))
        palette_colors = (palette_colors * times)[:N]
    else:
        palette_colors = palette_colors[:N]

    fig = go.Figure()
    show_leg = show_legend and (df_wide.shape[1] <= legend_limit)
    for i, col in enumerate(df_wide.columns):
        label = str(col)
        if cohort_sizes and col in cohort_sizes:
            label = f"{col} (n={int(cohort_sizes[col]):,})"
        fig.add_trace(go.Scatter(
            x=x_months,
            y=Y[:, i],
            mode='lines',
            name=label,
            line=dict(color=palette_colors[i], width=line_width),
            hovertemplate=f"Vintage: {col}<br>Month: %{{x}}<br>Default rate: %{{y:.2%}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#002244', family='Inter, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='Deal Age (months)', font=dict(size=13, color='#64748B')),
            tickfont=dict(size=11, color='#64748B'),
            gridcolor='#E2E8F0',
            linecolor='#E2E8F0',
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Cumulative Default Rate', font=dict(size=13, color='#64748B')),
            tickformat='.2%',
            tickfont=dict(size=11, color='#64748B'),
            gridcolor='#E2E8F0',
            linecolor='#E2E8F0',
            zeroline=False,
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#E2E8F0',
            font=dict(size=12, color='#002244', family='Inter, sans-serif')
        ),
        showlegend=show_leg,
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#E2E8F0',
            borderwidth=1,
            font=dict(color='#002244', size=11, family='Inter, sans-serif'),
            title=dict(text='Vintage Cohorts', font=dict(size=12, color='#64748B')),
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=30, t=60, b=50),
        font=dict(family='Inter, sans-serif'),
    )

    return fig
# ──────────────────────────────────────────────────────────────────────────────
# Integrity checks (vectorized)
# ──────────────────────────────────────────────────────────────────────────────
def _years_list(ser: pd.Series) -> list:
    try:
        years = pd.to_datetime(ser, errors='coerce').dt.year.dropna().astype(int)
        return sorted(np.unique(years).tolist())
    except Exception:
        return []

def run_integrity_checks(df: pd.DataFrame, dpd_threshold: int, gap_days: int = 120, after_mat_tol_days: int = 31):
    summary = {}
    vintage_issues = []
    row_issue_map: dict[int, list[str]] = {}

    def track(mask: pd.Series, issue: str, data: pd.DataFrame = None):
        """Add up to 500 offending rows for `issue` using the index of `data`."""
        nonlocal row_issue_map
        if data is None:
            data = dfn
        mask = mask.fillna(False)
        if mask.any():
            idxs = data[mask].head(500).index
            for i in idxs:
                row_issue_map.setdefault(int(i), []).append(issue)

    try:
        dfn = normalize_columns(df)
    except KeyError as e:
        return {'fatal': str(e)}, pd.DataFrame(), pd.DataFrame()
    dfn = ensure_types(dfn)

    # Nulls + non-parsables
    for c in ['Loan ID','Origination date','Observation date','Maturity date',
              'Days past due','Origination amount','Current amount']:
        summary[f'Nulls in {c}'] = int(dfn[c].isna().sum())
    for c in ['Origination date','Observation date','Maturity date',
              'Days past due','Origination amount','Current amount']:
        coerced_nan = dfn[c].isna() & dfn[f'__orig_{c}'].notna()
        summary[f'Non-parsable {c} (after coercion)'] = int(coerced_nan.sum())
        track(coerced_nan, f'Non-parsable {c}')


    summary['Years (Origination)'] = _years_list(dfn['Origination date'])
    summary['Years (Observation)'] = _years_list(dfn['Observation date'])
    summary['Years (Maturity)']    = _years_list(dfn['Maturity date'])


        # Date logic
    mask_obs_before_orig = dfn['Observation date'] < dfn['Origination date']
    summary['Observation before Origination'] = int(mask_obs_before_orig.sum())
    track(mask_obs_before_orig, 'Observation before Origination')

    mask_mat_before_orig = dfn['Maturity date'] < dfn['Origination date']
    summary['Maturity before Origination'] = int(mask_mat_before_orig.sum())
    track(mask_mat_before_orig, 'Maturity before Origination')

    mask_obs_after_mat = dfn['Observation date'] > (dfn['Maturity date'] + pd.to_timedelta(after_mat_tol_days, unit='D'))
    summary['Observation well after Maturity (> tol)'] = int(mask_obs_after_mat.sum())
    track(mask_obs_after_mat, 'Observation well after Maturity')


    

    # Snapshot uniqueness & continuity
    dup_mask = dfn.duplicated(subset=['Loan ID', 'Observation date'], keep=False)
    summary['Duplicate snapshots (Loan ID + Observation date)'] = int(dup_mask.sum())
    track(dup_mask, 'Duplicate snapshot')

    multi_orig = dfn.groupby('Loan ID')['Origination date'].nunique() > 1
    summary['Loans with multiple Origination dates'] = int(multi_orig.sum())
    if multi_orig.any():
        mask_multi_orig = dfn['Loan ID'].isin(multi_orig[multi_orig].index)
        track(mask_multi_orig, 'Multiple origination dates')

    multi_mat = dfn.groupby('Loan ID')['Maturity date'].nunique() > 1
    summary['Loans with changing Maturity date'] = int(multi_mat.sum())
    if multi_mat.any():
        mask_multi_mat = dfn['Loan ID'].isin(multi_mat[multi_mat].index)
        track(mask_multi_mat, 'Maturity date changed')

    dfn = dfn.sort_values(['Loan ID','Observation date'])
    diffs_days = dfn.groupby('Loan ID', sort=False)['Observation date'].diff().dt.days
    out_of_order = diffs_days < 0
    large_gap = diffs_days > gap_days
    summary['Out-of-order snapshots'] = int(out_of_order.fillna(False).sum())
    summary[f'Large gaps in Observation (>{gap_days} days)'] = int(large_gap.fillna(False).sum())
    track(out_of_order, 'Out-of-order snapshots')
    track(large_gap, 'Large gap between snapshots')

    # DPD quality
    neg_dpd_mask = dfn['Days past due'] < 0
    summary['Negative Days past due'] = int(neg_dpd_mask.sum())
    track(neg_dpd_mask, 'Negative DPD')

    non_int_mask = dfn['Days past due'].notna() & ((dfn['Days past due'] % 1) != 0)
    summary['Non-integer DPD values'] = int(non_int_mask.sum())
    track(non_int_mask, 'Non-integer DPD')

    extreme_dpd_mask = dfn['Days past due'] > 3650
    summary['Extreme DPD (> 3650)'] = int(extreme_dpd_mask.sum())
    track(extreme_dpd_mask, 'Extreme DPD')

    prev_dpd = dfn.groupby('Loan ID', sort=False)['Days past due'].shift()
    cure_mask = (prev_dpd >= 180) & (dfn['Days past due'] == 0)
    summary['Sudden cures (>=180 to 0 next)'] = int(cure_mask.fillna(False).sum())
    track(cure_mask, 'Sudden cure 180->0')
   
    # Amounts
    orig_amt_nonpos = dfn['Origination amount'] <= 0
    summary['Origination amount <= 0'] = int(orig_amt_nonpos.sum())
    track(orig_amt_nonpos, 'Non-positive Origination amount')

    orig_amt_changes = dfn.groupby('Loan ID')['Origination amount'].nunique() > 1
    summary['Loans with changing Origination amount'] = int(orig_amt_changes.sum())
    if orig_amt_changes.any():
        mask_orig_amt_changes = dfn['Loan ID'].isin(orig_amt_changes[orig_amt_changes].index)
        track(mask_orig_amt_changes, 'Origination amount changed')

    curr_amt_neg = dfn['Current amount'] < 0
    summary['Negative Current amount'] = int(curr_amt_neg.sum())
    track(curr_amt_neg, 'Negative Current amount')

    curr_gt_orig = dfn['Current amount'] > dfn['Origination amount']
    summary['Current amount > Origination amount'] = int(curr_gt_orig.sum())
    track(curr_gt_orig, 'Current > Origination')

    # Consistency + QOB aggregation sanity
    dfd = add_vintage_mob(dfn).sort_values(['Loan ID','Observation date'])
    dfd['is_def'] = (dfd['Days past due'] >= dpd_threshold).astype(np.uint8)
    dfd['is_def_cum'] = dfd.groupby('Loan ID', sort=False)['is_def'].cummax()
    def_cum_reset = dfd.groupby('Loan ID', sort=False)['is_def_cum'].diff() < 0
    summary['is_def_cum resets (should be 0)'] = int(def_cum_reset.fillna(False).sum())
    track(def_cum_reset, 'is_def_cum reset', data=dfd)

    dfd_q = add_qob(dfd)
    agg = dfd_q.groupby(['Vintage','QOB'], sort=False).agg(
        total_loans=('Loan ID','nunique'),
        total_default=('is_def_cum','sum')
    ).reset_index()
    vintages = sorted(agg['Vintage'].unique().tolist())
    summary['Vintages observed'] = vintages

    incr_rows = []
    for v in vintages:
        sub = agg[agg['Vintage']==v].sort_values('QOB')
        t = sub['total_loans'].to_numpy()
        if len(t) >= 2:
            inc = (np.diff(t) > 0)
            if inc.any():
                where = np.where(inc)[0]
                for idx in where:
                    q_prev = int(sub.iloc[idx]['QOB'])
                    q_curr = int(sub.iloc[idx+1]['QOB'])
                    incr_rows.append({'Vintage': v, 'QOB_prev': q_prev, 'QOB_curr': q_curr,
                                      'total_prev': int(sub.iloc[idx]['total_loans']),
                                      'total_curr': int(sub.iloc[idx+1]['total_loans']),
                                      'issue': 'Denominator increased with QOB'})
    summary['Vintage denominators increasing (count, QOB)'] = len(incr_rows)
    if incr_rows:
        vintage_issues.append(pd.DataFrame(incr_rows))

    qob1 = agg[agg['QOB']==1].set_index('Vintage')['total_loans']
    orig_counts = dfd_q.groupby('Vintage')['Loan ID'].nunique()
    coverage = pd.DataFrame({'orig_loans': orig_counts}).join(qob1.rename('qob1_loans'), how='left')
    coverage['qob1_coverage_%'] = 100 * (coverage['qob1_loans'] / coverage['orig_loans'])
    low_cov = coverage[coverage['qob1_coverage_%'].fillna(0) < 80]
    summary['Vintages with QOB1 coverage < 80%'] = int(len(low_cov))
    if not low_cov.empty:
        tmp = low_cov.reset_index()
        tmp['issue'] = 'Low QOB1 coverage'
        vintage_issues.append(tmp[['Vintage','orig_loans','qob1_loans','qob1_coverage_%','issue']])

    raw = dfd_q.groupby(['Vintage','QOB'], sort=False).agg(default_rate=('is_def_cum','mean')).reset_index()
    non_monotone = []
    for v in vintages:
        sub = raw[raw['Vintage']==v].sort_values('QOB')
        r = sub['default_rate'].to_numpy()
        if len(r) >= 2 and np.any(np.diff(r) < -1e-12):
            non_monotone.append(v)
    summary['Vintages with non-monotone raw default rate (QOB)'] = len(non_monotone)
    if non_monotone:
        vintage_issues.append(pd.DataFrame({'Vintage': non_monotone,'issue': 'Raw default rate not monotone (QOB)'}))
    if row_issue_map:
        sample_indices = list(row_issue_map.keys())
        rows = dfn.loc[sample_indices].copy()
        rows['issues'] = ['; '.join(row_issue_map[idx]) for idx in sample_indices]
        issues_df = rows.reset_index()
    else:
        issues_df = pd.DataFrame()

    vintage_issues_df = pd.concat(vintage_issues, ignore_index=True) if vintage_issues else pd.DataFrame()

    

    summary['Rows (total)'] = int(len(dfn))
    summary['Distinct loans'] = int(dfn['Loan ID'].nunique())
    try:
        summary['Date range (Origination)'] = f"{dfn['Origination date'].min().date()} → {dfn['Origination date'].max().date()}"
        summary['Date range (Observation)'] = f"{dfn['Observation date'].min().date()} → {dfn['Observation date'].max().date()}"
        summary['Date range (Maturity)']    = f"{dfn['Maturity date'].min().date()} → {dfn['Maturity date'].max().date()}"
    except Exception:
        pass

    return summary, issues_df, vintage_issues_df

# Human-readable descriptions for integrity checks
CHECK_DESCRIPTIONS = {
    'Observation before Origination': 'Observation date occurs before origination date for a loan.',
    'Maturity before Origination': 'Maturity date occurs before origination date.',
    'Observation well after Maturity (> tol)': 'Observation date is far beyond maturity date.',
    'Duplicate snapshots (Loan ID + Observation date)': 'Same loan has multiple rows with identical observation date.',
    'Loans with multiple Origination dates': 'A loan appears with more than one origination date.',
    'Loans with changing Maturity date': 'A loan shows different maturity dates across records.',
    'Out-of-order snapshots': 'Observation dates are not in chronological order for a loan.',
    'Large gaps in Observation': 'Long intervals between successive observations for a loan.',
    'Negative Days past due': 'Days past due value is negative.',
    'Non-integer DPD values': 'Days past due is not a whole number.',
    'Extreme DPD (> 3650)': 'Days past due exceeds ten years.',
    'Sudden cures (>=180 to 0 next)': 'DPD drops from ≥180 to 0 in the next snapshot.',
    'Origination amount <= 0': 'Origination amount is non-positive.',
    'Loans with changing Origination amount': 'Origination amount varies for the same loan.',
    'Negative Current amount': 'Current amount is negative.',
    'Current amount > Origination amount': 'Current amount exceeds the origination amount.',
    'is_def_cum resets (should be 0)': 'Cumulative default flag decreases, which should not happen.',
    'Vintage denominators increasing (count, QOB)': 'Number of loans grows with ageing, implying duplicates.',
    'Vintages with QOB1 coverage < 80%': 'Vintages where fewer than 80% of loans have an initial snapshot.',
    'Vintages with non-monotone raw default rate (QOB)': 'Default rate decreases between QOBs for a vintage.',
    'Rows (total)': 'Total rows in dataset.',
    'Distinct loans': 'Count of unique Loan ID values.',
    'Date range (Origination)': 'Earliest and latest origination dates.',
    'Date range (Observation)': 'Earliest and latest observation dates.',
    'Date range (Maturity)': 'Earliest and latest maturity dates.',
    'Years (Origination)': 'Distinct origination years present.',
    'Years (Observation)': 'Distinct observation years present.',
    'Years (Maturity)': 'Distinct maturity years present.',
    'Vintages observed': 'Vintages represented after processing.'
}

def explain_check(name: str) -> str:
    if name.startswith('Nulls in '):
        col = name[len('Nulls in '):]
        return f'Rows where {col} is missing.'
    if name.startswith('Non-parsable '):
        col = name[len('Non-parsable '):].split(' (')[0]
        return f'Values in {col} could not be parsed to the required type.'
    if name.startswith('Large gaps in Observation'):
        return CHECK_DESCRIPTIONS['Large gaps in Observation']
    return CHECK_DESCRIPTIONS.get(name, '')

# ──────────────────────────────────────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────────────────────────────────────
def export_integrity_pdf(summary: dict, dataset_label: str = 'Full dataset') -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        plt.axis('off')

        y = 0.95

        # Title
        plt.text(0.5, y, 'Data Integrity Report', ha='center', va='top',
                fontsize=22, weight='bold', color='#1E3A8A')
        y -= 0.05

        # Subtitle
        plt.text(0.5, y, 'Vintage Default-Rate Analytics', ha='center', va='top',
                fontsize=14, color='#64748B')
        y -= 0.04

        # Dataset info
        import datetime
        plt.text(0.5, y, f'Dataset: {dataset_label}  |  Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ha='center', va='top', fontsize=11, color='#002244')
        y -= 0.06

        # Summary section header
        plt.text(0.05, y, 'Analysis Summary', ha='left', va='top',
                fontsize=16, weight='bold', color='#002244')
        y -= 0.05

        # Build summary bullets
        bullets = []
        explanations = []
        for k, v in summary.items():
            if isinstance(v, list):
                v = ', '.join(map(str, v[:10])) + (' ...' if len(v) > 10 else '')
            bullets.append(f'  {k}: {v}')
            desc = explain_check(k)
            if desc:
                explanations.append(f'  {k}: {desc}')

        # Print summary items with larger font
        wrapped = []
        for line in bullets:
            wrapped.extend(textwrap.wrap(line, width=80))
        for line in wrapped:
            plt.text(0.05, y, line, ha='left', va='top', fontsize=11, color='#002244')
            y -= 0.028
            if y < 0.38:
                break

        y -= 0.03

        # Explanations section header
        plt.text(0.05, y, 'Check Definitions', ha='left', va='top',
                fontsize=16, weight='bold', color='#002244')
        y -= 0.05

        # Print explanations with readable font
        wrapped_desc = []
        for line in explanations:
            wrapped_desc.extend(textwrap.wrap(line, width=85))
        for line in wrapped_desc:
            plt.text(0.05, y, line, ha='left', va='top', fontsize=10, color='#1E3A8A')
            y -= 0.025
            if y < 0.05:
                break

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    return buf.getvalue()


def export_issues_excel(issues_df: pd.DataFrame, vintage_issues: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as xw:
        (issues_df if (issues_df is not None and not issues_df.empty)
         else pd.DataFrame({'note':['No row-level issues sampled']})).to_excel(xw, index=False, sheet_name='Row issues')
        (vintage_issues if (vintage_issues is not None and not vintage_issues.empty)
         else pd.DataFrame({'note':['No vintage-level issues']})).to_excel(xw, index=False, sheet_name='Vintage issues')
    return out.getvalue()
# ──────────────────────────────────────────────────────────────────────────────
# Vintage default summary (Cum_PD, Obs_Time, annualized rate)
# ──────────────────────────────────────────────────────────────────────────────
def compute_vintage_default_summary(raw_df: pd.DataFrame, dpd_threshold: int) -> pd.DataFrame:
    dfn = normalize_columns(raw_df); dfn = ensure_types(dfn); dfn = add_vintage_mob(dfn)
    dfn = dfn.sort_values(['Loan ID','Observation date'])
    dfn['__def'] = (dfn['Days past due'] >= dpd_threshold)

    g = dfn.groupby('Loan ID', sort=False)
    first_obs = g['Observation date'].min()
    last_obs  = g['Observation date'].max()
    first_vintage = g['Vintage'].first()
    first_def_date = (dfn.loc[dfn['__def']]
                        .groupby('Loan ID', sort=False)['Observation date']
                        .min())

    loan_df = pd.DataFrame({'Vintage': first_vintage, 'first_obs': first_obs, 'last_obs': last_obs})
    loan_df['def_date']  = first_def_date
    loan_df['defaulted'] = loan_df['def_date'].notna()
    loan_df['obs_end']   = loan_df['def_date'].fillna(loan_df['last_obs'])

    obs_days = (loan_df['obs_end'] - loan_df['first_obs']).dt.days
    loan_df['Obs_Time_years'] = (obs_days / 365.25).astype('float32')
    loan_df.loc[loan_df['Obs_Time_years'] <= 0, 'Obs_Time_years'] = np.nan

    out = (loan_df.groupby('Vintage', as_index=False)
           .agg(Unique_loans=('Vintage','size'),
                Defaulted_loans=('defaulted','sum'),
                Cum_PD=('defaulted','mean'),
                Observation_Time=('Obs_Time_years','median')))
    m = out['Observation_Time'] > 0
    out['Default_rate_pa'] = np.nan
    out.loc[m, 'Default_rate_pa'] = 1 - np.power(1 - out.loc[m, 'Cum_PD'], 1 / out.loc[m, 'Observation_Time'])
    return out.sort_values('Vintage').reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# UI - SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Sidebar branding - light background with dark text
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 16px 0 24px 0;
            border-bottom: 1px solid {WB_BORDER};
            margin-bottom: 24px;
        ">
            <div style="
                background: {WB_PRIMARY};
                width: 56px;
                height: 56px;
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 12px auto;
            ">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <h3 style="margin: 0; font-size: 1.1rem; font-weight: 600; color: {WB_PRIMARY};">Analytics Suite</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.75rem; color: {WB_MUTED};">Vintage Curve Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick start guide
    with st.expander("📘 Quick Start Guide", expanded=False):
        st.markdown(
            """
            **Step 1:** Configure your default threshold

            **Step 2:** Upload your Excel file (.xlsx)

            **Step 3:** Select sheet and click Load

            **Step 4:** Explore the analysis tabs
            """
        )

    st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# UI - MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 3], gap="large")

with left:
    # Settings Card
    st.markdown(
        f"""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            border: 1px solid {WB_BORDER};
            margin-bottom: 24px;
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <div style="
                    background: linear-gradient(135deg, {WB_PRIMARY} 0%, {WB_SECONDARY} 100%);
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <circle cx="12" cy="12" r="3"/>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                    </svg>
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.1rem; color: {WB_TEXT};">Configuration</h3>
                    <p style="margin: 0; font-size: 0.8rem; color: {WB_MUTED};">Analysis parameters</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    dpd_threshold = st.number_input(
        '🎯 Default Threshold (DPD ≥)',
        min_value=1,
        max_value=365,
        value=90,
        step=1,
        help='Loans are considered in default when Days Past Due exceeds this threshold'
    )

    pretty_ints = st.checkbox(
        '📊 Format with thousand separators',
        value=False,
        help='Display numbers as 12,345 instead of 12345'
    )

    st.markdown("---")

    # Upload Card
    st.markdown(
        f"""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            border: 1px solid {WB_BORDER};
            margin-bottom: 16px;
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="
                    background: linear-gradient(135deg, {WB_SUCCESS} 0%, #059669 100%);
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.1rem; color: {WB_TEXT};">Data Import</h3>
                    <p style="margin: 0; font-size: 0.8rem; color: {WB_MUTED};">Upload your Excel file</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        'Select Excel file',
        type=['xlsx'],
        accept_multiple_files=False,
        help='Upload a .xlsx file containing loan portfolio data'
    )

    # Persist full dataset and flagged row indices
    if 'df_full' not in st.session_state:
        st.session_state['df_full'] = None
    if 'flagged_indices' not in st.session_state:
        st.session_state['flagged_indices'] = set()

    if uploaded:
        size_mb = uploaded.size / (1024 * 1024)

        # File info display
        st.markdown(
            f"""
            <div style="
                background: {WB_LIGHT};
                border-radius: 10px;
                padding: 12px 16px;
                margin: 12px 0;
                border-left: 4px solid {WB_PRIMARY};
            ">
                <div style="font-size: 0.9rem; font-weight: 600; color: {WB_TEXT};">📁 {uploaded.name}</div>
                <div style="font-size: 0.8rem; color: {WB_MUTED};">Size: {size_mb:,.2f} MB</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if size_mb > MAX_MB:
            st.warning('⚠️ Large file detected. Consider filtering columns or using CSV format.')

        from openpyxl import load_workbook
        names = load_workbook(filename=BytesIO(uploaded.getvalue()), read_only=True, data_only=True).sheetnames
        sheet = st.selectbox('📋 Select worksheet', options=names, index=0)
        header_row = st.number_input('📍 Header row position', min_value=1, value=1, step=1, help='Row 1 = first row')

        if st.button('🚀 Load Dataset', type='primary', use_container_width=True):
            with st.status('Loading dataset...', expanded=True) as status:
                st.write("📖 Reading Excel file...")
                df_full = load_full(uploaded.getvalue(), sheet=sheet, header=header_row - 1)
                st.write(f"✅ Loaded {len(df_full):,} rows and {len(df_full.columns)} columns")
                st.session_state['df_full'] = df_full
                status.update(label='✅ Dataset loaded successfully!', state='complete')
with right:
    if st.session_state['df_full'] is not None:
        chosen_df_raw = st.session_state['df_full']

        # Dataset summary metrics
        st.markdown(
            f"""
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            ">
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    border: 1px solid {WB_BORDER};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">Total Rows</div>
                    <div style="color: {WB_TEXT}; font-size: 1.8rem; font-weight: 700;">{len(chosen_df_raw):,}</div>
                </div>
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    border: 1px solid {WB_BORDER};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">Columns</div>
                    <div style="color: {WB_TEXT}; font-size: 1.8rem; font-weight: 700;">{len(chosen_df_raw.columns)}</div>
                </div>
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    border: 1px solid {WB_BORDER};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">DPD Threshold</div>
                    <div style="color: {WB_PRIMARY}; font-size: 1.8rem; font-weight: 700;">≥{dpd_threshold}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Segmentation filter with card styling
        st.markdown(
            f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 16px 20px;
                border: 1px solid {WB_BORDER};
                margin-bottom: 24px;
            ">
                <div style="font-size: 0.85rem; font-weight: 600; color: {WB_TEXT}; margin-bottom: 8px;">
                    🔍 Data Segmentation (Optional)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        non_reserved = [c for c in chosen_df_raw.columns
                        if str(c).strip() not in RESERVED_COLS]
        seg_col = st.selectbox('Filter by column', ['None'] + non_reserved, label_visibility="collapsed")
        if seg_col != 'None':
            unique_vals = chosen_df_raw[seg_col].dropna().unique().tolist()
            selected_vals = st.multiselect(f'Select values for {seg_col}', unique_vals, default=unique_vals)
            if selected_vals:
                chosen_df_raw = chosen_df_raw[chosen_df_raw[seg_col].isin(selected_vals)]

        # Tabs with icons
        tab_integrity, tab_tables, tab_charts = st.tabs([
            "🛡️ Data Integrity",
            "📊 Summary Tables",
            "📈 Vintage Charts"
        ])

        # ════════════════════════════════════════════════════════════════════
        # INTEGRITY TAB
        # ════════════════════════════════════════════════════════════════════
        with tab_integrity:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {WB_LIGHT} 0%, white 100%);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid {WB_BORDER};
                    margin-bottom: 24px;
                ">
                    <h3 style="margin: 0 0 8px 0; color: {WB_TEXT};">Data Quality Assessment</h3>
                    <p style="margin: 0; color: {WB_MUTED}; font-size: 0.9rem;">
                        Run comprehensive integrity checks to identify data quality issues, missing values, and inconsistencies in your loan portfolio data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            dataset_label = 'Full Dataset'

            col_btn, col_spacer = st.columns([1, 2])
            with col_btn:
                run_checks = st.button('🔍 Run Integrity Analysis', type='primary', use_container_width=True)

            if run_checks:
                with st.status('🔄 Analyzing data quality...', expanded=True) as status:
                    st.write("Validating data schema...")
                    st.write("Checking date consistency...")
                    st.write("Analyzing value ranges...")
                    summary, issues_df, vintage_issues_df = run_integrity_checks(chosen_df_raw, dpd_threshold=dpd_threshold)
                    status.update(label='✅ Analysis complete!', state='complete')

                if 'fatal' in summary:
                    st.error(f"❌ Critical Error: {summary['fatal']}")
                else:
                    # Store flagged row indices for optional exclusion in curves
                    if issues_df is not None and not issues_df.empty and 'index' in issues_df.columns:
                        st.session_state['flagged_indices'] = set(issues_df['index'].tolist())
                    else:
                        st.session_state['flagged_indices'] = set()

                    st.success('✅ Integrity checks completed successfully!')

                    # Results summary card
                    st.markdown(
                        f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            border: 1px solid {WB_BORDER};
                            margin: 16px 0;
                        ">
                            <h4 style="margin: 0 0 16px 0; color: {WB_TEXT};">📋 Analysis Summary</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.json(summary)

                    # Download buttons in columns
                    st.markdown("#### 📥 Export Reports")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)

                    with dl_col1:
                        pdf_bytes = export_integrity_pdf(summary, dataset_label=dataset_label)
                        st.download_button(
                            '📄 PDF Report',
                            pdf_bytes,
                            'integrity_report.pdf',
                            'application/pdf',
                            use_container_width=True
                        )

                    with dl_col2:
                        xlsx_bytes = export_issues_excel(issues_df, vintage_issues_df)
                        st.download_button(
                            '📊 Excel Issues',
                            xlsx_bytes,
                            'integrity_issues_sample.xlsx',
                            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )

                    st.markdown("---")

                    # Issues display
                    if issues_df is not None and not issues_df.empty:
                        st.markdown(f"#### ⚠️ Row-Level Issues ({len(issues_df)} samples)")
                        st.dataframe(issues_df.head(500), use_container_width=True, height=300)
                    else:
                        st.info('✅ No row-level data quality issues detected.')

                    if vintage_issues_df is not None and not vintage_issues_df.empty:
                        st.markdown(f"#### 📊 Vintage/Cohort Issues ({len(vintage_issues_df)} items)")
                        st.dataframe(vintage_issues_df.head(500), use_container_width=True, height=300)
                    else:
                        st.info('✅ No vintage-level issues detected.')

        # ════════════════════════════════════════════════════════════════════
        # TABLES TAB
        # ════════════════════════════════════════════════════════════════════
        with tab_tables:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {WB_LIGHT} 0%, white 100%);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid {WB_BORDER};
                    margin-bottom: 24px;
                ">
                    <h3 style="margin: 0 0 8px 0; color: {WB_TEXT};">Vintage Performance Summary</h3>
                    <p style="margin: 0; color: {WB_MUTED}; font-size: 0.9rem;">
                        Comprehensive breakdown of loan performance metrics by vintage cohort, including cumulative default rates and annualized statistics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            try:
                summary_df = compute_vintage_default_summary(chosen_df_raw, dpd_threshold=dpd_threshold)

                # Key metrics cards
                total_loans = summary_df['Unique_loans'].sum()
                total_defaults = summary_df['Defaulted_loans'].sum()
                avg_pd = (total_defaults / total_loans * 100) if total_loans > 0 else 0

                st.markdown(
                    f"""
                    <div style="
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 16px;
                        margin-bottom: 24px;
                    ">
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            border: 1px solid {WB_BORDER};
                            text-align: center;
                        ">
                            <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase;">Total Loans</div>
                            <div style="color: {WB_TEXT}; font-size: 1.5rem; font-weight: 700;">{total_loans:,}</div>
                        </div>
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            border: 1px solid {WB_BORDER};
                            text-align: center;
                        ">
                            <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase;">Total Defaults</div>
                            <div style="color: #EF4444; font-size: 1.5rem; font-weight: 700;">{total_defaults:,}</div>
                        </div>
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            border: 1px solid {WB_BORDER};
                            text-align: center;
                        ">
                            <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase;">Avg Default Rate</div>
                            <div style="color: {WB_PRIMARY}; font-size: 1.5rem; font-weight: 700;">{avg_pd:.2f}%</div>
                        </div>
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 20px;
                            border: 1px solid {WB_BORDER};
                            text-align: center;
                        ">
                            <div style="color: {WB_MUTED}; font-size: 0.75rem; text-transform: uppercase;">Vintages</div>
                            <div style="color: {WB_TEXT}; font-size: 1.5rem; font-weight: 700;">{len(summary_df)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 16px 20px;
                        border: 1px solid {WB_BORDER};
                        margin-bottom: 16px;
                    ">
                        <div style="font-size: 0.9rem; font-weight: 600; color: {WB_TEXT};">📊 Vintage Default Summary Table</div>
                        <div style="font-size: 0.8rem; color: {WB_MUTED};">
                            Observation Time = default date − first observation (if defaulted), else last observation − first observation (in years)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Rename only for display
                disp = summary_df.rename(columns={
                    "Unique_loans": "Unique loans",
                    "Defaulted_loans": "Defaulted loans",
                    "Observation_Time": "Obs Time (years)",
                    "Default_rate_pa": "Annualized default rate",
                    "Cum_PD": "Cum PD",
                })
                disp["Cum PD (%)"] = disp["Cum PD"] * 100
                disp["Annualized default rate (%)"] = disp["Annualized default rate"] * 100

                table = disp[[
                    "Vintage",
                    "Unique loans",
                    "Defaulted loans",
                    "Cum PD (%)",
                    "Obs Time (years)",
                    "Annualized default rate (%)",
                ]]
                styles = {
                    "Unique loans": "{:,.0f}" if pretty_ints else "{:.0f}",
                    "Defaulted loans": "{:,.0f}" if pretty_ints else "{:.0f}",
                    "Cum PD (%)": "{:.2f}",
                    "Obs Time (years)": "{:.2f}",
                    "Annualized default rate (%)": "{:.2f}",
                }
                styler = (
                    table.style
                    .format(styles)
                    .background_gradient(subset=["Cum PD (%)", "Annualized default rate (%)"], cmap="Blues")
                    .hide(axis="index")
                )

                st.dataframe(styler, use_container_width=True, height=400)

                st.markdown("#### 📥 Export Data")
                st.download_button(
                    '📊 Download CSV',
                    summary_df.to_csv(index=False).encode('utf-8'),
                    'vintage_default_summary.csv',
                    'text/csv',
                    use_container_width=False
                )
            except Exception as e:
                st.error(f'❌ Could not compute vintage default summary: {e}')

        # ════════════════════════════════════════════════════════════════════
        # CHARTS TAB
        # ════════════════════════════════════════════════════════════════════
        with tab_charts:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {WB_LIGHT} 0%, white 100%);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid {WB_BORDER};
                    margin-bottom: 24px;
                ">
                    <h3 style="margin: 0 0 8px 0; color: {WB_TEXT};">Vintage Curve Visualization</h3>
                    <p style="margin: 0; color: {WB_MUTED}; font-size: 0.9rem;">
                        Interactive default rate curves showing cumulative performance evolution across loan cohorts over time.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_chart, col_settings = st.columns([3, 1], gap="large")

            with col_settings:
                # Settings panel card
                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid {WB_BORDER};
                        margin-bottom: 16px;
                    ">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                            <div style="
                                background: linear-gradient(135deg, {WB_PRIMARY} 0%, {WB_SECONDARY} 100%);
                                width: 32px;
                                height: 32px;
                                border-radius: 8px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            ">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                                    <line x1="4" y1="21" x2="4" y2="14"></line>
                                    <line x1="4" y1="10" x2="4" y2="3"></line>
                                    <line x1="12" y1="21" x2="12" y2="12"></line>
                                    <line x1="12" y1="8" x2="12" y2="3"></line>
                                    <line x1="20" y1="21" x2="20" y2="16"></line>
                                    <line x1="20" y1="12" x2="20" y2="3"></line>
                                    <line x1="1" y1="14" x2="7" y2="14"></line>
                                    <line x1="9" y1="8" x2="15" y2="8"></line>
                                    <line x1="17" y1="16" x2="23" y2="16"></line>
                                </svg>
                            </div>
                            <span style="font-weight: 600; color: {WB_TEXT};">Chart Settings</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                granularity = st.selectbox('📅 Time Granularity', ['Quarterly (QOB)', 'Monthly (MOB)'], index=0)
                gran_key = 'QOB' if 'QOB' in granularity else 'MOB'
                max_months_show = st.slider('📏 Max Months to Display', min_value=12, max_value=180, value=60, step=6)

                st.markdown("---")
                st.markdown("**Curve Processing**")
                smooth_curves = st.checkbox('🔄 Smooth curves', value=True)
                force_monotone = st.checkbox('📈 Force monotone (cummax)', value=True)
                cure_adjusted = st.checkbox(
                    '💊 Cure-adjusted mode',
                    value=False,
                    help='Use current DPD status at each snapshot instead of cumulative ever-defaulted flag.'
                )
                exclude_flagged = st.checkbox(
                    '🚫 Exclude flagged rows',
                    value=False,
                    help='Exclude rows flagged by integrity checks. Run integrity checks first.'
                )

                st.markdown("---")
                st.markdown("**Visual Options**")
                show_legend = st.checkbox('📋 Show legend', value=True)
                palette_option = st.selectbox('🎨 Color palette', ['Gradient', 'Plotly', 'Viridis'])
                base_color = st.color_picker(
                    '🖌️ Base color',
                    value=WB_PRIMARY,
                    help='Used when Gradient palette is selected.'
                )
                line_width = st.slider('✏️ Line width', min_value=1, max_value=5, value=2)

            with col_chart:
                try:
                    prog_bar = st.progress(0.0, text="Initializing chart pipeline...")
                    upd = mk_progress_updater(prog_bar, steps=5)

                    if gran_key == 'QOB':
                        max_periods = max(1, math.ceil(max_months_show / 3))
                    else:
                        max_periods = max_months_show

                    excl = st.session_state.get('flagged_indices', set()) if exclude_flagged else None

                    df_plot_any, cohort_sizes = build_chart_data_fast(
                        chosen_df_raw, dpd_threshold=dpd_threshold,
                        max_periods=max_periods,
                        granularity=gran_key,
                        smooth=smooth_curves,
                        force_monotone=force_monotone,
                        cure_adjusted=cure_adjusted,
                        exclude_indices=excl,
                        prog=upd,
                    )

                    if df_plot_any.empty:
                        prog_bar.progress(1.0, text="Complete")
                        st.warning('⚠️ Insufficient data to generate curves for the selected parameters.')
                    else:
                        vintages = df_plot_any.columns.tolist()
                        st.markdown(
                            f"""
                            <div style="
                                background: white;
                                border-radius: 10px;
                                padding: 12px 16px;
                                border: 1px solid {WB_BORDER};
                                margin-bottom: 16px;
                            ">
                                <span style="font-size: 0.85rem; font-weight: 500; color: {WB_TEXT};">
                                    🏷️ Select vintages to display ({len(vintages)} available)
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        selected_vintages = st.multiselect(
                            'Select vintages',
                            vintages,
                            default=vintages,
                            label_visibility="collapsed"
                        )

                        if not selected_vintages:
                            prog_bar.progress(1.0, text="Complete")
                            st.info('ℹ️ Select at least one vintage to display the chart.')
                        else:
                            df_plot = df_plot_any[selected_vintages]
                            prog_bar.progress(0.9, text="Rendering visualization...")
                            ttl = f'Vintage Default-Rate Evolution | DPD ≥ {dpd_threshold}'
                            if cure_adjusted:
                                ttl += ' | Cure-Adjusted'
                            fig = plot_curves_percent_with_months(
                                df_wide=df_plot,
                                title=ttl,
                                show_legend=show_legend,
                                legend_limit=50,
                                palette=palette_option,
                                base_color=base_color,
                                line_width=line_width,
                                cohort_sizes=cohort_sizes,
                            )
                            prog_bar.progress(1.0, text="✅ Chart ready")

                            st.plotly_chart(fig, use_container_width=True, config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                            })

                            # Export section
                            st.markdown("#### 📥 Export Chart Data")
                            period_col = df_plot.index.name or gran_key
                            export_df = df_plot.reset_index()
                            if gran_key == 'QOB':
                                export_df.insert(0, "Months", export_df[period_col] * 3)

                            exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
                            with exp_col1:
                                st.download_button(
                                    f'📊 Download CSV',
                                    export_df.to_csv(index=False).encode('utf-8'),
                                    f'vintage_curves_{gran_key.lower()}.csv',
                                    'text/csv',
                                    use_container_width=True
                                )

                except Exception as e:
                    st.error(f'❌ Chart generation failed: {e}')


    else:
        # Empty state - no data loaded
        st.markdown(
            f"""
            <div style="
                background: white;
                border-radius: 20px;
                padding: 60px 40px;
                text-align: center;
                border: 2px dashed {WB_BORDER};
                margin: 40px 0;
            ">
                <div style="
                    background: {WB_LIGHT};
                    width: 80px;
                    height: 80px;
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 24px auto;
                ">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="{WB_PRIMARY}" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <h2 style="margin: 0 0 12px 0; color: {WB_TEXT}; font-weight: 600;">No Data Loaded</h2>
                <p style="margin: 0; color: {WB_MUTED}; font-size: 1rem; max-width: 400px; margin: 0 auto;">
                    Upload an Excel file (.xlsx) using the panel on the left to begin your vintage curve analysis.
                </p>
                <div style="margin-top: 24px;">
                    <span style="
                        background: {WB_LIGHT};
                        color: {WB_PRIMARY};
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: 500;
                    ">
                        Supported format: .xlsx
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        















