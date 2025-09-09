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

st.set_page_config(page_title='Vintage Curves (QOB) + Integrity — Ultra-Fast', layout='wide')

# Density selector allows comfortable or compact spacing modes
density_mode = st.sidebar.selectbox('Density', ['Comfortable', 'Compact'], index=0)

# Blue palette with white background
# Darker blues for sidebar and accents
WB_PRIMARY = "#1E3A8A"      # Button and link color
WB_SECONDARY = "#1E40AF"    # Sidebar background
WB_BG = "#FFFFFF"           # Page background
WB_TEXT = "#002244"         # General text color
st.markdown(
    f"""
    <style>
        :root {{
            --space-1: 8px;
            --space-2: 16px;
            --space-3: 24px;
            --space-4: 32px;
            --space-5: 48px;
        }}
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {WB_BG};
            line-height: 1.5;
        }}
        [data-testid="stAppViewContainer"] {{
            padding: var(--space-3);
        }}
        *, *::placeholder {{
            color: {WB_TEXT} !important;
        }}
       [data-testid="stSidebar"] {{
            background-color: {WB_SECONDARY};
            padding: var(--space-3);
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        [data-testid="stSidebar"] .stSelectbox * {{
            color: black !important;
        }}
        .block-container {{
            padding-top: var(--space-4);
            padding-bottom: var(--space-4);
        }}
        .stButton>button {{
            background-color: {WB_PRIMARY};
            margin-top: var(--space-2);
            margin-bottom: var(--space-2);
            padding: var(--space-1) var(--space-2);
            border: none;
        }}
        .stButton>button, .stButton>button * {{
            color: white !important;
        }}
        .stButton>button:hover {{
            filter: brightness(1.1);
        }}
        .stButton>button:active {{
            filter: brightness(0.9);
        }}
        .stButton>button:focus {{
            outline: 3px solid white;
            outline-offset: 2px;
        }}
        a {{
            color: {WB_PRIMARY} !important;
        }}
         p {{
            max-width: 75ch;
        }}
        [data-baseweb="tag"] {{
            background-color: {WB_PRIMARY};
            color: white !important;
        }}
        [data-baseweb="tag"] [data-baseweb="close"] {{
            color: white !important;
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
            .stButton>button {
                padding: var(--space-1);
                margin-top: var(--space-1);
                margin-bottom: var(--space-1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title('Vintage Default-Rate Tool')

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
    if not col_curr_amt: missing.append('Current amount')
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")
    return df.rename(columns={
        col_loan:     'Loan ID',
        col_dpd:      'Days past due',
        col_orig:     'Origination date',
        col_obs:      'Observation date',
        col_mat:      'Maturity date',
        col_orig_amt: 'Origination amount',
        col_curr_amt: 'Current amount',
    })

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
    sample = df[cols].head(200000) if len(df) > 200000 else df[cols]
    h = pd.util.hash_pandas_object(sample, index=True).values
    m = hashlib.blake2b(digest_size=16)
    m.update(h.tobytes()); m.update(str(sample.shape).encode()); m.update(",".join(map(str, sample.columns)).encode())
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
# QUARTER (QOB) chart pipeline
# ──────────────────────────────────────────────────────────────────────────────
def build_chart_data_fast_quarter(raw_df: pd.DataFrame, dpd_threshold: int, max_qob: int = 20,
                                  prog: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    if prog: prog("Preparing base dataset …")
    base = prepare_base_cached(raw_df)
    base = add_qob(base)

    if prog: prog("Computing default flags …")
    flags = (base['Days past due'].to_numpy(np.float32, copy=False) >= dpd_threshold).astype(np.uint8, copy=False)
    codes = base['Loan ID'].cat.codes.to_numpy(np.int64, copy=False)

    if prog: prog("Applying per-loan cummax …")
    is_def_cum = _cum_or_by_group(codes, flags)

    if prog: prog("Aggregating cohorts (QOB) …")
    agg = (pd.DataFrame({
                'Vintage': base['Vintage'].to_numpy(),
                'QOB': base['QOB'].to_numpy(),
                'is_def_cum': is_def_cum,
                'LoanID': base['Loan ID'].to_numpy()
           })
           .groupby(['Vintage','QOB'], sort=False)
           .agg(total_loans=('LoanID','nunique'),
                total_default=('is_def_cum','sum'))
           .reset_index())

    if agg.empty:
        if prog: prog("No data to plot.")
        return pd.DataFrame()

    agg['QOB'] = pd.to_numeric(agg['QOB'], errors='coerce').astype('int16')
    agg['default_rate'] = (agg['total_default'] / agg['total_loans']).astype('float32')
    max_q = min(int(agg['QOB'].max()), max_qob)

    wide = (agg.pivot(index='QOB', columns='Vintage', values='default_rate')
               .sort_index()
               .reindex(range(1, max_q+1)))
    wide = wide.rolling(2, 1, center=True).mean().cummax().astype('float32')
    wide.index.name = 'QOB'
    return wide

# ──────────────────────────────────────────────────────────────────────────────
# Plotting: % y-axis, months x-axis (QOB*3), optional legend
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves_percent_with_months(df_wide: pd.DataFrame,
                                    title: str,
                                    show_legend: bool = True,
                                    legend_limit: int = 40,
                                    palette: str = "Gradient",
                                    base_color: Optional[str] = None,
                                    line_width: int = 1):
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
        fig.add_trace(go.Scatter(
            x=x_months,
            y=Y[:, i],
            mode='lines',
            name=str(col),
            line=dict(color=palette_colors[i], width=line_width),
            hovertemplate=f"Vintage: {col}<br>Month: %{{x}}<br>Default rate: %{{y:.2%}}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Deal Age (months)',
        yaxis=dict(title='Cumulative default rate', tickformat='.2%'),
        hovermode='x unified',
        showlegend=show_leg,
        legend=dict(bgcolor=WB_PRIMARY, font=dict(color="white")),
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
        """Add up to 50 offending rows for `issue` using the index of `data`."""
        nonlocal row_issue_map
        if data is None:
            data = dfn
        mask = mask.fillna(False)
        if mask.any():
            idxs = data[mask].head(50).index
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
        plt.text(0.5, y, 'Data Integrity Report', ha='center', va='top', fontsize=18, weight='bold'); y -= 0.05
        plt.text(0.5, y, f'Dataset: {dataset_label}', ha='center', va='top', fontsize=11); y -= 0.05
        bullets = []
        explanations = []
        for k, v in summary.items():
            if isinstance(v, list):
                v = ', '.join(map(str, v[:12])) + (' …' if len(v) > 12 else '')
            bullets.append(f'• {k}: {v}')
            desc = explain_check(k)
            if desc:
                explanations.append(f'• {k}: {desc}')
        wrapped = []
        for line in bullets:
            wrapped.extend(textwrap.wrap(line, width=90))
        for line in wrapped:
            plt.text(0.05, y, line, ha='left', va='top', fontsize=10); y -= 0.03
        y -= 0.02
        plt.text(0.5, y, 'Check explanations', ha='center', va='top', fontsize=12, weight='bold'); y -= 0.04
        wrapped_desc = []
        for line in explanations:
            wrapped_desc.extend(textwrap.wrap(line, width=90))
        for line in wrapped_desc:
            plt.text(0.05, y, line, ha='left', va='top', fontsize=8, color='#5B9BD5'); y -= 0.03
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
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
                Observation_Time=('Obs_Time_years','mean')))
    m = out['Observation_Time'] > 0
    out['Default_rate_pa'] = np.nan
    out.loc[m, 'Default_rate_pa'] = 1 - np.power(1 - out.loc[m, 'Cum_PD'], 1 / out.loc[m, 'Observation_Time'])
    return out.sort_values('Vintage').reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Instructions')
    st.markdown(
        "1. Configure settings below.\n"
        "2. Upload an Excel (.xlsx) file.\n"
        "3. Click **Load dataset**.\n"
        "4. Explore the tabs for integrity checks, tables, and charts."
    )

left, right = st.columns([0.5, 2], gap="large")

with left:
    st.header('Settings')
    dpd_threshold = st.number_input('Default if Days past due ≥', min_value=1, max_value=365, value=90, step=1)
    pretty_ints = st.checkbox(
        'Thousands separators for integer columns',
        value=False,
        help='Shows 12,345 instead of 12345; disables numeric sorting on those two columns.',
    )

    st.subheader('Upload Excel')
    uploaded = st.file_uploader('Upload a .xlsx file', type=['xlsx'], accept_multiple_files=False)

    # Persist full dataset
    if 'df_full' not in st.session_state:
        st.session_state['df_full'] = None

    if uploaded:
        size_mb = uploaded.size / (1024 * 1024)
        st.caption(f'File size, {size_mb:,.1f} MB')
        if size_mb > MAX_MB:
            st.warning('Large file, consider filtering columns or using CSV.')

        from openpyxl import load_workbook
        names = load_workbook(filename=BytesIO(uploaded.getvalue()), read_only=True, data_only=True).sheetnames
        sheet = st.selectbox('Select sheet', options=names, index=0)
        header_row = st.number_input('Header row [1 means first row]', min_value=1, value=1, step=1)

        if st.button('Load dataset', type='primary'):
            with st.status('Loading dataset...', expanded=False) as status:
                df_full = load_full(uploaded.getvalue(), sheet=sheet, header=header_row - 1)
                st.session_state['df_full'] = df_full
                status.update(label='Dataset loaded.', state='complete')
with right:
    if st.session_state['df_full'] is not None:
        st.divider()
        chosen_df_raw = st.session_state['df_full']
        tab_integrity, tab_tables, tab_charts = st.tabs(["Integrity", "Tables", "Charts"])

        # Integrity
        with tab_integrity:
            st.subheader('Integrity checks, PDF (summary only) & Excel export')
            dataset_label = 'Full'

            if st.button('Run integrity checks and generate outputs'):
                with st.status('Running checks...', expanded=False):
                    summary, issues_df, vintage_issues_df = run_integrity_checks(chosen_df_raw, dpd_threshold=dpd_threshold)

                if 'fatal' in summary:
                    st.error(summary['fatal'])
                else:
                    st.success('Checks complete.')
                    st.json(summary)

                    pdf_bytes = export_integrity_pdf(summary, dataset_label=dataset_label)
                    st.download_button('Download integrity report (PDF, summary-only)', pdf_bytes,
                                       'integrity_report.pdf', 'application/pdf')

                    xlsx_bytes = export_issues_excel(issues_df, vintage_issues_df)
                    st.download_button('Download issues sample (Excel)', xlsx_bytes,
                                       'integrity_issues_sample.xlsx',
                                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                    if issues_df is not None and not issues_df.empty:
                        st.caption('Row-level issues (sample)')
                        st.dataframe(issues_df.head(200), use_container_width=True)
                    else:
                        st.info('No row-level issues sampled.')

                    if vintage_issues_df is not None and not vintage_issues_df.empty:
                        st.caption('Vintage/cohort issues')
                        st.dataframe(vintage_issues_df.head(200), use_container_width=True)
                    else:
                        st.info('No vintage-level issues detected.')

            st.divider()

        # ---- Vintage table (explicit formatting) ----
        with tab_tables:
            st.subheader('Unique loans & default summary by vintage')
            try:
                summary_df = compute_vintage_default_summary(chosen_df_raw, dpd_threshold=dpd_threshold)
                st.caption('Observation_Time = default date − first obs (if defaulted), else last obs − first obs (years).')

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
                    .bar(subset=["Cum PD (%)", "Annualized default rate (%)"], color="#5B9BD5")
                    .hide(axis="index")
                )

                st.dataframe(styler, use_container_width=True)

                st.download_button(
                    'Download CSV (vintage default summary)',
                    summary_df.to_csv(index=False).encode('utf-8'),
                    'vintage_default_summary.csv','text/csv'
                )
            except Exception as e:
                st.info(f'Could not compute vintage default summary: {e}')

            st.divider()

        # ---- Vintage curves — QOB engine, months axis, % y, legend ----
        with tab_charts:
           st.subheader('Vintage curves (months on axis)')
            col_chart, col_settings = st.columns([3, 0.5], gap="large")

            with col_settings:
                st.header('Chart settings')
                max_months_show = st.slider('Show curves up to (months)', min_value=12, max_value=180, value=60, step=6)
                show_legend = st.checkbox('Show legend in chart', value=True)
                palette_option = st.selectbox('Color palette', ['Gradient', 'Plotly', 'Viridis'])
                base_color = st.color_picker(
                    'Base chart color',
                    value=st.get_option("theme.primaryColor") or '#1f77b4',
                    help='Used when Gradient palette is selected.',
                )
                line_width = st.slider('Line width', min_value=1, max_value=5, value=1)

            with col_chart:
                try:
                    prog_bar = st.progress(0.0, text="Initializing …")
                    upd = mk_progress_updater(prog_bar, steps=5)

                    max_qob_show = max(1, math.ceil(max_months_show / 3))
                    df_plot_any = build_chart_data_fast_quarter(
                        chosen_df_raw, dpd_threshold=dpd_threshold, max_qob=max_qob_show, prog=upd
                    )

                    if df_plot_any.empty:
                        prog_bar.progress(1.0, text="No data to plot.")
                        st.info('Not enough data to plot curves for the chosen dataset.')
                    else:
                        vintages = df_plot_any.columns.tolist()
                        selected_vintages = st.multiselect('Vintages to display', vintages, default=vintages)
                        if not selected_vintages:
                            prog_bar.progress(1.0, text="No vintages selected.")
                            st.info('Select at least one vintage to plot.')
                        else:
                            df_plot = df_plot_any[selected_vintages]
                            prog_bar.progress(0.9, text="Rendering chart …")
                            ttl = f'Vintage Default-Rate Evolution | DPD≥{dpd_threshold}'
                            fig = plot_curves_percent_with_months(
                                df_wide=df_plot,
                                title=ttl,
                                show_legend=show_legend,
                                legend_limit=50,
                                palette=palette_option,
                                base_color=base_color,
                                line_width=line_width,
                            )
                            prog_bar.progress(1.0, text="Done")

                            st.plotly_chart(fig, use_container_width=True)

                            export_df = df_plot.reset_index().rename(columns={"QOB":"QOB"})
                            export_df.insert(0, "Months", export_df["QOB"] * 3)
                            st.download_button('Download curves (CSV; Months + QOB)',
                                               export_df.to_csv(index=False).encode('utf-8'),
                                               'vintage_curves_qob.csv','text/csv')

                except Exception as e:
                    st.info(f'Plot skipped, {e}')


    else:
        st.caption('Upload an Excel to continue.')
        








