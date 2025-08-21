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
from io import BytesIO
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages

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
st.title('Vintage Default-Rate Evolution (QOB) & Data Integrity — Ultra-Fast')

MAX_MB = 50
PREVIEW_ROWS = 2000
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
def load_preview(file_bytes: bytes, sheet: str, nrows: int, header: int):
    bio = BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet, nrows=nrows, header=header, engine='openpyxl')

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
                                    legend_limit: int = 40):
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

    fig, ax = plt.subplots(figsize=(10, 5))
    lines = ax.plot(x_months, Y, linewidth=0.9, antialiased=False)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel('Cumulative default rate')
    ax.set_xlabel('Deal Age (months)')
    ax.set_title(title)

    if show_legend and (df_wide.shape[1] <= legend_limit):
        for line, label in zip(lines, map(str, df_wide.columns)):
            line.set_label(label)
        ax.legend(title='Vintage', ncols=2, fontsize=8, frameon=False, loc='upper left')

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, clear_figure=True)
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
    row_issues = []
    vintage_issues = []

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
        if coerced_nan.any():
            row_issues.append(dfn[coerced_nan].head(50).assign(issue=f'Non-parsable {c}'))

    summary['Years (Origination)'] = _years_list(dfn['Origination date'])
    summary['Years (Observation)'] = _years_list(dfn['Observation date'])
    summary['Years (Maturity)']    = _years_list(dfn['Maturity date'])

    # Date logic
    mask_obs_before_orig = dfn['Observation date'] < dfn['Origination date']
    summary['Observation before Origination'] = int(mask_obs_before_orig.sum())
    if mask_obs_before_orig.any():
        row_issues.append(dfn[mask_obs_before_orig].head(50).assign(issue='Observation before Origination'))

    mask_mat_before_orig = dfn['Maturity date'] < dfn['Origination date']
    summary['Maturity before Origination'] = int(mask_mat_before_orig.sum())
    if mask_mat_before_orig.any():
        row_issues.append(dfn[mask_mat_before_orig].head(50).assign(issue='Maturity before Origination'))

    mask_obs_after_mat = dfn['Observation date'] > (dfn['Maturity date'] + pd.to_timedelta(after_mat_tol_days, unit='D'))
    summary['Observation well after Maturity (> tol)'] = int(mask_obs_after_mat.sum())
    if mask_obs_after_mat.any():
        row_issues.append(dfn[mask_obs_after_mat].head(50).assign(issue='Observation well after Maturity'))

    # Snapshot uniqueness & continuity
    dup_mask = dfn.duplicated(subset=['Loan ID', 'Observation date'], keep=False)
    summary['Duplicate snapshots (Loan ID + Observation date)'] = int(dup_mask.sum())
    if dup_mask.any():
        row_issues.append(dfn[dup_mask].head(50).assign(issue='Duplicate snapshot'))

    multi_orig = dfn.groupby('Loan ID')['Origination date'].nunique() > 1
    summary['Loans with multiple Origination dates'] = int(multi_orig.sum())
    if multi_orig.any():
        row_issues.append(dfn[dfn['Loan ID'].isin(multi_orig[multi_orig].index)].head(50).assign(issue='Multiple origination dates'))

    multi_mat = dfn.groupby('Loan ID')['Maturity date'].nunique() > 1
    summary['Loans with changing Maturity date'] = int(multi_mat.sum())
    if multi_mat.any():
        row_issues.append(dfn[dfn['Loan ID'].isin(multi_mat[multi_mat].index)].head(50).assign(issue='Maturity date changed'))

    dfn = dfn.sort_values(['Loan ID','Observation date'])
    diffs_days = dfn.groupby('Loan ID', sort=False)['Observation date'].diff().dt.days
    out_of_order = diffs_days < 0
    large_gap = diffs_days > gap_days
    summary['Out-of-order snapshots'] = int(out_of_order.fillna(False).sum())
    summary[f'Large gaps in Observation (>{gap_days} days)'] = int(large_gap.fillna(False).sum())
    if out_of_order.any():
        row_issues.append(dfn[out_of_order.fillna(False)].head(50).assign(issue='Out-of-order snapshots'))
    if large_gap.any():
        row_issues.append(dfn[large_gap.fillna(False)].head(50).assign(issue='Large gap between snapshots'))

    # DPD quality
    neg_dpd_mask = dfn['Days past due'] < 0
    summary['Negative Days past due'] = int(neg_dpd_mask.sum())
    if neg_dpd_mask.any():
        row_issues.append(dfn[neg_dpd_mask].head(50).assign(issue='Negative DPD'))

    non_int_mask = dfn['Days past due'].notna() & ((dfn['Days past due'] % 1) != 0)
    summary['Non-integer DPD values'] = int(non_int_mask.sum())
    if non_int_mask.any():
        row_issues.append(dfn[non_int_mask].head(50).assign(issue='Non-integer DPD'))

    extreme_dpd_mask = dfn['Days past due'] > 3650
    summary['Extreme DPD (> 3650)'] = int(extreme_dpd_mask.sum())
    if extreme_dpd_mask.any():
        row_issues.append(dfn[extreme_dpd_mask].head(50).assign(issue='Extreme DPD'))

    prev_dpd = dfn.groupby('Loan ID', sort=False)['Days past due'].shift()
    cure_mask = (prev_dpd >= 180) & (dfn['Days past due'] == 0)
    summary['Sudden cures (>=180 to 0 next)'] = int(cure_mask.fillna(False).sum())
    if cure_mask.any():
        row_issues.append(dfn[cure_mask.fillna(False)].head(50).assign(issue='Sudden cure 180->0'))

    # Amounts
    orig_amt_nonpos = dfn['Origination amount'] <= 0
    summary['Origination amount <= 0'] = int(orig_amt_nonpos.sum())
    if orig_amt_nonpos.any():
        row_issues.append(dfn[orig_amt_nonpos].head(50).assign(issue='Non-positive Origination amount'))

    orig_amt_changes = dfn.groupby('Loan ID')['Origination amount'].nunique() > 1
    summary['Loans with changing Origination amount'] = int(orig_amt_changes.sum())
    if orig_amt_changes.any():
        row_issues.append(dfn[dfn['Loan ID'].isin(orig_amt_changes[orig_amt_changes].index)].head(50).assign(issue='Origination amount changed'))

    curr_amt_neg = dfn['Current amount'] < 0
    summary['Negative Current amount'] = int(curr_amt_neg.sum())
    if curr_amt_neg.any():
        row_issues.append(dfn[curr_amt_neg].head(50).assign(issue='Negative Current amount'))

    curr_gt_orig = dfn['Current amount'] > dfn['Origination amount']
    summary['Current amount > Origination amount'] = int(curr_gt_orig.sum())
    if curr_gt_orig.any():
        row_issues.append(dfn[curr_gt_orig].head(50).assign(issue='Current > Origination'))

    # Consistency + QOB aggregation sanity
    dfd = add_vintage_mob(dfn).sort_values(['Loan ID','Observation date'])
    dfd['is_def'] = (dfd['Days past due'] >= dpd_threshold).astype(np.uint8)
    dfd['is_def_cum'] = dfd.groupby('Loan ID', sort=False)['is_def'].cummax()
    def_cum_reset = dfd.groupby('Loan ID', sort=False)['is_def_cum'].diff() < 0
    summary['is_def_cum resets (should be 0)'] = int(def_cum_reset.fillna(False).sum())
    if def_cum_reset.any():
        row_issues.append(dfd[def_cum_reset.fillna(False)].head(50).assign(issue='is_def_cum reset'))

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

    issues_df = pd.concat(row_issues, ignore_index=True) if row_issues else pd.DataFrame()
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
        for k, v in summary.items():
            if isinstance(v, list): v = ', '.join(map(str, v[:12])) + (' …' if len(v) > 12 else '')
            bullets.append(f'• {k}: {v}')
        wrapped = []
        for line in bullets:
            wrapped.extend(textwrap.wrap(line, width=90))
        for line in wrapped:
            plt.text(0.05, y, line, ha='left', va='top', fontsize=10); y -= 0.03
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()

def export_issues_excel(row_issues: pd.DataFrame, vintage_issues: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as xw:
        (row_issues if (row_issues is not None and not row_issues.empty)
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
st.subheader('Upload Excel')
uploaded = st.file_uploader('Upload a .xlsx file', type=['xlsx'], accept_multiple_files=False)

# Persist full dataset
if 'df_full' not in st.session_state:
    st.session_state['df_full'] = None

# Sidebar settings — in MONTHS
with st.sidebar:
    st.header('Settings')
    dpd_threshold = st.number_input('Default if Days past due ≥', min_value=1, max_value=365, value=90, step=1)
    max_months_show = st.slider('Show curves up to (months)', min_value=12, max_value=180, value=60, step=6)
    show_legend = st.checkbox('Show legend in chart', value=True)
    pretty_ints = st.checkbox(
        "Thousands separators for integer columns",
        value=False,
        help="Shows 12,345 instead of 12345; disables numeric sorting on those two columns."
    )

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    st.caption(f'File size, {size_mb:,.1f} MB')
    if size_mb > MAX_MB:
        st.warning('Large file, consider filtering columns or using CSV.')

    from openpyxl import load_workbook
    names = load_workbook(filename=BytesIO(uploaded.getvalue()), read_only=True, data_only=True).sheetnames
    sheet = st.selectbox('Select sheet', options=names, index=0)
    header_row = st.number_input('Header row [1 means first row]', min_value=1, value=1, step=1)

    # Preview (fast)
    with st.status('Reading a fast preview...', expanded=False) as status:
        preview_df = load_preview(uploaded.getvalue(), sheet=sheet, nrows=PREVIEW_ROWS, header=header_row - 1)
        status.update(label='Preview ready.', state='complete')
    st.write(f'Preview, up to {PREVIEW_ROWS:,} rows')
    st.dataframe(preview_df.head(1000), use_container_width=True)

    st.divider()

    # Full
    if st.button('Load full dataset', type='primary'):
        with st.status('Loading full dataset...', expanded=False) as status:
            df_full = load_full(uploaded.getvalue(), sheet=sheet, header=header_row - 1)
            st.session_state['df_full'] = df_full
            status.update(label='Full dataset loaded.', state='complete')

    st.divider()

    # Integrity
    st.subheader('Integrity checks, PDF (summary only) & Excel export')
    dataset_choice = st.radio('Choose dataset for checks', options=('Preview', 'Full (if loaded)'), horizontal=True)
    chosen_df_raw = preview_df if dataset_choice == 'Preview' or st.session_state['df_full'] is None else st.session_state['df_full']
    dataset_label = 'Preview' if (dataset_choice == 'Preview' or st.session_state['df_full'] is None) else 'Full'

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

        if pretty_ints:
            disp["Unique loans"] = disp["Unique loans"].map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
            disp["Defaulted loans"] = disp["Defaulted loans"].map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
            col_config = {
                "Vintage": st.column_config.TextColumn("Vintage"),
                "Unique loans": st.column_config.TextColumn("Unique loans"),
                "Defaulted loans": st.column_config.TextColumn("Defaulted loans"),
                "Cum PD (%)": st.column_config.ProgressColumn("Cum PD (%)", format="%.2f%%", min_value=0.0, max_value=100.0),
                "Obs Time (years)": st.column_config.NumberColumn("Obs Time (years)", format="%.2f"),
                "Annualized default rate (%)": st.column_config.NumberColumn("Annualized default rate (%)", format="%.2f%%"),
            }
        else:
            col_config = {
                "Vintage": st.column_config.TextColumn("Vintage"),
                "Unique loans": st.column_config.NumberColumn("Unique loans", format="%d"),
                "Defaulted loans": st.column_config.NumberColumn("Defaulted loans", format="%d"),
                "Cum PD": st.column_config.ProgressColumn("Cum PD", format="%.2f%%", min_value=0.0, max_value=1.0),
                "Obs Time (years)": st.column_config.NumberColumn("Obs Time (years)", format="%.2f"),
                "Annualized default rate": st.column_config.NumberColumn("Annualized default rate", format="%.2f%%"),
            }

        st.dataframe(
            disp[["Vintage","Unique loans","Defaulted loans","Cum PD (%)","Obs Time (years)","Annualized default rate (%)"]],
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
        )

        st.download_button(
            'Download CSV (vintage default summary)',
            summary_df.to_csv(index=False).encode('utf-8'),
            'vintage_default_summary.csv','text/csv'
        )
    except Exception as e:
        st.info(f'Could not compute vintage default summary: {e}')

    st.divider()

    # ---- Vintage curves — QOB engine, months axis, % y, legend ----
    st.subheader('Vintage curves (months on axis; QOB engine for stable denominators)')
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
            prog_bar.progress(0.9, text="Rendering chart …")
            ttl = f'Vintage Default-Rate Evolution | DPD≥{dpd_threshold}'
            fig = plot_curves_percent_with_months(df_wide=df_plot_any, title=ttl,
                                                  show_legend=show_legend, legend_limit=50)
            prog_bar.progress(1.0, text="Done")

            export_df = df_plot_any.reset_index().rename(columns={"QOB":"QOB"})
            export_df.insert(0, "Months", export_df["QOB"] * 3)
            st.download_button('Download curves (CSV; Months + QOB)',
                               export_df.to_csv(index=False).encode('utf-8'),
                               'vintage_curves_qob.csv','text/csv')

            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=160)
                st.download_button('Download chart (PNG)', buf.getvalue(),
                                   'vintage_curves_qob.png', 'image/png')
    except Exception as e:
        st.info(f'Plot skipped, {e}')

else:
    st.caption('Upload an Excel to continue.')



