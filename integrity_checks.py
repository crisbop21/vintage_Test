import textwrap
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data_processing import (
    normalize_columns,
    ensure_types,
    add_vintage_mob,
    add_qob,
)

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


def export_integrity_pdf(summary: dict, dataset_label: str = 'Full dataset') -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
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
