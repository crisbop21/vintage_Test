import hashlib
from io import BytesIO
from typing import Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st


def cache_data_smart(**kwargs):
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            def passthrough(func):
                return func
            return passthrough
    except Exception:
        pass
    return st.cache_data(**kwargs)


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
    df = df.copy()
    df['Vintage'] = df['Origination date'].dt.to_period('Q').astype(str)
    mob = ((df['Observation date'].dt.year - df['Origination date'].dt.year) * 12
          + (df['Observation date'].dt.month - df['Origination date'].dt.month)) + 1
    df['MOB'] = mob
    df = df[df['MOB'] > 0]
    return df


def add_qob(df: pd.DataFrame) -> pd.DataFrame:
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


def _df_key(df: pd.DataFrame) -> str:
    cols = [c for c in ['Loan ID','Origination date','Observation date','Maturity date',
                        'Days past due','Origination amount','Current amount'] if c in df.columns]
    if not cols:
        cols = list(df.columns)
    sample = df[cols].head(200000) if len(df) > 200000 else df[cols]
    h = pd.util.hash_pandas_object(sample, index=True).values
    m = hashlib.blake2b(digest_size=16)
    m.update(h.tobytes()); m.update(str(sample.shape).encode()); m.update(','.join(map(str, sample.columns)).encode())
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


try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


if NUMBA_OK:
    from numba import njit
    @njit(cache=True, fastmath=True)
    def _cum_or_by_group(codes: np.ndarray, flags: np.ndarray) -> np.ndarray:
        n = flags.size
        out = np.empty(n, np.uint8)
        if n == 0:
            return out
        prev = codes[0]; acc = flags[0] != 0; out[0] = 1 if acc else 0
        for i in range(1, n):
            c = codes[i]
            if c != prev:
                prev = c; acc = flags[i] != 0
            else:
                if flags[i] != 0:
                    acc = True
            out[i] = 1 if acc else 0
        return out
else:
    def _cum_or_by_group(codes: np.ndarray, flags: np.ndarray) -> np.ndarray:
        s_codes = pd.Series(codes, copy=False)
        s_flags = pd.Series(flags, copy=False)
        return (s_flags.groupby(s_codes, sort=False).cummax().astype(np.uint8).to_numpy(copy=False))


def build_chart_data_fast_quarter(raw_df: pd.DataFrame, dpd_threshold: int, max_qob: int = 20,
                                  prog: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    if prog:
        prog('Preparing base dataset …')
    base = prepare_base_cached(raw_df)
    base = add_qob(base)

    if prog:
        prog('Computing default flags …')
    flags = (base['Days past due'].to_numpy(np.float32, copy=False) >= dpd_threshold).astype(np.uint8, copy=False)
    codes = base['Loan ID'].cat.codes.to_numpy(np.int64, copy=False)

    if prog:
        prog('Applying per-loan cummax …')
    is_def_cum = _cum_or_by_group(codes, flags)

    if prog:
        prog('Aggregating cohorts (QOB) …')
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
        if prog:
            prog('No data to plot.')
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
