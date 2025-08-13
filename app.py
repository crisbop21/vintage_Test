
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Vintage Curves", layout="wide")
st.title("Vintage Default-Rate Evolution")

MAX_MB = 50
PREVIEW_ROWS = 2000

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    key = {str(c).strip().lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            c = key.get(n.lower())
            if c is not None:
                return c
        return None
    col_loan = find("loan id","loan_id","loan number","loan no","id")
    col_dpd  = find("days past due","dpd","days_past_due")
    col_orig = find("origination date","origination_date","orig date","orig_date")
    col_obs  = find("observation date","observation_date","obs date","obs_date")
    missing = []
    if not col_loan: missing.append("Loan ID")
    if not col_dpd:  missing.append("Days past due")
    if not col_orig: missing.append("Origination date")
    if not col_obs:  missing.append("Observation date")
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")
    return df.rename(columns={
        col_loan: "Loan ID",
        col_dpd:  "Days past due",
        col_orig: "Origination date",
        col_obs:  "Observation date"
    })

@st.cache_data(show_spinner=False)
def load_preview(file_bytes: bytes, sheet: str, nrows: int, header: int):
    bio = BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet, nrows=nrows, header=header, engine="openpyxl")

@st.cache_data(show_spinner=False)
def load_full(file_bytes: bytes, sheet: str, header: int):
    bio = BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet, header=header, engine="openpyxl")

def build_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    for c in ["Origination date","Observation date"]:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["Vintage"] = df["Origination date"].dt.to_period("Q").astype(str)
    df["MOB"] = (
        (df["Observation date"].dt.year - df["Origination date"].dt.year) * 12
        + (df["Observation date"].dt.month - df["Origination date"].dt.month)
    ) + 1
    df["is_def"] = (pd.to_numeric(df["Days past due"], errors="coerce") > 90).astype(int)
    df = (df.sort_values(["Loan ID","Observation date"])
            .groupby("Loan ID", group_keys=False)
            .apply(lambda g: g.assign(is_def_cum=g["is_def"].cummax()))
            .reset_index(drop=True))
    agg = (df.groupby(["Vintage","MOB"], as_index=False)
             .agg(total_loans=("Loan ID","nunique"), total_default=("is_def_cum","sum")))
    if agg.empty:
        return pd.DataFrame()
    agg["default_rate"] = agg["total_default"] / agg["total_loans"]
    vintages = agg["Vintage"].unique()
    max_month = int(agg["MOB"].max())
    series = {}
    for v in vintages:
        s = agg.loc[agg["Vintage"]==v, ["MOB","default_rate"]].set_index("MOB")
        s = s.reindex(range(1, max_month+1))
        s["default_rate"] = s["default_rate"].interpolate("linear").cummax()
        series[v] = s["default_rate"]
    df_rate = pd.DataFrame(series); df_rate.index.name = "MOB"
    def trunc(s: pd.Series) -> pd.Series:
        ch = s.ne(s.shift())
        if not ch.any(): return s
        last = ch[ch].index[-1]
        return s.where(s.index <= last)
    df_rate = df_rate.apply(trunc).interpolate("linear", axis=0, limit_area="inside").rolling(3, 1, center=True).mean()
    return df_rate.loc[:60]

st.subheader("Upload Excel")
uploaded = st.file_uploader("Upload a .xlsx file", type=["xlsx"], accept_multiple_files=False)
if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    st.caption(f"File size, {size_mb:,.1f} MB")
    if size_mb > MAX_MB:
        st.warning("Large file, consider filtering columns or using CSV.")
    from openpyxl import load_workbook
    names = load_workbook(filename=BytesIO(uploaded.getvalue()), read_only=True, data_only=True).sheetnames
    sheet = st.selectbox("Select sheet", options=names, index=0)
    header_row = st.number_input("Header row [1 means first row]", min_value=1, value=1, step=1)
    with st.status("Reading a fast preview...", expanded=True) as status:
        preview_df = load_preview(uploaded.getvalue(), sheet=sheet, nrows=PREVIEW_ROWS, header=header_row - 1)
        status.update(label="Preview ready.", state="complete")
    st.write(f"Preview, up to {PREVIEW_ROWS:,} rows")
    st.dataframe(preview_df.head(1000), use_container_width=True)
    st.subheader("Quick plot from preview")
    try:
        df_plot_prev = build_chart_data(preview_df)
        if df_plot_prev.empty:
            st.info("Not enough data in preview to build curves. Load full dataset.")
        else:
            x = df_plot_prev.index.to_numpy()
            fig, ax = plt.subplots(figsize=(10,6))
            for v in df_plot_prev.columns:
                ax.plot(x, df_plot_prev[v].to_numpy(), label=v)
            ax.set_xlabel("Deal Age, months"); ax.set_ylabel("Cumulative default rate")
            ax.set_title("Vintage curves, preview"); ax.legend(title="Vintage", bbox_to_anchor=(1,1))
            st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"Preview plot skipped, {e}")
    st.divider()
    if st.button("Load full dataset and plot", type="primary"):
        with st.status("Loading full dataset...", expanded=True) as status:
            df_full = load_full(uploaded.getvalue(), sheet=sheet, header=header_row - 1)
            status.update(label="Building curves...", state="running")
            df_plot = build_chart_data(df_full)
            status.update(label="Done.", state="complete")
        if df_plot.empty:
            st.error("Could not build curves from the full dataset, check the required columns.")
        else:
            x = df_plot.index.to_numpy()
            fig, ax = plt.subplots(figsize=(10,6))
            for v in df_plot.columns:
                ax.plot(x, df_plot[v].to_numpy(), label=v)
            ax.set_xlabel("Deal Age, months"); ax.set_ylabel("Cumulative default rate")
            ax.set_title("Vintage Default-Rate Evolution, up to month 60")
            ax.legend(title="Vintage", bbox_to_anchor=(1,1))
            st.pyplot(fig, clear_figure=True)
else:
    st.caption("Upload an Excel to continue.")
