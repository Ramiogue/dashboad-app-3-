import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page setup & lightweight styling
# =========================
st.set_page_config(page_title="Merchant Portal", page_icon="📊", layout="wide")

PRIMARY = "#0B6E4F"      # green
GREY_BG = "#f5f7fb"      # soft grey
CARD_BG = "#ffffff"      # white
TEXT = "#1f2937"

# Subtle CSS to get a Power BI-ish card feel
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {GREY_BG};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-bottom: 8px;
    }}
    @media (max-width: 1400px) {{
        .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 900px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid #e5e7eb;
        border-left: 6px solid {PRIMARY};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        height: 100%;
    }}
    .kpi-title {{
        font-size: 0.80rem;
        color: #6b7280;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
    }}
    .card {{
        background: {CARD_BG};
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Auth (from Secrets)
# =========================
# Secrets example:
# COOKIE_KEY = "replace_with_random_secret"
# merchant_id_col = "Merchant Number - Business Name"  # or "Device Serial"
#
# [users."DS-0001"]
# name = "Store A"
# email = "storea@example.com"
# password_hash = "<bcrypt-hash>"
# merchant_id = "DS-0001"

users_cfg = st.secrets.get("users", {})
cookie_key = st.secrets.get("COOKIE_KEY", "change-me")
MERCHANT_ID_COL = st.secrets.get("merchant_id_col", "Merchant Number - Business Name")

# Build credentials for auth
creds = {"usernames": {}}
for uname, u in users_cfg.items():
    creds["usernames"][uname] = {
        "name": u["name"],
        "email": u["email"],
        "password": u["password_hash"],  # bcrypt hash only
    }

authenticator = stauth.Authenticate(
    credentials=creds,
    cookie_name="merchant_portal",
    key=cookie_key,
    cookie_expiry_days=7,
)
authenticator.login(location="main")

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if auth_status is False:
    st.error("Invalid credentials")
    st.stop()
elif auth_status is None:
    st.info("Please log in.")
    st.stop()

authenticator.logout(location="sidebar")
st.sidebar.write(f"Hello, **{name}**")

def get_user_record(cfg: dict, uname: str):
    if uname in cfg:
        return cfg[uname]
    uname_cf = str(uname).casefold()
    for k, v in cfg.items():
        if str(k).casefold() == uname_cf:
            return v
    return None

merchant_rec = get_user_record(users_cfg, username)
if not merchant_rec or "merchant_id" not in merchant_rec:
    st.error(
        "Merchant mapping not found for this user. "
        f"Got username = '{username}'. Available users = {list(users_cfg.keys())}. "
        "Check App → Settings → Secrets and ensure this username block exists and has 'merchant_id'."
    )
    st.stop()

merchant_id_value = merchant_rec["merchant_id"]

# =========================
# Load & prep data
# =========================
@st.cache_data(ttl=60)
def load_transactions():
    for p in ("sample_merchant_transactions.csv", "data/sample_merchant_transactions.csv"):
        try:
            df = pd.read_csv(p)
            df["__path__"] = p
            return df
        except Exception:
            pass
    raise FileNotFoundError(
        "CSV not found. Place it at repo root as 'sample_merchant_transactions.csv' "
        "or under 'data/' as 'data/sample_merchant_transactions.csv'."
    )

tx = load_transactions()

required_cols = {
    MERCHANT_ID_COL,
    "Transaction Date", "Request Amount", "Settle Amount",
    "Transaction Type", "Auth Code", "Decline Reason", "Date Payment Extract",
    "Terminal ID", "Device Serial", "Product Type", "Issuing Bank", "BIN"
}
missing = required_cols - set(tx.columns)
if missing:
    st.error(f"Missing required column(s) in CSV: {', '.join(sorted(missing))}")
    st.stop()

# Clean types
tx[MERCHANT_ID_COL] = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"] = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"] = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"] = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"] = tx["Date Payment Extract"].astype(str).fillna("")
for c in ["Product Type", "Issuing Bank", "Decline Reason", "Terminal ID", "Device Serial"]:
    tx[c] = tx[c].astype(str).fillna("")

# Merchant slice
f0 = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if f0.empty:
    st.warning(
        f"No transactions found for '{merchant_id_value}' in column '{MERCHANT_ID_COL}'. "
        "Check Secrets → merchant_id."
    )
    st.stop()

# Flags
def is_purchase(s): return s.str.lower().eq("purchase")
def is_refund(s): return s.str.lower().eq("refund")
def is_reversal(s): return s.str.lower().eq("reversal")

def approved_mask(df: pd.DataFrame) -> pd.Series:
    dr = df["Decline Reason"].str.strip()
    return dr.str.startswith("00") | (df["Auth Code"].astype(str).str.len() > 0)

def settled_mask(df: pd.DataFrame) -> pd.Series:
    has_extract = df["Date Payment Extract"].str.len() > 0
    nonzero = df["Settle Amount"].fillna(0) != 0
    return has_extract & nonzero

f0["is_purchase"] = is_purchase(f0["Transaction Type"])
f0["is_refund"] = is_refund(f0["Transaction Type"])
f0["is_reversal"] = is_reversal(f0["Transaction Type"])
f0["is_approved"] = approved_mask(f0)
f0["is_settled"] = settled_mask(f0)

# =========================
# Sidebar filters (POS Entry Mode removed as requested)
# =========================
st.sidebar.subheader("Filters")

min_d = pd.to_datetime(f0["Transaction Date"].min()).date()
max_d = pd.to_datetime(f0["Transaction Date"].max()).date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_d, max_d

def _multi(label, series, sort=True):
    opts = sorted(series.unique()) if sort else list(series.unique())
    sel = st.sidebar.multiselect(label, options=opts, default=opts)
    return sel

sel_declines = _multi("Decline Reason", f0["Decline Reason"])
sel_prodtype = _multi("Product Type", f0["Product Type"])
sel_issuer   = _multi("Issuing Bank", f0["Issuing Bank"])

flt = (f0["Transaction Date"].dt.date >= start_date) & (f0["Transaction Date"].dt.date <= end_date)
flt &= f0["Decline Reason"].isin(sel_declines)
flt &= f0["Product Type"].isin(sel_prodtype)
flt &= f0["Issuing Bank"].isin(sel_issuer)

f = f0[flt].copy()

# =========================
# Metrics
# =========================
def safe_div(n, d): return (n / d) if d else np.nan

attempts_mask = f["is_purchase"]
approved_purchases = attempts_mask & f["is_approved"]
declined_purchases = attempts_mask & (~f["is_approved"])
settled_purchases  = attempts_mask & f["is_settled"]
refunds_f          = f["is_refund"]

gross_requests = float(f.loc[attempts_mask, "Request Amount"].sum())
net_settled    = float(f.loc[settled_purchases, "Settle Amount"].sum())
attempts_cnt   = int(attempts_mask.sum())
approved_cnt   = int(approved_purchases.sum())
declined_cnt   = int(declined_purchases.sum())
settled_cnt    = int(settled_purchases.sum())
approval_rate  = safe_div(approved_cnt, attempts_cnt)
decline_rate   = safe_div(declined_cnt, attempts_cnt)
aov_settled    = safe_div(net_settled, settled_cnt)
refund_total   = float(f.loc[refunds_f, "Settle Amount"].sum())

# KPI cards
def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(f"### 📊 Merchant Dashboard")
st.caption(f"Merchant: **{merchant_id_value}** • Date: {start_date} → {end_date}")

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
kpi_card("Gross Requests", f"R {gross_requests:,.0f}")
kpi_card("Net Settled", f"R {net_settled:,.0f}")
kpi_card("# Attempts", f"{attempts_cnt:,}")
kpi_card("Approval Rate", f"{(approval_rate*100):.1f}%" if not math.isnan(approval_rate) else "—")
kpi_card("Decline Rate", f"{(decline_rate*100):.1f}%" if not math.isnan(decline_rate) else "—")
kpi_card("Average Order Value (AOV)", f"R {aov_settled:,.2f}" if not math.isnan(aov_settled) else "—")
st.markdown('</div>', unsafe_allow_html=True)

if refund_total:
    st.caption(f"Refunds in range (excluded from KPIs): **R {refund_total:,.0f}**")

# =========================
# Row: Trends (left) + Issuing Bank Pie (right)
# =========================
c1, c2 = st.columns((2,1), gap="small")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Daily Net Settled & Approval Rate")
    df_day = (
        f.loc[attempts_mask, ["Transaction Date", "Settle Amount", "is_approved", "is_settled"]]
         .assign(date=lambda d: d["Transaction Date"].dt.date)
         .groupby("date", as_index=False)
         .agg(
             attempts=("is_settled", "count"),
             approved=("is_approved", "sum"),
             net_settled=("Settle Amount", "sum"),
         )
    )
    if not df_day.empty:
        df_day["approval_rate"] = df_day.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df_day["date"], y=df_day["net_settled"], name="Net Settled", marker_color=PRIMARY, opacity=0.75))
        fig1.add_trace(go.Scatter(x=df_day["date"], y=df_day["approval_rate"], yaxis="y2", name="Approval Rate", mode="lines+markers"))
        fig1.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Net Settled",
            yaxis2=dict(title="Approval Rate", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No purchase attempts in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Issuing Bank Mix (Net Settled)")
    issuer_df = (
        f.loc[settled_purchases, ["Issuing Bank", "Settle Amount"]]
         .groupby("Issuing Bank", as_index=False)
         .agg(net_settled=("Settle Amount", "sum"))
         .sort_values("net_settled", ascending=False)
    )
    if not issuer_df.empty:
        fig_pie = px.pie(
            issuer_df, values="net_settled", names="Issuing Bank",
            hole=0.45, title="", template="plotly_white",
        )
        # green + grey palette
        fig_pie.update_traces(texttemplate="%{label}<br>%{percent:.0%}", textposition="inside")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No settled revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Row: Top Decline Reasons (left) + Devices Decline Rate (right)
# =========================
c3, c4 = st.columns(2, gap="small")

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Decline Reasons (as % of Attempts)")
    # compute against attempts to give a stable denominator
    base_attempts = int(attempts_mask.sum())
    decl_df = (
        f.loc[declined_purchases, ["Decline Reason"]]
         .value_counts()
         .reset_index(name="count")
         .rename(columns={"index": "Decline Reason"})
         .sort_values("count", ascending=True)
    )
    if not decl_df.empty and base_attempts > 0:
        decl_df["pct_of_attempts"] = decl_df["count"] / base_attempts
        fig_decl = px.bar(
            decl_df, x="pct_of_attempts", y="Decline Reason",
            orientation="h", template="plotly_white",
        )
        fig_decl.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        fig_decl.update_xaxes(tickformat=".0%")
        fig_decl.update_traces(marker_color="#94a3b8", texttemplate="%{x:.0%}", textposition="outside")
        st.plotly_chart(fig_decl, use_container_width=True)
    else:
        st.info("No declines in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Devices by Decline Rate (worst 10)")
    by_dev = (
        f.loc[attempts_mask, ["Device Serial", "Terminal ID", "is_approved"]]
         .assign(attempt=1)
         .groupby(["Device Serial", "Terminal ID"], as_index=False)
         .agg(attempts=("attempt", "sum"), approved=("is_approved", "sum"))
    )
    if not by_dev.empty:
        by_dev["decline_rate"] = 1 - by_dev.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)
        by_dev = by_dev.sort_values("decline_rate", ascending=False).head(10)
        # Make labels more legible: show DeviceSerial – Terminal
        by_dev["DeviceLabel"] = by_dev["Device Serial"].astype(str) + " — " + by_dev["Terminal ID"].astype(str)
        fig_dev = px.bar(
            by_dev, x="decline_rate", y="DeviceLabel",
            orientation="h", template="plotly_white",
            hover_data={"attempts": True, "approved": True, "decline_rate": ":.1%"},
        )
        fig_dev.update_traces(marker_color="#64748b", texttemplate="%{x:.0%}", textposition="outside")
        fig_dev.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(by_dev["decline_rate"].max()) * 1.15)])
        fig_dev.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_dev, use_container_width=True)
    else:
        st.info("No device attempts in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Detail table + download
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Transactions (filtered)")

show_cols = [
    "Transaction Date", "Request Amount", "Settle Amount",
    "Transaction Type", "Decline Reason", "Auth Code",
    "Issuing Bank", "BIN", "Product Type",
    "Terminal ID", "Device Serial",
    "Date Payment Extract", "System Batch Number", "Device Batch Number",
    "System Trace Audit Number", "Retrieval Reference", "UTI", "Online Reference Number",
]
existing_cols = [c for c in show_cols if c in f.columns]
tbl = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
st.dataframe(tbl, use_container_width=True, height=440)

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered transactions (CSV)",
    data=to_csv_bytes(tbl),
    file_name="filtered_transactions.csv",
    mime="text/csv",
)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footnote
# =========================
with st.expander("About the metrics"):
    st.write(
        """
- **Attempts**: all **Purchase** rows (approved + declined).
- **Approved**: Decline Reason starts with `"00"` or an Auth Code exists.
- **Settled**: `Date Payment Extract` present **and** `Settle Amount` ≠ 0 (purchases).
- **AOV (Average Order Value)**: Net Settled ÷ # Settled (purchases only).
- **Refunds** are tracked separately and excluded from KPIs.
"""
    )
