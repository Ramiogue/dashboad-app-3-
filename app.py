import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page & style
# =========================
st.set_page_config(
    page_title="Merchant Portal",
    page_icon="📊",
    layout="wide",
)

# =========================
# Auth (from Secrets)
# =========================
# Secrets example:
# COOKIE_KEY = "replace_with_random_secret"
#
# [users."M001"]
# name = "Merchant A"
# email = "a@example.com"
# password_hash = "$2b$12$D9ZTRh3DUXkdIibFhes.7eKWQLTc.cdJJwtOsFTLbBuxkanfkkbJm"  # for M001@123
# merchant_id = "M001 - Merchant A"  # must match a value in your chosen merchant_id_col
#
# [users."M002"]
# name = "Merchant B"
# email = "b@example.com"
# password_hash = "$2b$12$rGu5g8HsKr.0dqCUh1ksLOTQs6AttYRV/xWC/k99Ru3x5hRFJMU8O"  # for M002@123
# merchant_id = "M002 - Merchant B"
#
# # Optional: choose which column identifies a merchant in the CSV
# merchant_id_col = "Merchant Number - Business Name"  # or "Device Serial"

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
# Load & prepare data
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
    "Terminal ID", "Device Serial", "Product Type", "Issuing Bank", "BIN", "Pos Entry Mode"
}
missing = required_cols - set(tx.columns)
if missing:
    st.error(f"Missing required column(s) in CSV: {', '.join(sorted(missing))}")
    st.stop()

# Types & cleaning
tx[MERCHANT_ID_COL] = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"] = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"] = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"] = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"] = tx["Date Payment Extract"].astype(str).fillna("")

for c in ["Product Type", "Issuing Bank", "Decline Reason", "Pos Entry Mode", "Terminal ID", "Device Serial"]:
    tx[c] = tx[c].astype(str).fillna("")

# Filter to this merchant first (RLS analogue)
merchant_tx = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if merchant_tx.empty:
    st.warning(
        f"No transactions found for merchant value '{merchant_id_value}' in column '{MERCHANT_ID_COL}'.\n"
        "Check that your Secrets 'merchant_id' matches CSV values (exact case & spaces)."
    )
    st.stop()

# =========================
# Derived flags & business logic
# =========================
def is_purchase(s: pd.Series) -> pd.Series:
    return s.str.lower().eq("purchase")

def is_refund(s: pd.Series) -> pd.Series:
    return s.str.lower().eq("refund")

def is_reversal(s: pd.Series) -> pd.Series:
    return s.str.lower().eq("reversal")

def approved_mask(df: pd.DataFrame) -> pd.Series:
    # Treat "00 - Approved or completed successfully" OR presence of Auth Code as approved
    dr = df["Decline Reason"].str.strip()
    return dr.str.startswith("00") | (df["Auth Code"].astype(str).str.len() > 0)

def settled_mask(df: pd.DataFrame) -> pd.Series:
    # Settled if a payment extract date exists AND Settle Amount != 0
    has_extract = df["Date Payment Extract"].str.len() > 0
    nonzero = df["Settle Amount"].fillna(0) != 0
    return has_extract & nonzero

merchant_tx["is_purchase"] = is_purchase(merchant_tx["Transaction Type"])
merchant_tx["is_refund"] = is_refund(merchant_tx["Transaction Type"])
merchant_tx["is_reversal"] = is_reversal(merchant_tx["Transaction Type"])

merchant_tx["is_approved"] = approved_mask(merchant_tx)
merchant_tx["is_settled"] = settled_mask(merchant_tx)

# Attempts: all purchase attempts (approved + declined)
attempts_mask = merchant_tx["is_purchase"]

# Approved/Declined within purchase attempts
approved_purchases = attempts_mask & merchant_tx["is_approved"]
declined_purchases = attempts_mask & (~merchant_tx["is_approved"])

# Settled purchases only (exclude refunds)
settled_purchases = attempts_mask & merchant_tx["is_settled"]

# =========================
# Sidebar filters
# =========================
st.sidebar.subheader("Filters")

min_d = pd.to_datetime(merchant_tx["Transaction Date"].min()).date()
max_d = pd.to_datetime(merchant_tx["Transaction Date"].max()).date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_d, max_d

def _multi(label, series, sort=True):
    opts = sorted(series.unique()) if sort else list(series.unique())
    sel = st.sidebar.multiselect(label, options=opts, default=opts)
    return sel

sel_declines = _multi("Decline Reason", merchant_tx["Decline Reason"])
sel_prodtype = _multi("Product Type", merchant_tx["Product Type"])
sel_issuer = _multi("Issuing Bank", merchant_tx["Issuing Bank"])
sel_pos = _multi("POS Entry Mode", merchant_tx["Pos Entry Mode"])

# Apply filters (date first)
flt = (merchant_tx["Transaction Date"].dt.date >= start_date) & (merchant_tx["Transaction Date"].dt.date <= end_date)
flt &= merchant_tx["Decline Reason"].isin(sel_declines)
flt &= merchant_tx["Product Type"].isin(sel_prodtype)
flt &= merchant_tx["Issuing Bank"].isin(sel_issuer)
flt &= merchant_tx["Pos Entry Mode"].isin(sel_pos)

f = merchant_tx[flt].copy()

attempts_mask_f = f["is_purchase"]
approved_purchases_f = attempts_mask_f & f["is_approved"]
declined_purchases_f = attempts_mask_f & (~f["is_approved"])
settled_purchases_f  = attempts_mask_f & f["is_settled"]
refunds_f            = f["is_refund"]

# =========================
# Top KPIs
# =========================
def safe_div(n, d):
    return (n / d) if d else np.nan

gross_requests = float(f.loc[attempts_mask_f, "Request Amount"].sum())
net_settled    = float(f.loc[settled_purchases_f, "Settle Amount"].sum())
attempts_cnt   = int(attempts_mask_f.sum())
approved_cnt   = int(approved_purchases_f.sum())
declined_cnt   = int(declined_purchases_f.sum())
settled_cnt    = int(settled_purchases_f.sum())
approval_rate  = safe_div(approved_cnt, attempts_cnt)
decline_rate   = safe_div(declined_cnt, attempts_cnt)
aov_settled    = safe_div(net_settled, settled_cnt)  # Average Order Value (settled only)
refund_total   = float(f.loc[refunds_f, "Settle Amount"].sum())

st.title("📊 Merchant Dashboard")
st.caption(f"Merchant: **{merchant_id_value}**  •  Source: `{f['__path__'].iat[0]}`  •  Date: {start_date} → {end_date}")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Gross Requests", f"R {gross_requests:,.0f}")
k2.metric("Net Settled", f"R {net_settled:,.0f}")
k3.metric("# Attempts", f"{attempts_cnt:,}")
k4.metric("Approval Rate", f"{(approval_rate*100):.1f}%" if not math.isnan(approval_rate) else "—")
k5.metric("Decline Rate", f"{(decline_rate*100):.1f}%" if not math.isnan(decline_rate) else "—")
k6.metric("Average Order Value (AOV)", f"R {aov_settled:,.2f}" if not math.isnan(aov_settled) else "—")

if refund_total:
    st.caption(f"Refunds in range: **R {refund_total:,.0f}** (shown separately, excluded from AOV/Net Settled KPIs).")

# =========================
# Trends
# =========================
st.subheader("Trends")

# Daily aggregates (purchases only)
df_day = (
    f.loc[attempts_mask_f, ["Transaction Date", "Request Amount", "Settle Amount", "is_approved", "is_settled"]]
    .assign(date=lambda d: d["Transaction Date"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        attempts=("Request Amount", "count"),
        approved=("is_approved", "sum"),
        settled=("is_settled", "sum"),
        net_settled=("Settle Amount", "sum"),
    )
)
if not df_day.empty:
    df_day["approval_rate"] = df_day.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)

    # Net settled line
    fig1 = px.line(df_day, x="date", y="net_settled", title="Daily Net Settled")
    st.plotly_chart(fig1, use_container_width=True)

    # Approval rate line
    fig2 = px.line(df_day, x="date", y="approval_rate", title="Daily Approval Rate")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No purchase attempts in the selected period.")

# =========================
# Funnel (Attempts → Approved → Settled)
# =========================
st.subheader("Payments Funnel")
funnel_vals = [attempts_cnt, approved_cnt, settled_cnt]
funnel_labels = ["Attempts", "Approved", "Settled"]
fig_funnel = go.Figure(go.Funnel(y=funnel_labels, x=funnel_vals))
st.plotly_chart(fig_funnel, use_container_width=True)

# =========================
# Declines by reason
# =========================
st.subheader("Top Decline Reasons")
decl_df = (
    f.loc[declined_purchases_f, ["Decline Reason"]]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "Decline Reason"})
    .sort_values("count", ascending=True)
)
if decl_df.empty:
    st.write("No declines in the selected period.")
else:
    fig_decl = px.bar(decl_df, x="count", y="Decline Reason", orientation="h", title="Declines by Reason")
    st.plotly_chart(fig_decl, use_container_width=True)

# =========================
# Issuer mix (volume vs approval)
# =========================
st.subheader("Issuing Bank — Volume vs Approval")
issuer_df = (
    f.loc[attempts_mask_f, ["Issuing Bank", "Request Amount", "is_approved", "is_settled", "Settle Amount"]]
    .groupby("Issuing Bank", as_index=False)
    .agg(
        attempts=("Request Amount", "count"),
        approved=("is_approved", "sum"),
        net_settled=("Settle Amount", "sum"),
    )
)
if not issuer_df.empty:
    issuer_df["approval_rate"] = issuer_df.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)
    fig_ib = px.scatter(
        issuer_df,
        x="approval_rate", y="net_settled", size="attempts", hover_name="Issuing Bank",
        title="Issuing Bank: Approval Rate vs Net Settled (bubble size = attempts)"
    )
    fig_ib.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_ib, use_container_width=True)

# =========================
# Device reliability (by Terminal or Device Serial)
# =========================
st.subheader("Devices (Decline Rate)")
by_dev = (
    f.loc[attempts_mask_f, ["Terminal ID", "Device Serial", "is_approved"]]
    .assign(attempt=1)
    .groupby(["Device Serial", "Terminal ID"], as_index=False)
    .agg(
        attempts=("attempt", "sum"),
        approved=("is_approved", "sum"),
    )
)
if not by_dev.empty:
    by_dev["decline_rate"] = 1 - by_dev.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)
    by_dev = by_dev.sort_values("decline_rate", ascending=False).head(15)
    fig_dev = px.bar(
        by_dev,
        x="decline_rate", y="Terminal ID",
        hover_data=["Device Serial", "attempts", "approved"],
        orientation="h",
        title="Top Devices by Decline Rate (worst 15)"
    )
    fig_dev.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_dev, use_container_width=True)

# =========================
# Product mix (Credit/Debit/Unknown)
# =========================
st.subheader("Card Product Type Mix")
mix = (
    f.loc[attempts_mask_f & f["is_approved"], ["Product Type", "Settle Amount"]]
    .groupby("Product Type", as_index=False)
    .agg(net_settled=("Settle Amount", "sum"))
    .sort_values("net_settled", ascending=False)
)
if not mix.empty:
    fig_mix = px.bar(mix, x="Product Type", y="net_settled", title="Volume by Product Type (Approved/Settled)")
    st.plotly_chart(fig_mix, use_container_width=True)

# =========================
# Detail table & export
# =========================
st.subheader("Transactions (filtered)")
show_cols = [
    "Transaction Date", "Request Amount", "Settle Amount",
    "Transaction Type", "Decline Reason", "Auth Code",
    "Issuing Bank", "BIN", "Product Type",
    "Pos Entry Mode", "Terminal ID", "Device Serial",
    "Date Payment Extract", "System Batch Number", "Device Batch Number",
    "System Trace Audit Number", "Retrieval Reference", "UTI", "Online Reference Number",
]
existing_cols = [c for c in show_cols if c in f.columns]
tbl = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
st.dataframe(tbl, use_container_width=True, height=420)

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered transactions (CSV)",
    data=to_csv_bytes(tbl),
    file_name="filtered_transactions.csv",
    mime="text/csv",
)

# =========================
# Footer / diagnostics
# =========================
with st.expander("About this dashboard"):
    st.write(
        f"""
- **AOV (Average Order Value)** = Net Settled ÷ # Settled (purchases only).
- **Attempts** = all **Purchase** transactions (approved + declined).
- **Approved** = Decline Reason starts with **"00"** or Auth Code present.
- **Settled** = Date Payment Extract present and Settle Amount ≠ 0 (purchases only).
- **Refunds** shown separately and excluded from AOV / Net Settled KPIs.
- Merchant identity column: **{MERCHANT_ID_COL}** → value: **{merchant_id_value}**.
"""
    )
