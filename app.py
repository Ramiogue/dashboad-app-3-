import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page & palette
# =========================
st.set_page_config(page_title="Merchant Dashboard", page_icon=None, layout="wide")

PRIMARY = "#0B6E4F"      # brand green
GREEN_2 = "#149E67"
GREY_BG = "#f5f7fb"      # app bg
CARD_BG = "#ffffff"      # card bg
TEXT = "#1f2937"
GREY_BAR = "#94a3b8"
GREY_BAR_DARK = "#64748b"

def apply_plotly_layout(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        title_x=0.01,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# =========================
# Global CSS (Power BI style + login inputs fix + equal KPI heights)
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{ background: {GREY_BG}; }}
    .block-container {{ padding-top: 0.8rem; padding-bottom: 1.2rem; }}

    /* Header */
    .header-row {{
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom: 0.25rem;
    }}
    .title-left h1 {{ font-size: 1.15rem; margin: 0; color: {TEXT}; }}

    /* KPI cards (uniform height, single row via st.columns) */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid #e5e7eb;
        border-left: 4px solid {PRIMARY};
        border-radius: 12px;
        padding: 8px 10px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);

        /* SAME HEIGHT FOR ALL */
        height: 84px;                /* fixed height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 2px;
        overflow: hidden;            /* no wrap-induced growth */
    }}
    .kpi-title {{ font-size: 0.72rem; color: #6b7280; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi-value {{ font-size: 1.25rem; font-weight: 800; color: {TEXT}; line-height: 1.05; margin: 0; }}
    .kpi-sub   {{ font-size: 0.75rem; color: #6b7280; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

    .card {{
        background: {CARD_BG};
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        margin-bottom: 10px;
    }}
    .card h3 {{ margin-top: 0.2rem; color: {TEXT}; font-size: 1.0rem; }}

    /* Login inputs: white boxes + green focus */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPassword"] input,
    .stTextInput input,
    .stPassword input,
    div[data-baseweb="input"] input {{
      background-color: #ffffff !important;
      color: #111827 !important;
      border: 1px solid #cbd5e1 !important;
      border-radius: 10px !important;
      padding: 10px 12px !important;
      box-shadow: none !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stPassword"] input:focus,
    .stTextInput input:focus,
    .stPassword input:focus,
    div[data-baseweb="input"] input:focus {{
      border: 1.5px solid {PRIMARY} !important;
      outline: none !important;
      box-shadow: 0 0 0 3px rgba(11,110,79,0.10) !important;
    }}
    div[data-testid="stTextInput"] label,
    div[data-testid="stPassword"] label {{
      margin-bottom: 6px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Auth (from Secrets)
# =========================
# Secrets example (TOML):
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

creds = {"usernames": {}}
for uname, u in users_cfg.items():
    creds["usernames"][uname] = {"name": u["name"], "email": u["email"], "password": u["password_hash"]}

authenticator = stauth.Authenticate(
    credentials=creds,
    cookie_name="merchant_portal",
    key=cookie_key,
    cookie_expiry_days=7,
)

# Optional: put login inside a card for a cleaner frame
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    authenticator.login(location="main")
    st.markdown('</div>', unsafe_allow_html=True)

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
    if uname in cfg: return cfg[uname]
    uname_cf = str(uname).casefold()
    for k, v in cfg.items():
        if str(k).casefold() == uname_cf: return v
    return None

merchant_rec = get_user_record(users_cfg, username)
if not merchant_rec or "merchant_id" not in merchant_rec:
    st.error("Merchant mapping not found for this user. Check Secrets for 'merchant_id'.")
    st.stop()
merchant_id_value = merchant_rec["merchant_id"]

# =========================
# Load & prep data
# =========================
@st.cache_data(ttl=60)
def load_transactions():
    for p in ("sample_merchant_transactions.csv", "data/sample_merchant_transactions.csv"):
        try:
            df = pd.read_csv(p); df["__path__"] = p; return df
        except Exception: pass
    raise FileNotFoundError("CSV not found. Put 'sample_merchant_transactions.csv' at repo root or under 'data/'.")

tx = load_transactions()
required_cols = {
    MERCHANT_ID_COL, "Transaction Date","Request Amount","Settle Amount",
    "Transaction Type","Auth Code","Decline Reason","Date Payment Extract",
    "Terminal ID","Device Serial","Product Type","Issuing Bank","BIN"
}
missing = required_cols - set(tx.columns)
if missing:
    st.error(f"Missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

tx[MERCHANT_ID_COL] = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"] = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"] = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"] = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"] = tx["Date Payment Extract"].astype(str).fillna("")
for c in ["Product Type","Issuing Bank","Decline Reason","Terminal ID","Device Serial"]:
    tx[c] = tx[c].astype(str).fillna("")

f0 = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if f0.empty:
    st.warning(f"No transactions for '{merchant_id_value}' in '{MERCHANT_ID_COL}'.")
    st.stop()

def is_purchase(s): return s.str.lower().eq("purchase")
def is_refund(s):  return s.str.lower().eq("refund")
def is_reversal(s):return s.str.lower().eq("reversal")
def approved_mask(df):
    dr = df["Decline Reason"].str.strip()
    return dr.str.startswith("00") | (df["Auth Code"].astype(str).str.len() > 0)
def settled_mask(df):
    has_extract = df["Date Payment Extract"].str.len() > 0
    nonzero = df["Settle Amount"].fillna(0) != 0
    return has_extract & nonzero

f0["is_purchase"] = is_purchase(f0["Transaction Type"])
f0["is_refund"]   = is_refund(f0["Transaction Type"])
f0["is_reversal"] = is_reversal(f0["Transaction Type"])
f0["is_approved"] = approved_mask(f0)
f0["is_settled"]  = settled_mask(f0)

# =========================
# Sidebar filters (POS Entry removed)
# =========================
st.sidebar.subheader("Filters")
min_d = pd.to_datetime(f0["Transaction Date"].min()).date()
max_d = pd.to_datetime(f0["Transaction Date"].max()).date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_d, max_d))

def _multi(label, series):
    opts = sorted(series.unique())
    return st.sidebar.multiselect(label, options=opts, default=opts)

sel_declines = _multi("Decline Reason", f0["Decline Reason"])
sel_prodtype = _multi("Product Type",    f0["Product Type"])
sel_issuer   = _multi("Issuing Bank",    f0["Issuing Bank"])

flt = (f0["Transaction Date"].dt.date >= start_date) & (f0["Transaction Date"].dt.date <= end_date)
flt &= f0["Decline Reason"].isin(sel_declines)
flt &= f0["Product Type"].isin(sel_prodtype)
flt &= f0["Issuing Bank"].isin(sel_issuer)
f = f0[flt].copy()

# =========================
# KPIs (single row via st.columns, equal heights via CSS above)
# =========================
def safe_div(n, d): return (n / d) if d else np.nan

# “# Transactions” = ALL rows (Purchases + Refunds + Reversals)
transactions_cnt = int(len(f))

# Purchase-flow metrics (for funnel/ratios)
attempts_mask     = f["is_purchase"]
approved_purchases= attempts_mask & f["is_approved"]
declined_purchases= attempts_mask & (~f["is_approved"])
settled_purchases = attempts_mask & f["is_settled"]
refunds_f         = f["is_refund"]

gross_requests = float(f.loc[attempts_mask, "Request Amount"].sum())
net_settled    = float(f.loc[settled_purchases, "Settle Amount"].sum())
settled_cnt    = int(settled_purchases.sum())
approved_cnt   = int(approved_purchases.sum())
attempts_cnt   = int(attempts_mask.sum())  # for funnel only
approval_rate  = safe_div(approved_cnt, attempts_cnt)
decline_rate   = safe_div(int(declined_purchases.sum()), attempts_cnt)
aov_settled    = safe_div(net_settled, settled_cnt)

refund_total   = float(f.loc[refunds_f, "Settle Amount"].sum())  # negative
net_after_ref  = net_settled + refund_total

# Optional header image
header_path = None
for p in ("assets/header.jpg","assets/header.png","assets/header.jpeg","assets/header.webp"):
    if os.path.exists(p): header_path = p; break
if header_path:
    st.image(header_path, use_column_width=True)
else:
    st.markdown('<div class="header-row"><div class="title-left"><h1>Merchant Dashboard</h1></div></div>', unsafe_allow_html=True)

st.caption(f"Merchant: **{merchant_id_value}**  •  Source: `{f['__path__'].iat[0]}`  •  Date: {start_date} → {end_date}")

def kpi_card(title, value, sub=""):
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

cols = st.columns(6, gap="small")
with cols[0]:
    kpi_card("# Transactions", f"{transactions_cnt:,}")
with cols[1]:
    kpi_card("Gross Requests", f"R {gross_requests:,.0f}")
with cols[2]:
    kpi_card("Net Settled", f"R {net_settled:,.0f}")
with cols[3]:
    kpi_card("Refunds", f"R {refund_total:,.0f}", "Negative")
with cols[4]:
    kpi_card("Net After Refunds", f"R {net_after_ref:,.0f}")
with cols[5]:
    kpi_card("Average Order Value (AOV)", f"R {aov_settled:,.2f}" if not math.isnan(aov_settled) else "—")

# =========================
# Row 1: Trend + Approvals  |  Issuing Bank Donut
# =========================
c1, c2 = st.columns((2,1), gap="small")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Daily Net Settled & Approval Rate")
    df_day_base = f.loc[attempts_mask, ["Transaction Date", "Settle Amount", "is_approved"]].copy()
    if not df_day_base.empty:
        df_day_base["date"] = df_day_base["Transaction Date"].dt.date
        df_day = df_day_base.groupby("date", as_index=False).agg(
            attempts=("Transaction Date", "count"),
            approved=("is_approved", "sum"),
            net_settled=("Settle Amount", "sum"),
        )
        df_day["approval_rate"] = df_day.apply(lambda r: safe_div(r["approved"], r["attempts"]), axis=1)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df_day["date"], y=df_day["net_settled"], name="Net Settled", marker_color=PRIMARY, opacity=0.85))
        fig1.add_trace(go.Scatter(x=df_day["date"], y=df_day["approval_rate"], yaxis="y2", name="Approval Rate", mode="lines+markers", line=dict(color=GREY_BAR_DARK)))
        fig1.update_layout(yaxis_title="Net Settled", yaxis2=dict(title="Approval Rate", overlaying="y", side="right", tickformat=".0%"))
        apply_plotly_layout(fig1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No purchase attempts in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Issuing Bank Mix (Net Settled)")
    issuer_df = (
        f.loc[settled_purchases, ["Issuing Bank", "Settle Amount"]]
         .groupby("Issuing Bank", as_index=False)
         .agg(net_settled=("Settle Amount", "sum"))
         .sort_values("net_settled", ascending=False)
    )
    if not issuer_df.empty:
        fig_pie = px.pie(issuer_df, values="net_settled", names="Issuing Bank", hole=0.5)
        fig_pie.update_traces(texttemplate="%{label}<br>%{percent:.0%}", textposition="inside",
                              marker=dict(line=dict(color="white", width=1)))
        apply_plotly_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No settled revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Row 2: Top Decline Reasons (full width)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Top Decline Reasons (as % of Attempts)")
base_attempts = int(attempts_mask.sum())
decl_df = (
    f.loc[attempts_mask & (~f["is_approved"]), ["Decline Reason"]]
     .value_counts()
     .reset_index(name="count")
     .rename(columns={"index": "Decline Reason"})
     .sort_values("count", ascending=True)
)
if not decl_df.empty and base_attempts > 0:
    decl_df["pct_of_attempts"] = decl_df["count"] / base_attempts
    fig_decl = px.bar(decl_df, x="pct_of_attempts", y="Decline Reason", orientation="h")
    fig_decl.update_traces(marker_color=GREY_BAR, texttemplate="%{x:.0%}", textposition="outside")
    fig_decl.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(decl_df["pct_of_attempts"].max()) * 1.15)])
    apply_plotly_layout(fig_decl)
    st.plotly_chart(fig_decl, use_container_width=True)
else:
    st.info("No declines in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Row 3: Payments Funnel + Table
# =========================
c5, c6 = st.columns((1,2), gap="small")

with c5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Payments Funnel")
    funnel_vals = [attempts_cnt, approved_cnt, settled_cnt]
    funnel_labels = ["Attempts", "Approved", "Settled"]
    fig_funnel = go.Figure(go.Funnel(y=funnel_labels, x=funnel_vals, marker={"color":[GREY_BAR, GREEN_2, PRIMARY]}))
    apply_plotly_layout(fig_funnel)
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Transactions (filtered)")
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

    st.download_button("Download filtered transactions (CSV)", data=to_csv_bytes(tbl),
                       file_name="filtered_transactions.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footnote
# =========================
with st.expander("About the metrics"):
    st.write(
        """
- **# Transactions**: all rows in range (Purchases + Refunds + Reversals).
- **Attempts** (for funnel only): all **Purchase** rows (approved + declined).
- **Approved**: Decline Reason starts with `"00"` or an Auth Code exists.
- **Settled**: `Date Payment Extract` present **and** `Settle Amount` ≠ 0 (purchases).
- **AOV (Average Order Value)**: Net Settled ÷ # Settled (purchases only).
- **Refunds** are negative amounts and shown as their own KPI; **Net After Refunds** = Net Settled + Refunds.
"""
    )
