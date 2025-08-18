import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth

# =========================
# Page
# =========================
st.set_page_config(page_title="Merchant Dashboard", page_icon=None, layout="wide")

# =========================
# Professional Theme Tokens
# =========================
PRIMARY = "#0B6E4F"      # single accent (brand green)
TEXT    = "#0f172a"      # slate-900
CARD_BG = "#ffffff"

GREY_50  = "#f8fafc"     # page bg
GREY_100 = "#f1f5f9"     # light section bg
GREY_200 = "#e2e8f0"     # borders
GREY_300 = "#cbd5e1"
GREY_400 = "#94a3b8"

# Sidebar & filter panel tones (darker than page)
SIDEBAR_BG         = "#eef2f6"
FILTER_HDR_BG_DEF  = "#e6ebf2"   # expander header (collapsed)
FILTER_HDR_BG_OPEN = "#dde4ee"   # expander header (open)
FILTER_CNT_BG_OPEN = "#d6dde9"   # expander content (open)

# Semantic
DANGER  = "#dc2626"      # declines / errors
INFO    = "#2563eb"

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
# Global CSS (polished)
# =========================
st.markdown(
    f"""
    <style>
    /* Canvas */
    .stApp {{ background: {GREY_50}; }}
    .block-container {{ padding-top: .8rem; padding-bottom: 1.2rem; }}

    /* Typography */
    html, body, [class^="css"] {{ color: {TEXT}; }}

    /* Header */
    .header-row {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.25rem; }}
    .title-left h1 {{ font-size:1.15rem; margin:0; color:{TEXT}; }}

    /* Cards */
    .card {{
      background:{CARD_BG};
      border:1px solid {GREY_200};
      border-radius:12px;
      padding:12px;
      box-shadow:0 1px 2px rgba(2,6,23,0.04);
      margin-bottom:10px;
    }}
    .card h3 {{ margin-top:.2rem; color:{TEXT}; font-size:1.0rem; }}

    /* KPI cards */
    .kpi-card {{
      background:{CARD_BG};
      border:1px solid {GREY_200};
      border-left:4px solid {PRIMARY};
      border-radius:12px;
      padding:10px 12px;
      box-shadow:0 1px 2px rgba(2,6,23,0.04);
      height:84px; display:flex; flex-direction:column; justify-content:center; gap:2px; overflow:hidden;
    }}
    .kpi-title {{ font-size:.72rem; color:{GREY_400}; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .kpi-value {{ font-size:1.25rem; font-weight:800; color:{TEXT}; line-height:1.05; margin:0; }}
    .kpi-sub   {{ font-size:.75rem; color:{GREY_400}; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPassword"] input,
    .stTextInput input, .stPassword input,
    div[data-baseweb="input"] input {{
      background:#fff !important; color:{TEXT} !important;
      border:1px solid {GREY_300} !important; border-radius:10px !important; padding:10px 12px !important;
      box-shadow:none !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stPassword"] input:focus,
    .stTextInput input:focus, .stPassword input:focus,
    div[data-baseweb="input"] input:focus {{
      border:1.5px solid {PRIMARY} !important;
      outline:none !important; box-shadow:0 0 0 3px rgba(11,110,79,.10) !important;
    }}
    div[data-testid="stTextInput"] label,
    div[data-testid="stPassword"] label {{ margin-bottom:6px !important; color:{GREY_400}; }}

    /* Sidebar: darker than canvas */
    [data-testid="stSidebar"] {{
      background:{SIDEBAR_BG};
      box-shadow: inset -1px 0 0 {GREY_200};
    }}

    /* Filters expander frame */
    [data-testid="stSidebar"] details {{
      border:1px solid {GREY_200}; border-radius:12px; overflow:hidden;
    }}

    /* Expander header (collapsed vs open) */
    [data-testid="stSidebar"] details > summary.streamlit-expanderHeader {{
      background:{FILTER_HDR_BG_DEF}; color:{TEXT}; font-weight:700; padding:8px 12px; list-style:none;
    }}
    [data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader {{
      background:{FILTER_HDR_BG_OPEN}; border-bottom:1px solid {GREY_200};
    }}

    /* Expander content (open) */
    [data-testid="stSidebar"] details[open] .streamlit-expanderContent {{
      background:{FILTER_CNT_BG_OPEN}; padding:8px 12px;
    }}

    /* Multiselect chips */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{ border-radius:8px; }}

    /* Sidebar date input pill */
    [data-testid="stSidebar"] .stDateInput input {{
      background:#fff !important; border:1px solid {GREY_300} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Auth (from Secrets)
# =========================
# COOKIE_KEY = "replace_with_random_secret"
# merchant_id_col = "Device Serial"  # or "Merchant Number - Business Name"
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

# Optional: login inside a card
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
        except Exception:
            pass
    raise FileNotFoundError("CSV not found. Put 'sample_merchant_transactions.csv' at repo root or under 'data/'.")

tx = load_transactions()
required_cols = {
    MERCHANT_ID_COL, "Transaction Date","Request Amount","Settle Amount",
    "Auth Code","Decline Reason","Date Payment Extract",
    "Terminal ID","Device Serial","Product Type","Issuing Bank","BIN"
}
missing = required_cols - set(tx.columns)
if missing:
    st.error(f"Missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

# Robust cleaning (avoid literal "nan" strings)
tx[MERCHANT_ID_COL]      = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"]   = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"]     = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"]      = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"]= tx["Date Payment Extract"].fillna("").astype(str)

for c in ["Product Type","Issuing Bank","Decline Reason","Terminal ID","Device Serial","Auth Code"]:
    tx[c] = tx[c].fillna("").astype(str)

f0 = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if f0.empty:
    st.warning(f"No transactions for '{merchant_id_value}' in '{MERCHANT_ID_COL}'.")
    st.stop()

# Approval & Settlement
def approved_mask(df):
    dr = df["Decline Reason"].astype(str).str.strip()
    return dr.str.startswith("00") | (df["Auth Code"].astype(str).str.strip().ne(""))

def settled_mask(df):
    has_extract = df["Date Payment Extract"].astype(str).str.strip().ne("")
    nonzero = pd.to_numeric(df["Settle Amount"], errors="coerce").fillna(0).ne(0)
    return has_extract & nonzero

f0["is_approved"] = approved_mask(f0)
f0["is_settled"]  = settled_mask(f0)

# =========================
# Sidebar filters (collapsible, darker when open)
# =========================
with st.sidebar.expander("Filters", expanded=True):
    valid_dates = f0["Transaction Date"].dropna()
    min_d = pd.to_datetime(valid_dates.min()).date()
    max_d = pd.to_datetime(valid_dates.max()).date()

    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    def _multi(label, series):
        opts = sorted(series.astype(str).unique())
        return st.multiselect(label, options=opts, default=opts)

    sel_declines = _multi("Decline Reason", f0["Decline Reason"])
    sel_prodtype = _multi("Product Type",    f0["Product Type"])
    sel_issuer   = _multi("Issuing Bank",    f0["Issuing Bank"])

start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_d, max_d))

flt = (f0["Transaction Date"].dt.date >= start_date) & (f0["Transaction Date"].dt.date <= end_date)
flt &= f0["Decline Reason"].isin(sel_declines)
flt &= f0["Product Type"].isin(sel_prodtype)
flt &= f0["Issuing Bank"].isin(sel_issuer)
f = f0[flt].copy()
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# =========================
# KPIs
# =========================
def safe_div(n, d): return (n / d) if d else np.nan

transactions_cnt = int(len(f))
attempts_cnt  = int(len(f))
approved_cnt  = int(f["is_approved"].sum())
approval_rate = safe_div(approved_cnt, attempts_cnt)
decline_rate  = safe_div(attempts_cnt - approved_cnt, attempts_cnt)

settled_rows = f["is_settled"]
revenue      = float(f.loc[settled_rows, "Setle Amount".replace("Setle","Settle")].sum())  # safe in case of typo
revenue      = float(f.loc[settled_rows, "Settle Amount"].sum())
settled_cnt  = int(settled_rows.sum())
aov          = safe_div(revenue, settled_cnt)

# Header
header_path = None
for p in ("assets/header.jpg","assets/header.png","assets/header.jpeg","assets/header.webp"):
    if os.path.exists(p): header_path = p; break
if header_path:
    st.image(header_path, use_column_width=True)
else:
    st.markdown('<div class="header-row"><div class="title-left"><h1>Merchant Dashboard</h1></div></div>', unsafe_allow_html=True)

src_path = f["__path__"].iat[0] if "__path__" in f.columns and not f.empty else "—"
st.caption(f"Merchant: **{merchant_id_value}**  •  Source: `{src_path}`  •  Date: {start_date} → {end_date}")

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
    kpi_card("Total Requests", f"R {float(f['Request Amount'].sum()):,.0f}")
with cols[2]:
    kpi_card("Revenue", f"R {revenue:,.0f}")
with cols[3]:
    kpi_card("Approval Rate", f"{(approval_rate*100):.1f}%" if not math.isnan(approval_rate) else "—")
with cols[4]:
    kpi_card("Decline Rate", f"{(decline_rate*100):.1f}%" if not math.isnan(decline_rate) else "—")
with cols[5]:
    kpi_card("Average Order Value (AOV)", f"R {aov:,.2f}" if not math.isnan(aov) else "—")

# =========================
# Revenue per Month — LINE (accent green)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Revenue per Month")

df_month = (
    f.loc[settled_rows, ["Transaction Date", "Settle Amount"]]
      .assign(month_start=lambda d: pd.to_datetime(d["Transaction Date"]).dt.to_period("M").dt.to_timestamp())
      .groupby("month_start", as_index=False)
      .agg(revenue=("Settle Amount", "sum"))
      .sort_values("month_start")
)

if not df_month.empty:
    full_months = pd.date_range(df_month["month_start"].min(),
                                df_month["month_start"].max(),
                                freq="MS")
    df_month = (
        df_month.set_index("month_start")
                .reindex(full_months, fill_value=0)
                .rename_axis("month_start")
                .reset_index()
    )
    df_month["month_label"] = df_month["month_start"].dt.strftime("%b %Y")

    fig_m = px.line(df_month, x="month_start", y="revenue", markers=True)
    fig_m.update_traces(line=dict(color=PRIMARY, width=2), marker=dict(color=PRIMARY))
    fig_m.update_xaxes(title_text="", tickformat="%b %Y", dtick="M1")
    fig_m.update_yaxes(title_text="Revenue")
    fig_m.update_layout(title_text="Revenue per Month (Line)")
    apply_plotly_layout(fig_m)
    st.plotly_chart(fig_m, use_container_width=True)

    st.dataframe(
        df_month[["month_label", "revenue"]]
            .rename(columns={"month_label": "Month", "revenue": "Revenue"})
            .reset_index(drop=True),
        use_container_width=True, height=220
    )
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Row: Issuing Bank Mix + Decline Reasons
# =========================
cA, cB = st.columns(2, gap="small")

with cA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Issuing Bank Mix (Revenue)")
    issuer_df = (
        f.loc[settled_rows, ["Issuing Bank", "Settle Amount"]]
         .groupby("Issuing Bank", as_index=False)
         .agg(revenue=("Settle Amount", "sum"))
         .sort_values("revenue", ascending=False)
    )
    if not issuer_df.empty:
        fig_pie = px.pie(issuer_df, values="revenue", names="Issuing Bank", hole=0.5)
        issuer_colors = ["#334155","#475569","#64748b","#94a3b8","#cbd5e1","#e2e8f0"]
        fig_pie.update_traces(marker=dict(colors=issuer_colors, line=dict(color="#ffffff", width=1)),
                              texttemplate="%{label}<br>%{percent:.0%}", textposition="inside")
        apply_plotly_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Top Decline Reasons (as % of Attempts)")
    base_attempts = int(len(f))
    decl_df = (
        f.loc[~f["is_approved"], ["Decline Reason"]]
         .value_counts()
         .reset_index(name="count")
         .rename(columns={"index": "Decline Reason"})
         .sort_values("count", ascending=True)
    )
    if not decl_df.empty and base_attempts > 0:
        decl_df["pct_of_attempts"] = decl_df["count"] / base_attempts
        fig_decl = px.bar(decl_df, x="pct_of_attempts", y="Decline Reason", orientation="h")
        fig_decl.update_traces(marker_color=DANGER, texttemplate="%{x:.0%}", textposition="outside")
        fig_decl.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(decl_df["pct_of_attempts"].max()) * 1.15)])
        apply_plotly_layout(fig_decl)
        st.plotly_chart(fig_decl, use_container_width=True)
    else:
        st.info("No declines in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Transactions table
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Transactions (filtered)")
show_cols = [
    "Transaction Date", "Request Amount", "Settle Amount",
    "Decline Reason", "Auth Code",
    "Issuing Bank", "BIN", "Product Type",
    "Terminal ID", "Device Serial",
    "Date Payment Extract", "System Batch Number", "Device Batch Number",
    "System Trace Audit Number", "Retrieval Reference", "UTI", "Online Reference Number",
]
existing_cols = [c for c in show_cols if c in f.columns]
tbl = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
st.dataframe(tbl, use_container_width=True, height=520)

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
- **# Transactions**: all rows in the selected range.
- **Approval Rate**: rows with approval (Decline Reason starts with `"00"` or Auth Code present) ÷ all rows.
- **Revenue**: sum of **Settle Amount** where a settlement file exists (`Date Payment Extract` present) and amount ≠ 0.
- **AOV**: Revenue ÷ # of settled rows.
- **Total Requests**: Sum of **Request Amount** for all rows in the selected range.
- Visuals use a restrained palette (green for positive metrics, red for declines, neutrals elsewhere).
"""
    )
