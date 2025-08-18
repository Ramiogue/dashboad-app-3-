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
# TEAL-DARK THEME (matches your example)
# =========================
# Surfaces
CANVAS_BG   = "#0D1F24"  # page background (deep teal-blue)
SIDEBAR_BG  = "#0B1A1F"  # sidebar (slightly darker)
CARD_BG     = "#132A30"  # cards
CARD_BG_2   = "#152F36"  # kpi / inputs surface
BORDER      = "#1E3A40"  # borders & gridlines
DIVIDER     = "#21444B"  # subtle dividers

# Typography
TEXT        = "#EAF4F6"  # near-white text
TEXT_MUTED  = "#96ABB1"  # secondary text

# Accents / semantics
ACCENT      = "#2EE6D0"  # bright aqua (lines, highlights)
ACCENT_SOFT = "#18BFB2"  # softer aqua (bars/fills)
DANGER      = "#F87171"  # declines / issues

# Discrete series (neutral-cool)
SERIES = ["#2EE6D0", "#5FD4FF", "#4CB7C5", "#9EEFE5", "#67E8F9", "#94E2FF"]

def apply_plotly_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=12),
        title_x=0.01,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=SERIES,
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER, tickcolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, linecolor=BORDER, tickcolor=BORDER)
    return fig

def currency_fmt(x):
    try: return f"R {float(x):,.0f}"
    except Exception: return "R 0"

def section_title(txt):
    return f"""
    <div class="section-title">
        <h2>{txt}</h2>
    </div>
    """

# =========================
# Global CSS — teal-dark skin
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{ background: {CANVAS_BG}; color: {TEXT}; }}
    .block-container {{ padding-top:.8rem; padding-bottom:1.2rem; max-width:1280px; margin:0 auto; }}

    /* Header */
    .header-row {{
      display:flex; align-items:center; justify-content:space-between; margin-bottom:.25rem;
      border-bottom:1px solid {DIVIDER}; padding-bottom:6px;
    }}
    .title-left h1 {{ font-size:1.15rem; margin:0; color:{TEXT}; }}

    /* Section title w/ aqua underline */
    .section-title h2 {{ font-size:1.2rem; margin:12px 0 6px 0; color:{TEXT}; position:relative; padding-bottom:8px; }}
    .section-title h2:after {{ content:""; position:absolute; left:0; bottom:0; height:3px; width:64px; background:{ACCENT}; border-radius:3px; }}

    /* Cards */
    .card {{
      background:{CARD_BG};
      border:1px solid {BORDER};
      border-radius:12px;
      padding:12px;
      box-shadow:0 1px 2px rgba(0,0,0,0.35);
      margin-bottom:10px;
    }}
    .card h3 {{ margin-top:.2rem; color:{TEXT}; font-size:1.0rem; }}

    /* KPI cards */
    .kpi-card {{
      background:{CARD_BG_2};
      border:1px solid {BORDER};
      border-radius:12px;
      padding:10px 12px;
      box-shadow:0 1px 2px rgba(0,0,0,0.35);
      height:84px; display:flex; flex-direction:column; justify-content:center; gap:2px; overflow:hidden;
      position:relative;
    }}
    .kpi-card:before {{
      content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:{ACCENT};
      border-top-left-radius:12px; border-bottom-left-radius:12px;
    }}
    .kpi-title {{ font-size:.72rem; color:{TEXT_MUTED}; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .kpi-value {{ font-size:1.25rem; font-weight:800; color:{TEXT}; line-height:1.05; margin:0; }}
    .kpi-sub   {{ font-size:.75rem; color:{TEXT_MUTED}; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

    /* Inputs — dark */
    div[data-testid="stTextInput"] input,
    div[data-testid="stPassword"] input,
    .stTextInput input, .stPassword input,
    div[data-baseweb="input"] input {{
      background:{CARD_BG_2} !important; color:{TEXT} !important;
      border:1px solid {BORDER} !important; border-radius:10px !important; padding:10px 12px !important;
      box-shadow:none !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stPassword"] input:focus,
    .stTextInput input:focus, .stPassword input:focus,
    div[data-baseweb="input"] input:focus {{
      border:1.5px solid {ACCENT} !important;
      outline:none !important; box-shadow:0 0 0 3px rgba(46,230,208,.20) !important;
    }}
    div[data-testid="stTextInput"] label,
    div[data-testid="stPassword"] label {{ margin-bottom:6px !important; color:{TEXT_MUTED}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background:{SIDEBAR_BG}; box-shadow:inset -1px 0 0 {BORDER}; color:{TEXT}; }}

    [data-testid="stSidebar"] details {{ border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
    [data-testid="stSidebar"] details > summary.streamlit-expanderHeader {{
      background:#0E2630; color:{TEXT}; font-weight:700; padding:8px 12px; list-style:none;
    }}
    [data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader {{
      background:#10323E; border-bottom:1px solid {BORDER};
    }}
    [data-testid="stSidebar"] details[open] .streamlit-expanderContent {{ background:#0F2B35; padding:8px 12px; }}

    [data-testid="stSidebar"] .stDateInput input {{ background:{CARD_BG_2} !important; border:1px solid {BORDER} !important; color:{TEXT} !important; }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{ border-radius:8px; background:#0F2B35; color:{TEXT}; }}

    /* Soft divider */
    .soft-divider {{ height:10px; border-radius:999px; background:{CARD_BG}; border:1px solid {BORDER}; margin:6px 0 16px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Auth (from Secrets)
# =========================
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

# Cleaning
tx[MERCHANT_ID_COL]       = tx[MERCHANT_ID_COL].astype(str).str.strip()
tx["Transaction Date"]    = pd.to_datetime(tx["Transaction Date"], errors="coerce")
tx["Request Amount"]      = pd.to_numeric(tx["Request Amount"], errors="coerce")
tx["Settle Amount"]       = pd.to_numeric(tx["Settle Amount"], errors="coerce")
tx["Date Payment Extract"]= tx["Date Payment Extract"].fillna("").astype(str)
for c in ["Product Type","Issuing Bank","Decline Reason","Terminal ID","Device Serial","Auth Code"]:
    tx[c] = tx[c].fillna("").astype(str)

f0 = tx[tx[MERCHANT_ID_COL] == merchant_id_value].copy()
if f0.empty:
    st.warning(f"No transactions for '{merchant_id_value}' in '{MERCHANT_ID_COL}'.")
    st.stop()

# Metrics flags
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
# Sidebar filters (collapsible)
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
    kpi_card("Total Requests", currency_fmt(f['Request Amount'].sum()))
with cols[2]:
    kpi_card("Revenue", currency_fmt(revenue))
with cols[3]:
    kpi_card("Approval Rate", f"{(approval_rate*100):.1f}%" if not math.isnan(approval_rate) else "—")
with cols[4]:
    kpi_card("Decline Rate", f"{(decline_rate*100):.1f}%" if not math.isnan(decline_rate) else "—")
with cols[5]:
    kpi_card("Average Order Value (AOV)", f"R {aov:,.2f}" if not math.isnan(aov) else "—")

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

# =========================
# Charts (same visuals, teal-dark styling)
# =========================

# Revenue per Month — LINE
st.markdown(section_title("Revenue per Month"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
df_month = (
    f.loc[settled_rows, ["Transaction Date", "Settle Amount"]]
      .assign(month_start=lambda d: pd.to_datetime(d["Transaction Date"]).dt.to_period("M").dt.to_timestamp())
      .groupby("month_start", as_index=False)
      .agg(revenue=("Settle Amount", "sum"))
      .sort_values("month_start")
)
if not df_month.empty:
    full_months = pd.date_range(df_month["month_start"].min(), df_month["month_start"].max(), freq="MS")
    df_month = df_month.set_index("month_start").reindex(full_months, fill_value=0).rename_axis("month_start").reset_index()
    df_month["month_label"] = df_month["month_start"].dt.strftime("%b %Y")

    fig_m = px.line(df_month, x="month_start", y="revenue", markers=True)
    fig_m.update_traces(line=dict(color=ACCENT, width=3), marker=dict(color=ACCENT))
    fig_m.update_xaxes(title_text="", tickformat="%b %Y", dtick="M1")
    fig_m.update_yaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
    fig_m.update_layout(title_text="Revenue per Month (Line)")
    fig_m.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: R %{y:,.0f}<extra></extra>")
    apply_plotly_layout(fig_m)
    st.plotly_chart(fig_m, use_container_width=True, height=360)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# Product Type Mix Over Time — STACKED AREA
st.markdown(section_title("Product Type Mix Over Time"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
mix = f.loc[settled_rows, ["Transaction Date","Product Type","Settle Amount"]].copy()
if not mix.empty:
    mix["month"] = pd.to_datetime(mix["Transaction Date"]).dt.to_period("M").dt.to_timestamp()
    mix = mix.groupby(["month","Product Type"], as_index=False)["Settle Amount"].sum().rename(columns={"Settle Amount":"revenue"}).sort_values("month")
    fig_mix = px.area(mix, x="month", y="revenue", color="Product Type", color_discrete_sequence=SERIES)
    fig_mix.update_traces(stackgroup="one", line=dict(width=1, color=CANVAS_BG), opacity=0.9)
    fig_mix.update_layout(title_text="Revenue Mix by Product Type (Stacked Area)")
    fig_mix.update_xaxes(title_text="", tickformat="%b %Y", dtick="M1")
    fig_mix.update_yaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
    fig_mix.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>%{fullData.name}: R %{y:,.0f}<extra></extra>")
    apply_plotly_layout(fig_mix)
    st.plotly_chart(fig_mix, use_container_width=True, height=360)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# Product Type Mix (Pie / Donut)
st.markdown(section_title("Product Type Mix (Pie)"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
prod_pie = f.loc[settled_rows, ["Product Type", "Settle Amount"]].copy()
if not prod_pie.empty:
    prod_pie = prod_pie.groupby("Product Type", as_index=False)["Settle Amount"].sum().rename(columns={"Settle Amount": "revenue"}).sort_values("revenue", ascending=False)
    fig_pie_pt = px.pie(prod_pie, values="revenue", names="Product Type", hole=0.55, color_discrete_sequence=SERIES)
    fig_pie_pt.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>R %{value:,.0f}<br>%{percent:.0%}",
        hovertemplate="%{label}<br>Revenue: R %{value:,.0f} (%{percent:.1%})<extra></extra>",
        marker=dict(line=dict(color=CANVAS_BG, width=1.5))
    )
    fig_pie_pt.update_layout(title_text="Revenue by Product Type")
    apply_plotly_layout(fig_pie_pt)
    st.plotly_chart(fig_pie_pt, use_container_width=True, height=420)
else:
    st.info("No settled revenue in the selected period.")
st.markdown('</div>', unsafe_allow_html=True)

# Two-column row: Top Issuers + Top Declines
c1, c2 = st.columns((1.2, 1), gap="small")

with c1:
    st.markdown(section_title("Top Issuing Banks by Revenue"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    issuer_df = (
        f.loc[settled_rows, ["Issuing Bank", "Settle Amount"]]
         .groupby("Issuing Bank", as_index=False)
         .agg(revenue=("Settle Amount", "sum"))
         .sort_values("revenue", ascending=False)
         .head(10)
    )
    if not issuer_df.empty:
        fig_bank = px.bar(issuer_df.sort_values("revenue"), x="revenue", y="Issuing Bank",
                          orientation="h", color_discrete_sequence=[ACCENT_SOFT])
        fig_bank.update_traces(marker_line_color=CANVAS_BG, marker_line_width=1)
        fig_bank.update_xaxes(title_text="Revenue (R)", tickprefix="R ", separatethousands=True)
        fig_bank.update_yaxes(title_text="")
        fig_bank.update_layout(title_text="Top 10 Issuers (Revenue)")
        fig_bank.update_traces(hovertemplate="%{y}<br>Revenue: R %{x:,.0f}<extra></extra>")
        apply_plotly_layout(fig_bank)
        st.plotly_chart(fig_bank, use_container_width=True, height=420)
    else:
        st.info("No revenue in range.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown(section_title("Top Decline Reasons"), unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
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
        fig_decl = px.bar(decl_df, x="pct_of_attempts", y="Decline Reason", orientation="h",
                          color_discrete_sequence=[DANGER])
        fig_decl.update_traces(texttemplate="%{x:.0%}", textposition="outside")
        fig_decl.update_xaxes(tickformat=".0%", range=[0, max(0.01, float(decl_df["pct_of_attempts"].max()) * 1.15)])
        fig_decl.update_yaxes(title_text="")
        fig_decl.update_layout(title_text="As % of All Attempts")
        fig_decl.update_traces(hovertemplate="%{y}<br>% of Attempts: %{x:.1%}<extra></extra>")
        apply_plotly_layout(fig_decl)
        st.plotly_chart(fig_decl, use_container_width=True, height=420)
    else:
        st.info("No declines in the selected period.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Transactions table
# =========================
st.markdown(section_title("Transactions (Filtered)"), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
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

# Optional display currency
for col in ["Request Amount", "Settle Amount"]:
    if col in tbl.columns:
        tbl[col] = tbl[col].apply(lambda v: f"R {v:,.2f}" if pd.notnull(v) else "")

st.dataframe(tbl, use_container_width=True, height=520)

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    raw = f[existing_cols].sort_values("Transaction Date", ascending=False).reset_index(drop=True)
    return raw.to_csv(index=False).encode("utf-8")

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
- Styling only: visuals unchanged; palette reworked to deep teal with bright aqua accents to match your example.
"""
    )
