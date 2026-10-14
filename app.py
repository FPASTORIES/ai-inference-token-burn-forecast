"""
AI Inference, Credit Depletion & Contract Pricing Deal Desk Model  (v2.1 - corrected engine, peer-review follow-ups)
------------------------------------------------------------------------------------------
FP&A planning model for direct AI inference, vector storage, infrastructure step-costs,
enterprise contract cohorts (term / renewal / expansion), promotional-credit drawdown,
invoice-lag cash timing, scenario sensitivities and a like-for-like pricing comparison.

All prices, rates and presets are user-editable ESTIMATES. Verify against vendor billing
portals, billing-system exports and accounting policy before using for decisions.
See the "Change log & limitations" section at the bottom of the page.
"""
import calendar
import datetime
import html
import inspect
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference Deal Desk & Unit Economics Model",
    page_icon="⚡",
    layout="wide",
)

# ==========================================
# 0. STYLING & SMALL UI HELPERS
# ==========================================
st.markdown("""
<style>
.stApp {background:#fcfbf9;color:#2d3748;font-family:'Inter', sans-serif;}
section[data-testid="stSidebar"] {background:#f5f2ed !important; border-right: 1px solid #e2d9d0;}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important; margin-top: 6px !important; margin-bottom: 2px !important;
    padding-bottom: 0px !important; font-weight: 700 !important;
}
section[data-testid="stSidebar"] label {font-size: 0.78rem !important; color: #4a5568 !important;}
div[data-testid="stMetric"] {
    background:#fff; border:1px solid #e2d9d0; padding:6px 10px !important;
    border-radius:6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {font-size:0.75rem !important; color:#615a52 !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.1rem !important; font-weight:700;}
.compact-header {
    background: linear-gradient(135deg,#f5f0eb,#ebe5df); padding: 8px 14px; border-radius: 6px;
    border: 1px solid #e2d9d0; margin-bottom: 8px;
}
.compact-header h1 {color:#2d3748; margin:0 0 2px 0; font-size:1.25rem; font-weight:700;}
.compact-header p {color:#615a52; margin:0; font-size:0.8rem;}
.cfo-commentary {
    background: #ffffff; border: 1px solid #cbd5e0; border-left: 4px solid #3182ce;
    padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;
}
.cfo-commentary h4 {margin:0 0 4px 0; font-size:0.9rem; color:#2b6cb0; font-weight:700;}
.cfo-commentary ul {margin:0; padding-left:16px; font-size:0.82rem; color:#4a5568;}
.cfo-commentary li {margin-bottom:2px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="compact-header">
    <h1>⚡ AI Deal Desk & Unit Economics Simulator</h1>
    <p>Founder presets, enterprise contract cohorts, typical/heavy-user costs, like-for-like pricing simulation, credit runway and cash timing.</p>
</div>
""", unsafe_allow_html=True)


def _supports_stretch(fn):
    """True only if this Streamlit build's `fn` really accepts width='stretch' (older builds take an int or only **kwargs)."""
    try:
        prm = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        return False
    return prm is not None and ("stretch" in str(prm.annotation) or prm.default == "stretch")


def _stretch(fn, obj, **kw):
    """width='stretch' on current Streamlit; use_container_width on older versions."""
    if _supports_stretch(fn):
        try:
            return fn(obj, width="stretch", **kw)
        except Exception:
            pass
    return fn(obj, use_container_width=True, **kw)


def show_df(obj, **kw):
    return _stretch(st.dataframe, obj, **kw)


def show_chart(fig):
    return _stretch(st.plotly_chart, fig)


def style_frame(df):
    """Format a DataFrame from column-name conventions (no hand-typed keys that can silently miss)."""
    fmt = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if "1,000 Prompts" in c:
            fmt[c] = "${:,.4f}"
        elif "/ Paid" in c or "/ Account-Month" in c:
            fmt[c] = "${:,.2f}"
        elif c.startswith("Δ") and "(pp)" in c:
            fmt[c] = "{:+.1f}pp"
        elif c.startswith("Δ") and "($)" in c:
            fmt[c] = "${:+,.0f}"
        elif "(%)" in c:
            fmt[c] = "{:.1f}%"
        elif "($)" in c:
            fmt[c] = "${:,.0f}"
        elif "Index" in c:
            fmt[c] = "{:.3f}"
        elif "Average" in c or c.startswith("Avg"):
            fmt[c] = "{:,.1f}"
        else:
            fmt[c] = "{:,.0f}"
    return df.style.format(fmt, na_rep="—")


# ==========================================
# 1. SIDEBAR ASSUMPTIONS
# ==========================================
FOUNDER_PRESETS = {
    "Pre-launch": {"dau": 100, "pro": 0, "ent": 0, "growth": 5.0, "deals": 0.0, "prompts": 3, "infra": 500.0, "ai_credits": 0.0, "cloud_credits": 0.0},
    "Early traction": {"dau": 5000, "pro": 200, "ent": 50, "growth": 8.0, "deals": 3.0, "prompts": 5, "infra": 2500.0, "ai_credits": 20000.0, "cloud_credits": 30000.0},
    "Scaling": {"dau": 50000, "pro": 2500, "ent": 250, "growth": 10.0, "deals": 8.0, "prompts": 8, "infra": 10000.0, "ai_credits": 50000.0, "cloud_credits": 100000.0},
}

st.sidebar.header("0. Founder Onboarding")
founder_stage = st.sidebar.selectbox(
    "Choose a starting profile", ["Early traction", "Pre-launch", "Scaling"], index=0,
    help="Presets populate starting values only. All assumptions remain editable.",
)
preset = FOUNDER_PRESETS[founder_stage]
current_now = datetime.datetime.now()
start_year = int(st.sidebar.number_input("Forecast Start Year", min_value=2024, max_value=2035, value=current_now.year))
start_month = int(st.sidebar.slider("Forecast Start Month", 1, 12, current_now.month))
forecast_horizon = int(st.sidebar.slider("Forecast Horizon (Months)", 1, 36, 12))

st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle(
    "🛠️ Enable Advanced Deal Desk Parameters", value=False,
    help="Start with the simpler founder view. Turn this on to edit every model assumption.",
)

# Engine defaults. Advanced widgets override these; simple mode uses them as-is.
DEFAULTS = {
    # funnel & revenue
    "paid_dau_factor": 0.70, "new_conv": 0.15, "base_conv": 0.02, "churn": 0.03,
    "arpu_pro": 30.0, "arpu_ent": 250.0, "pro_expansion_pct": 0.10, "fee_pct": 0.029,
    # enterprise contracts
    "ent_term": 12, "ent_renewal_rate": 0.80, "ent_in_term_churn": 0.0, "ent_expansion": 0.015,
    "ent_seats": 20, "ent_prompts": 3000, "impl_fee": 0.0, "impl_cost": 0.0, "impl_rec_months": 1,
    # usage
    "p1_vol": 35, "p1_month": 3, "p1_width": 1.5, "p2_vol": 45, "p2_month": 11, "p2_width": 2.0,
    "heavy_share": 0.20, "heavy_mode": "Multiplier", "heavy_mult": 4.0, "pareto_share": 0.80,
    "free_mult": 0.50, "in_tok": 1000, "out_tok": 400, "in_grow": 0.03, "out_grow": 0.0,
    # routing & token pricing
    "front_mix": 0.30, "f_in": 2.50, "f_out": 10.00, "s_in": 0.15, "s_out": 0.60,
    "cache_rate": 0.40, "cache_ratio": 0.20, "batch_share": 0.20, "batch_ratio": 0.50, "overhead": 0.0,
    # infrastructure & credits
    "base_infra": preset["infra"], "step_type": "DAU Threshold", "step_thresh": 20000.0, "step_cost": 1500.0,
    "vec_cost": 0.05, "ai_credits": preset["ai_credits"], "cloud_credits": preset["cloud_credits"],
    "ai_expiry": 0, "cloud_expiry": 0, "invoice_lag": 0, "ai_elig": 1.0, "cloud_elig": 1.0,
    # scenario hooks (1.0 = unchanged)
    "usage_mult": 1.0, "price_mult": 1.0,
}
cfg = dict(DEFAULTS)

st.sidebar.header("1. Core Growth & Funnel")
starting_dau = st.sidebar.number_input(
    "Month-1 Daily Active Users (DAU)", min_value=0, value=preset["dau"], step=500, key=f"dau_{founder_stage}",
    help="DAU in the FIRST forecast month, exactly as entered. DAU growth applies from month 2, so month 1 reconciles to your actual DAU.",
)
starting_pro_subs = st.sidebar.number_input("Starting Pro Subscribers", min_value=0, value=preset["pro"], step=50, key=f"pro_{founder_stage}")
starting_ent_accounts = st.sidebar.number_input("Starting Enterprise Accounts (contracts)", min_value=0, value=preset["ent"], step=10, key=f"ent_{founder_stage}")
dau_monthly_growth = st.sidebar.slider(
    "Total DAU Monthly Growth (%)", 0.0, 25.0, float(preset["growth"]), 0.5, key=f"growth_{founder_stage}",
    help="Growth of total daily active users (free + paid), applied from month 2 onward. Net new DAU feeds new-user conversion. Month 1 has no new-user conversions; free-pool conversions still occur.",
) / 100
net_new_ent_contracts = st.sidebar.number_input(
    "Net New Enterprise Deals / Month", min_value=0.0, value=float(preset["deals"]), step=0.5, key=f"deals_{founder_stage}",
    help="Sales-led pipeline velocity, independent of PLG traffic. Fractional values allowed (e.g. 2.5 = 5 deals every 2 months).",
)
baseline_prompts = st.sidebar.slider("Baseline Prompts / Active Paid User / Day", 1, 20, int(preset["prompts"]), key=f"prompts_{founder_stage}")

st.sidebar.header("2. Targets")
margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

if advanced_mode:
    with st.sidebar.expander("📂 A. Subscriber Funnel & Revenue"):
        cfg["paid_dau_factor"] = st.slider("Paid Subscribers Active Daily (%)", 0, 100, 70, 5) / 100
        cfg["new_conv"] = st.slider("New-User Conversion to Pro (%)", 0.0, 50.0, 15.0, 0.5) / 100
        cfg["base_conv"] = st.slider("Existing Free-Pool Conversion (% / month)", 0.0, 10.0, 2.0, 0.1,
                                     help="Monthly probability that an existing free active user upgrades. Applied to the free pool net of paid actives.") / 100
        cfg["churn"] = st.slider("Monthly Pro Churn (%)", 0.0, 15.0, 3.0, 0.5) / 100
        cfg["arpu_pro"] = st.number_input("Pro Tier Monthly ARPU ($)", min_value=0.0, value=30.0, step=5.0)
        cfg["pro_expansion_pct"] = st.slider("Pro Overage / Expansion Revenue (% of Pro base)", 0.0, 50.0, 10.0, 1.0,
                                             help="Applies to Pro revenue only. Enterprise expansion is modeled per account below, so it is not double counted.") / 100
        cfg["fee_pct"] = st.slider("Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1) / 100

    with st.sidebar.expander("📂 B. Enterprise Contract Economics"):
        cfg["arpu_ent"] = st.number_input("Enterprise Monthly Contract Value / Account ($)", min_value=0.0, value=250.0, step=25.0)
        cfg["ent_term"] = int(st.number_input("Contract Term (months)", min_value=1, max_value=120, value=12, step=1))
        cfg["ent_renewal_rate"] = st.slider("Renewal Rate at Term End (%)", 0, 100, 80, 5,
                                            help="Share of accounts reaching term end that renew for another full term.") / 100
        cfg["ent_in_term_churn"] = st.slider("In-Term Logo Churn (% / month)", 0.0, 10.0, 0.0, 0.5,
                                             help="Committed contracts normally do not churn mid-term. Leave at 0 unless you have termination-for-convenience.") / 100
        cfg["ent_expansion"] = st.slider("Seat / Usage Expansion per Account (% / month)", 0.0, 10.0, 1.5, 0.5,
                                         help="Grows contract value, seats and prompts of an existing account. Does NOT add accounts. Timing assumption: expansion starts immediately and compounds monthly; real expansion usually arrives through contract amendments or seat ramps.") / 100
        cfg["ent_seats"] = int(st.number_input("Contracted Seats per Account (at signing)", min_value=1, value=20, step=5))
        cfg["ent_prompts"] = int(st.number_input("Enterprise Prompts per Account / Month (at signing)", min_value=0, value=3000, step=500))
        cfg["impl_fee"] = st.number_input("Implementation Fee per New Deal ($)", min_value=0.0, value=0.0, step=500.0)
        cfg["impl_rec_months"] = int(st.number_input("Implementation Fee Recognition Period (months; 1 = upfront)", min_value=1, max_value=36, value=1, step=1))
        cfg["impl_cost"] = st.number_input("Implementation Delivery Cost per New Deal ($)", min_value=0.0, value=0.0, step=500.0)

    with st.sidebar.expander("📂 C. Usage Peaks & Heavy-User Skew"):
        st.markdown("**Peak 1 (calendar month)**")
        cfg["p1_vol"] = st.slider("Peak 1 Prompts / User / Day", 1, 100, 35)
        cfg["p1_month"] = st.selectbox("Peak 1 Month", list(range(1, 13)), index=2, format_func=lambda m: calendar.month_name[m])
        cfg["p1_width"] = st.slider("Peak 1 Spread (σ, months)", 0.5, 4.0, 1.5, 0.5)
        st.markdown("**Peak 2 (calendar month)**")
        cfg["p2_vol"] = st.slider("Peak 2 Prompts / User / Day", 1, 100, 45)
        cfg["p2_month"] = st.selectbox("Peak 2 Month", list(range(1, 13)), index=10, format_func=lambda m: calendar.month_name[m])
        cfg["p2_width"] = st.slider("Peak 2 Spread (σ, months)", 0.5, 4.0, 2.0, 0.5)
        cfg["heavy_share"] = st.slider("Heavy Users as Share of Pro Users (%)", 0, 50, 20, 5) / 100
        cfg["heavy_mode"] = st.selectbox("Heavy-User Calibration", ["Multiplier", "Pareto prompt share"],
                                         help="Multiplier: heavy users prompt N x typical. Pareto: heavy users generate X% of Pro prompts; the multiplier is derived.")
        if cfg["heavy_mode"] == "Multiplier":
            cfg["heavy_mult"] = st.slider("Heavy-User Prompt Multiplier (vs. Typical)", 1.0, 20.0, 4.0, 0.5)
        else:
            cfg["pareto_share"] = st.slider("Prompt Share Generated by Heavy Users (%)", 50, 95, 80, 5) / 100
        cfg["free_mult"] = st.slider("Free Tier Usage Multiplier (%)", 0, 100, 50) / 100
        cfg["in_tok"] = st.number_input("Initial Input Tokens / Prompt", min_value=0, value=1000, step=100)
        cfg["out_tok"] = st.number_input("Initial Output Tokens / Prompt", min_value=0, value=400, step=50)
        cfg["in_grow"] = st.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
        cfg["out_grow"] = st.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

    with st.sidebar.expander("📂 D. Token Routing & Pricing Rules"):
        cfg["front_mix"] = st.slider("Frontier Model Traffic Share (%)", 0, 100, 30) / 100
        cfg["f_in"] = st.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25)
        cfg["f_out"] = st.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50)
        cfg["s_in"] = st.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
        cfg["s_out"] = st.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f")
        cfg["cache_rate"] = st.slider("Eligible Input Tokens Cached (%)", 0, 90, 40) / 100
        cfg["cache_ratio"] = st.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
        cfg["batch_share"] = st.slider("Traffic Using Batch Pricing (%)", 0, 80, 20) / 100
        cfg["batch_ratio"] = st.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100
        cfg["overhead"] = st.slider("Inference Overhead (%)", 0, 100, 0) / 100

    with st.sidebar.expander("📂 E. Infrastructure & Credit Drawdown"):
        cfg["base_infra"] = st.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=float(preset["infra"]), step=500.0, key=f"infra_{founder_stage}")
        cfg["step_type"] = st.selectbox("Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"])
        if cfg["step_type"] == "DAU Threshold":
            cfg["step_thresh"] = float(st.number_input("DAU Capacity Block (Users)", min_value=1000, value=20000, step=5000))
        elif cfg["step_type"] == "Paid Subscribers Threshold":
            cfg["step_thresh"] = float(st.number_input("Subscriber Capacity Block (Pro subs + enterprise accounts)", min_value=500, value=5000, step=1000))
        else:
            cfg["step_thresh"] = float(st.number_input("Monthly Prompt Block (Millions)", min_value=1, value=20, step=5)) * 1_000_000
        cfg["step_cost"] = st.number_input("Infra Step-Up Cost ($ per block)", min_value=0.0, value=1500.0, step=250.0)
        cfg["vec_cost"] = st.number_input("Vector DB Cost / 1k Prompts ($)", min_value=0.0, value=0.05, step=0.01)
        cfg["ai_credits"] = st.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=float(preset["ai_credits"]), step=2500.0, key=f"aicred_{founder_stage}")
        cfg["cloud_credits"] = st.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=float(preset["cloud_credits"]), step=5000.0, key=f"cloudcred_{founder_stage}")
        cfg["ai_expiry"] = int(st.number_input("AI Credits Usable Through Forecast Month (0 = no expiry)", min_value=0, max_value=36, value=0, step=1))
        cfg["cloud_expiry"] = int(st.number_input("Cloud Credits Usable Through Forecast Month (0 = no expiry)", min_value=0, max_value=36, value=0, step=1))
        cfg["invoice_lag"] = int(st.selectbox("Vendor Invoice Payment Lag (months)", [0, 1, 2, 3], index=0,
                                              help="Vendor invoices (API, vector, infra) are paid this many months late. Payment fees and implementation costs are paid immediately."))
        cfg["ai_elig"] = st.slider("AI Credit Eligible Share of API Cost (%)", 0, 100, 100) / 100
        cfg["cloud_elig"] = st.slider("Cloud Credit Eligible Share of Vector + Infra (%)", 0, 100, 100) / 100

P = dict(cfg)
P.update({
    "start_year": start_year, "start_month": start_month, "horizon": forecast_horizon,
    "dau": float(starting_dau), "pro": float(starting_pro_subs), "ent": float(starting_ent_accounts),
    "dau_growth": float(dau_monthly_growth), "net_new_ent": float(net_new_ent_contracts),
    "base_prompts": float(baseline_prompts),
})


# ==========================================
# 2. ENGINE
# ==========================================
def validate_params(p):
    errs = []
    if not (1 <= int(p["horizon"]) <= 36):
        errs.append("Forecast horizon must be between 1 and 36 months.")
    nonneg = ["dau", "pro", "ent", "net_new_ent", "arpu_pro", "arpu_ent", "base_prompts", "in_tok", "out_tok",
              "f_in", "f_out", "s_in", "s_out", "base_infra", "step_cost", "vec_cost", "ai_credits", "cloud_credits",
              "ent_prompts", "impl_fee", "impl_cost", "usage_mult", "price_mult", "dau_growth", "ent_expansion",
              "pro_expansion_pct", "fee_pct", "overhead", "in_grow", "out_grow"]
    for k in nonneg:
        if p[k] < 0:
            errs.append(f"'{k}' must be >= 0.")
    for k in ["paid_dau_factor", "new_conv", "base_conv", "churn", "ent_renewal_rate", "ent_in_term_churn", "cache_rate",
              "cache_ratio", "batch_share", "batch_ratio", "front_mix", "ai_elig", "cloud_elig", "free_mult", "heavy_share"]:
        if not (0.0 <= p[k] <= 1.0):
            errs.append(f"'{k}' must be between 0 and 1.")
    if int(p["ent_term"]) < 1:
        errs.append("Enterprise contract term must be at least 1 month.")
    if int(p["ent_seats"]) < 1:
        errs.append("Enterprise seats per account must be at least 1.")
    if int(p["impl_rec_months"]) < 1:
        errs.append("Implementation recognition period must be at least 1 month.")
    if p["p1_width"] <= 0 or p["p2_width"] <= 0:
        errs.append("Peak spread (sigma) must be positive.")
    if p["step_thresh"] <= 0:
        errs.append("Infrastructure capacity block must be positive.")
    if p["heavy_mode"] == "Pareto prompt share" and p["pareto_share"] >= 1.0:
        errs.append("Pareto prompt share must be below 100%.")
    if int(p["invoice_lag"]) not in (0, 1, 2, 3):
        errs.append("Invoice lag must be 0-3 months.")
    if errs:
        raise ValueError(" ".join(errs))


def heavy_multiplier(p):
    """Heavy-user prompt multiplier. In Pareto mode it is derived: heavy share h generating prompt share s => m = s(1-h)/(h(1-s))."""
    h = p["heavy_share"]
    if h <= 0 or h >= 1:
        return 1.0
    if p["heavy_mode"] == "Pareto prompt share":
        s = min(max(p["pareto_share"], 0.0), 0.99)
        return max(1.0, s * (1 - h) / (h * (1 - s)))
    return max(1.0, p["heavy_mult"])


def usage_curve(cal_months, base, v1, m1, w1, v2, m2, w2):
    """Prompts / active paid user / day by CALENDAR month. Peaks are never below baseline; distance wraps around the year."""
    cal = np.asarray(cal_months, dtype=float)

    def circ(m):
        d = np.abs(cal - m)
        return np.minimum(d, 12.0 - d)

    c1 = (max(v1, base) - base) * np.exp(-(circ(m1) ** 2) / (2 * w1 ** 2))
    c2 = (max(v2, base) - base) * np.exp(-(circ(m2) ** 2) / (2 * w2 ** 2))
    return base + c1 + c2


def simulate(p):
    validate_params(p)
    n = int(p["horizon"])
    start_period = pd.Period(freq="M", year=int(p["start_year"]), month=int(p["start_month"]))
    dates = [start_period + i for i in range(n)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    cal_months = [d.month for d in dates]

    curve = usage_curve(cal_months, p["base_prompts"], p["p1_vol"], p["p1_month"], p["p1_width"],
                        p["p2_vol"], p["p2_month"], p["p2_width"])
    h_mult = heavy_multiplier(p)
    act = p["paid_dau_factor"]
    term = int(p["ent_term"])
    seats = float(p["ent_seats"])
    impl_months = int(p["impl_rec_months"])
    impl_sched = np.zeros(n + impl_months + 1)
    pm = p["price_mult"]
    f_in, f_out, s_in, s_out = p["f_in"] * pm, p["f_out"] * pm, p["s_in"] * pm, p["s_out"] * pm

    # Enterprise cohorts. Starting accounts are a STAGGERED book: evenly spread over remaining terms 1..term,
    # so renewals arrive every month instead of one cliff. 'end' = month index at whose START the term ends.
    cohorts = []
    if p["ent"] > 0:
        for r in range(1, term + 1):
            cohorts.append({"acc": p["ent"] / term, "idx": 1.0, "end": r})

    ai_left = float(p["ai_credits"])
    cloud_left = float(p["cloud_credits"])
    prev_pro = float(p["pro"])
    dau_val = float(p["dau"])
    cum_billed = 0.0
    cum_recognized = 0.0
    rows = []

    for i in range(n):
        # ---------- enterprise cohort roll-forward ----------
        beg_acc = sum(c["acc"] for c in cohorts)
        beg_equiv = sum(c["acc"] * c["idx"] for c in cohorts)
        due = 0.0
        non_renewed = 0.0
        renewed_ids = set()  # cohorts that reached term end this month: exempt from in-term churn this month
        for c in cohorts:
            if c["end"] == i:
                renewed_ids.add(id(c))
                due += c["acc"]
                lost = c["acc"] * (1.0 - p["ent_renewal_rate"])
                non_renewed += lost
                c["acc"] -= lost
                c["end"] = i + term
        intrm = 0.0
        if p["ent_in_term_churn"] > 0:
            for c in cohorts:
                if id(c) in renewed_ids:
                    continue  # renewal-month exemption: their only exit this month is the renewal decision
                lost = c["acc"] * p["ent_in_term_churn"]
                c["acc"] -= lost
                intrm += lost
        for c in cohorts:
            c["idx"] *= (1.0 + p["ent_expansion"])
        new_ent = float(p["net_new_ent"])
        if new_ent > 0:
            cohorts.append({"acc": new_ent, "idx": 1.0, "end": i + term})
        cohorts = [c for c in cohorts if c["acc"] > 1e-12]
        end_acc = sum(c["acc"] for c in cohorts)
        end_equiv = sum(c["acc"] * c["idx"] for c in cohorts)
        avg_acc = (beg_acc + end_acc) / 2.0
        avg_equiv = (beg_equiv + end_equiv) / 2.0  # expansion-weighted account equivalents

        # ---------- DAU & Pro funnel ----------
        beg_pro = prev_pro
        if i == 0:
            prev_dau = dau_val
            new_users = 0.0
        else:
            prev_dau = dau_val
            dau_val = prev_dau * (1.0 + p["dau_growth"])
            new_users = max(0.0, dau_val - prev_dau)
        paid_active_beg = beg_pro * act + beg_equiv * seats * act
        free_pool_beg = max(0.0, prev_dau - paid_active_beg)
        conv_new = new_users * p["new_conv"]
        conv_base = free_pool_beg * p["base_conv"]
        conv_cap = free_pool_beg + new_users
        conv_total = conv_new + conv_base
        if conv_total > conv_cap and conv_total > 0:
            scale = conv_cap / conv_total
            conv_new *= scale
            conv_base *= scale
            conv_total = conv_cap
        churned_pro = beg_pro * p["churn"]
        ending_pro = beg_pro + conv_total - churned_pro
        prev_pro = ending_pro
        avg_pro = (beg_pro + ending_pro) / 2.0
        avg_dau = (prev_dau + dau_val) / 2.0

        pro_active = avg_pro * act
        ent_active_seats = avg_equiv * seats * act
        free_dau = max(0.0, avg_dau - pro_active - ent_active_seats)

        # ---------- prompts ----------
        typ_rate = max(0.0, float(curve[i])) * p["usage_mult"]
        typ_prompts = pro_active * (1 - p["heavy_share"]) * typ_rate * month_days[i]
        heavy_prompts = pro_active * p["heavy_share"] * typ_rate * h_mult * month_days[i]
        ent_prompts = avg_equiv * p["ent_prompts"] * p["usage_mult"]
        free_prompts = free_dau * typ_rate * p["free_mult"] * month_days[i]
        prompts_total = typ_prompts + heavy_prompts + ent_prompts + free_prompts

        # ---------- API / vector / infra ----------
        in_tok = p["in_tok"] * ((1 + p["in_grow"]) ** i)
        out_tok = p["out_tok"] * ((1 + p["out_grow"]) ** i)
        raw_in_m = prompts_total * in_tok / 1_000_000
        raw_out_m = prompts_total * out_tok / 1_000_000
        unc_in_m = raw_in_m * (1 - p["cache_rate"])
        cac_in_m = raw_in_m * p["cache_rate"]
        std_mix = 1.0 - p["front_mix"]
        front_in = (unc_in_m * f_in + cac_in_m * f_in * p["cache_ratio"]) * p["front_mix"]
        std_in = (unc_in_m * s_in + cac_in_m * s_in * p["cache_ratio"]) * std_mix
        front_out = raw_out_m * f_out * p["front_mix"]
        std_out = raw_out_m * s_out * std_mix
        api_cost = (front_in + std_in + front_out + std_out) * ((1 - p["batch_share"]) + p["batch_share"] * p["batch_ratio"]) * (1 + p["overhead"])
        vector_cost = prompts_total / 1000.0 * p["vec_cost"]

        ending_paid_accounts = ending_pro + end_acc
        if p["step_type"] == "DAU Threshold":
            trigger = dau_val
        elif p["step_type"] == "Paid Subscribers Threshold":
            trigger = ending_paid_accounts
        else:
            trigger = prompts_total
        thr = p["step_thresh"]
        step_mult = max(0, math.ceil(trigger / thr) - 1) if trigger > thr else 0
        infra_cost = p["base_infra"] + step_mult * p["step_cost"]

        # ---------- revenue ----------
        pro_base_rev = avg_pro * p["arpu_pro"]
        pro_exp_rev = pro_base_rev * p["pro_expansion_pct"]
        ent_rev = avg_equiv * p["arpu_ent"]
        subscription_rev = pro_base_rev + pro_exp_rev + ent_rev
        billed = new_ent * p["impl_fee"]
        if billed > 0:
            impl_sched[i:i + impl_months] += billed / impl_months
        impl_rev = float(impl_sched[i])
        cum_billed += billed
        cum_recognized += impl_rev
        recognized_rev = subscription_rev + impl_rev
        ending_mrr = ending_pro * p["arpu_pro"] * (1 + p["pro_expansion_pct"]) + end_equiv * p["arpu_ent"]

        # ---------- COGS ----------
        payment_fees = recognized_rev * p["fee_pct"]
        impl_cost = new_ent * p["impl_cost"]
        cogs = api_cost + vector_cost + infra_cost + payment_fees + impl_cost
        gp = recognized_rev - cogs
        gm = gp / recognized_rev * 100 if recognized_rev > 0 else np.nan

        # ---------- credits ----------
        ai_open, cloud_open = ai_left, cloud_left
        ai_expired = cloud_expired = 0.0
        if p["ai_expiry"] > 0 and (i + 1) > p["ai_expiry"] and ai_left > 0:
            ai_expired, ai_left = ai_left, 0.0
        if p["cloud_expiry"] > 0 and (i + 1) > p["cloud_expiry"] and cloud_left > 0:
            cloud_expired, cloud_left = cloud_left, 0.0
        ai_applied = min(ai_left, api_cost * p["ai_elig"])
        ai_left -= ai_applied
        cloud_applied = min(cloud_left, (vector_cost + infra_cost) * p["cloud_elig"])
        cloud_left -= cloud_applied
        credits_applied = ai_applied + cloud_applied
        vendor_payable = max(0.0, api_cost + vector_cost + infra_cost - credits_applied)
        immediate_cash = payment_fees + impl_cost
        cash_payable = vendor_payable + immediate_cash

        rows.append({
            "Month": month_names[i],
            "DAU": dau_val,
            "Beginning Pro Subs": beg_pro,
            "New-User Conversions": conv_new,
            "Free-Pool Conversions": conv_base,
            "Churned Pro Subs": churned_pro,
            "Ending Pro Subs": ending_pro,
            "Beginning Ent Accounts": beg_acc,
            "New Ent Deals": new_ent,
            "Ent Accounts Up For Renewal": due,
            "Ent Non-Renewals": non_renewed,
            "Ent In-Term Churn": intrm,
            "Ending Ent Accounts": end_acc,
            "Ent Expansion Index": (end_equiv / end_acc) if end_acc > 0 else 1.0,
            "Ending Paid Accounts": ending_paid_accounts,
            "Average Pro Subs": avg_pro,
            "Average Ent Accounts": avg_acc,
            "Typical Pro Prompts": typ_prompts,
            "Heavy Pro Prompts": heavy_prompts,
            "Enterprise Prompts": ent_prompts,
            "Free Prompts": free_prompts,
            "Monthly Prompts": prompts_total,
            "Ending MRR Run-Rate ($)": ending_mrr,
            "Subscription Revenue ($)": subscription_rev,
            "Implementation Fees Billed ($)": billed,
            "Implementation Revenue ($)": impl_rev,
            "Recognized Revenue ($)": recognized_rev,
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost,
            "Fixed Infrastructure ($)": infra_cost,
            "Payment Processing ($)": payment_fees,
            "Implementation Cost ($)": impl_cost,
            "Total Modeled COGS ($)": cogs,
            "Modeled Gross Profit ($)": gp,
            "Modeled Gross Margin (%)": gm,
            "Opening AI Credits ($)": ai_open,
            "AI Credits Expired ($)": ai_expired,
            "AI Credits Applied ($)": ai_applied,
            "Remaining AI Credits ($)": ai_left,
            "Opening Cloud Credits ($)": cloud_open,
            "Cloud Credits Expired ($)": cloud_expired,
            "Cloud Credits Applied ($)": cloud_applied,
            "Remaining Cloud Credits ($)": cloud_left,
            "Total Credits Applied ($)": credits_applied,
            "Vendor Invoices Payable ($)": vendor_payable,
            "Estimated Cash Payable ($)": cash_payable,
        })

    # ---------- invoice lag: vendor invoices lag; fees and implementation costs are paid immediately ----------
    lag = int(p["invoice_lag"])
    vendor = [r["Vendor Invoices Payable ($)"] for r in rows]
    for i, r in enumerate(rows):
        vendor_paid = vendor[i - lag] if i >= lag else 0.0
        r["Estimated Cash Paid This Month ($)"] = vendor_paid + r["Payment Processing ($)"] + r["Implementation Cost ($)"]
        r["Vendor Invoices Outstanding ($)"] = sum(vendor[max(0, i - lag + 1):i + 1]) if lag > 0 else 0.0
        # Accrual revenue less vendor invoices (net of credits, as incurred) and immediate costs. NOT cash profit:
        # it ignores invoice lag timing and revenue collections.
        r["Revenue Less Net Vendor Payables & Immediate Costs ($)"] = r["Recognized Revenue ($)"] - r["Estimated Cash Payable ($)"]
        r["Deferred Implementation Revenue ($)"] = float(np.sum([x["Implementation Fees Billed ($)"] for x in rows[:i + 1]])
                                                         - np.sum([x["Implementation Revenue ($)"] for x in rows[:i + 1]]))
    return pd.DataFrame(rows)


run_model_simulation = st.cache_data(show_spinner=False)(simulate)


def cost_allocation(df):
    """Split API + vector + infra cost between paid and free traffic by prompt volume, month by month."""
    pool = df["API Cost ($)"] + df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"]
    prompts = df["Monthly Prompts"].to_numpy()
    paid_prompts = (df["Typical Pro Prompts"] + df["Heavy Pro Prompts"] + df["Enterprise Prompts"]).to_numpy()
    # Share of the month's pool carried by paid traffic. Months with no traffic at all (idle capacity)
    # have no paid share, so their cost lands in the free-tier / idle bucket instead of vanishing.
    paid_share = np.divide(paid_prompts, prompts, out=np.zeros(len(df)), where=prompts > 0)
    pool_arr = pool.to_numpy()
    return float((pool_arr * paid_share).sum()), float((pool_arr * (1.0 - paid_share)).sum())


def usage_ceiling(df, p, target_pct):
    """Per-account Pro usage ceiling in the ENDING month, using that month's marginal cost per prompt."""
    row = df.iloc[-1]
    prompts = float(row["Monthly Prompts"])
    rev = p["arpu_pro"] * (1 + p["pro_expansion_pct"])
    allowed = rev * (1 - target_pct / 100.0 - p["fee_pct"])
    if prompts <= 0:
        return {"var_cost": 0.0, "loaded_cost": 0.0, "var": math.inf, "loaded": math.inf, "no_headroom": allowed <= 0, "rev": rev, "allowed": allowed}
    var_cost = float(row["API Cost ($)"] + row["Vector DB Cost ($)"]) / prompts
    loaded_cost = var_cost + float(row["Fixed Infrastructure ($)"]) / prompts
    if allowed <= 0:
        return {"var_cost": var_cost, "loaded_cost": loaded_cost, "var": 0.0, "loaded": 0.0, "no_headroom": True, "rev": rev, "allowed": allowed}
    return {
        "var_cost": var_cost, "loaded_cost": loaded_cost,
        "var": allowed / var_cost if var_cost > 0 else math.inf,
        "loaded": allowed / loaded_cost if loaded_cost > 0 else math.inf,
        "no_headroom": False, "rev": rev, "allowed": allowed,
    }


def credit_status(df, start_balance, opening_col, expired_col, remaining_col):
    if start_balance <= 0:
        return "No starting balance"
    exp = df[df[expired_col] > 0]
    if len(exp):
        return f"Expired {exp['Month'].iloc[0]} (${exp[expired_col].sum():,.0f} unused)"
    gone = df[(df[remaining_col] <= 0.005) & (df[opening_col] > 0.005)]
    if len(gone):
        return f"Exhausted {gone['Month'].iloc[0]}"
    return f"${df[remaining_col].iloc[-1]:,.0f} remains at horizon end"


# ---------- scenarios: each one moves ONE driver family (or is an explicit combined stress) ----------
SCENARIOS = [
    ("1. Base Case", {}),
    ("2. PLG Growth High (1.25x DAU growth)", {"growth": 1.25}),
    ("3. PLG Growth Low (0.75x DAU growth)", {"growth": 0.75}),
    ("4. Enterprise Deal Velocity High (1.25x)", {"deals": 1.25}),
    ("5. Enterprise Deal Velocity Low (0.75x)", {"deals": 0.75}),
    ("6. Prompt Intensity High (1.5x, all segments)", {"usage": 1.5}),
    ("7. Prompt Intensity Low (0.7x, all segments)", {"usage": 0.7}),
    ("8. API Price Inflation (+25%)", {"price": 1.25}),
    ("9. API Price Reduction (-20%)", {"price": 0.8}),
    ("10. Conversion Weakness (0.5x conversion rates)", {"conv": 0.5}),
    ("11. Churn Shock (2x Pro churn, renewal -15pp)", {"churn": 2.0, "renewal_delta": -0.15}),
    ("12. Combined Downside Stress", {"growth": 0.75, "deals": 0.75, "usage": 1.4, "price": 1.2, "conv": 0.75, "churn": 1.5, "renewal_delta": -0.10}),
]


def apply_scenario(p, s):
    q = dict(p)
    q["dau_growth"] = p["dau_growth"] * s.get("growth", 1.0)
    q["net_new_ent"] = p["net_new_ent"] * s.get("deals", 1.0)   # no rounding: fractional deals allowed
    q["usage_mult"] = p["usage_mult"] * s.get("usage", 1.0)     # scales typical, heavy, free AND enterprise prompts
    q["price_mult"] = p["price_mult"] * s.get("price", 1.0)
    q["new_conv"] = min(1.0, p["new_conv"] * s.get("conv", 1.0))
    q["base_conv"] = min(1.0, p["base_conv"] * s.get("conv", 1.0))
    q["churn"] = min(0.99, p["churn"] * s.get("churn", 1.0))
    q["ent_renewal_rate"] = min(1.0, max(0.0, p["ent_renewal_rate"] + s.get("renewal_delta", 0.0)))
    return q


def totals(d):
    rev = float(d["Recognized Revenue ($)"].sum())
    cogs = float(d["Total Modeled COGS ($)"].sum())
    return {
        "rev": rev, "cogs": cogs, "gp": rev - cogs, "gm": ((rev - cogs) / rev * 100) if rev > 0 else np.nan,
        "arr": float(d["Ending MRR Run-Rate ($)"].iloc[-1]) * 12,
        "payable": float(d["Estimated Cash Payable ($)"].sum()),
        "paid": float(d["Estimated Cash Paid This Month ($)"].sum()),
        "outstanding": float(d["Vendor Invoices Outstanding ($)"].iloc[-1]),
    }


# ---------- like-for-like pricing archetypes ----------
def compare_pricing(df, p, r):
    n = len(df)
    h = p["heavy_share"]
    pro_acc = df["Average Pro Subs"].to_numpy()
    ent_acc = df["Average Ent Accounts"].to_numpy()
    typ_p = df["Typical Pro Prompts"].to_numpy()
    hvy_p = df["Heavy Pro Prompts"].to_numpy()
    ent_p = df["Enterprise Prompts"].to_numpy()
    paid_p = typ_p + hvy_p + ent_p
    paid_cost, free_cost = cost_allocation(df)
    account_months = float(pro_acc.sum() + ent_acc.sum())

    def per_acct(prompts, accts):
        return np.divide(prompts, accts, out=np.zeros(n), where=accts > 1e-12)

    def overage(prompts, accts, allowance, price):
        use = per_acct(prompts, accts)
        return float((np.maximum(0.0, use - allowance) * accts).sum() * price)

    # segment-level overage: typical, heavy and enterprise measured separately (no pooling)
    def sub_like(pro_fee, pro_allow, ent_fee, ent_allow, ovg):
        base = float(pro_acc.sum() * pro_fee + ent_acc.sum() * ent_fee)
        over = (overage(typ_p, pro_acc * (1 - h), pro_allow, ovg) + overage(hvy_p, pro_acc * h, pro_allow, ovg)
                + overage(ent_p, ent_acc, ent_allow, ovg))
        return base + over

    basis_per_prompt = "Per paid prompt (Pro + Enterprise); no seat, account or expansion revenue"
    basis_fee = "Fixed fee per average contracted account + overage per prompt; excludes the expansion uplift"
    revenues = [
        ("1. Pay-as-you-go", float(paid_p.sum() * r["payg_rate"]), "Low (usage-driven)", "Low commitment; revenue volatility", basis_per_prompt),
        ("2. Subscription + allowance", sub_like(r["sub_pro_fee"], r["sub_pro_allow"], r["sub_ent_fee"], r["sub_ent_allow"], r["sub_over"]),
         "High (base fee)", "Allowance / overage boundary friction", basis_fee),
        ("3. Hybrid base + overage", sub_like(r["hyb_pro_fee"], r["hyb_pro_allow"], r["hyb_ent_fee"], r["hyb_ent_allow"], r["hyb_over"]),
         "High (base + usage)", "Requires clear meter and overage UX", basis_fee),
        ("4. Prepaid credit burndown",
         float(paid_p.sum() / r["prepaid_ppc"] * r["prepaid_price"] / max(1e-9, 1.0 - r["prepaid_breakage"])) if r["prepaid_ppc"] > 0 else 0.0,
         "Upfront cash; recognition varies", "Breakage, expiry and replenishment risk",
         "Prepaid price per credit pack, grossed up for breakage; no seat, account or expansion revenue"),
        ("0. Current model (subscriptions, ex-implementation)", float(df["Subscription Revenue ($)"].sum()),
         "High (contracted + subscription)", "Reference row: the revenue model driving the rest of the dashboard",
         "ARPU per average Pro sub and per expansion-weighted enterprise account; INCLUDES the Pro overage/expansion uplift and the enterprise expansion index"),
    ]
    out = []
    for label, rev, pred, risk, basis in revenues:
        fees = rev * p["fee_pct"]
        contrib = rev - paid_cost - fees
        after = contrib - free_cost
        out.append({
            "Pricing Archetype": label,
            "Simulated Revenue ($)": rev,
            "Paid-Usage Cost ($)": paid_cost,
            "Payment Fees ($)": fees,
            "Contribution Margin ($)": contrib,
            "Contribution Margin (%)": contrib / rev * 100 if rev > 0 else np.nan,
            "Free-Tier & Idle Cost ($)": free_cost,
            "Gross Profit After Free-Tier Cost ($)": after,
            "Gross Margin After Free-Tier Cost (%)": after / rev * 100 if rev > 0 else np.nan,
            "Revenue / Account-Month ($)": rev / account_months if account_months > 0 else np.nan,
            "Billing Basis": basis,
            "Revenue Predictability": pred,
            "Key Risk": risk,
        })
    res = pd.DataFrame(out).sort_values("Pricing Archetype").reset_index(drop=True)
    return res


# ---------- automated integrity checks ----------
def run_integrity_checks(df, p, base_totals=None, pricing_df=None):
    checks = []

    def add(name, ok, detail="", applicable=True):
        result = ("PASS" if ok else "FAIL") if applicable else "N/A"
        checks.append({"Check": name, "Result": result, "Detail": detail})

    def close(a, b):
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9, atol=1e-6))

    add("Gross profit = revenue - COGS (every month)", close(df["Modeled Gross Profit ($)"], df["Recognized Revenue ($)"] - df["Total Modeled COGS ($)"]))
    comp = (df["API Cost ($)"] + df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"] + df["Payment Processing ($)"] + df["Implementation Cost ($)"])
    add("COGS = API + vector + infra + payment fees + implementation cost", close(df["Total Modeled COGS ($)"], comp))
    add("Revenue = subscriptions + implementation recognized", close(df["Recognized Revenue ($)"], df["Subscription Revenue ($)"] + df["Implementation Revenue ($)"]))
    add("Pro roll-forward: beginning + conversions - churn = ending",
        close(df["Beginning Pro Subs"] + df["New-User Conversions"] + df["Free-Pool Conversions"] - df["Churned Pro Subs"], df["Ending Pro Subs"]))
    add("Pro continuity: beginning(t) = ending(t-1)", close(df["Beginning Pro Subs"].iloc[1:], df["Ending Pro Subs"].iloc[:-1]))
    add("Enterprise roll-forward: beginning + new - non-renewals - in-term churn = ending",
        close(df["Beginning Ent Accounts"] + df["New Ent Deals"] - df["Ent Non-Renewals"] - df["Ent In-Term Churn"], df["Ending Ent Accounts"]))
    add("Enterprise continuity: beginning(t) = ending(t-1)", close(df["Beginning Ent Accounts"].iloc[1:], df["Ending Ent Accounts"].iloc[:-1]))
    add("Starting enterprise accounts tie to input", abs(float(df["Beginning Ent Accounts"].iloc[0]) - p["ent"]) < 1e-6)
    add("Prompt segments sum to total", close(df["Typical Pro Prompts"] + df["Heavy Pro Prompts"] + df["Enterprise Prompts"] + df["Free Prompts"], df["Monthly Prompts"]))
    for tag, start, o, a, e, rem in [("AI", p["ai_credits"], "Opening AI Credits ($)", "AI Credits Applied ($)", "AI Credits Expired ($)", "Remaining AI Credits ($)"),
                                     ("Cloud", p["cloud_credits"], "Opening Cloud Credits ($)", "Cloud Credits Applied ($)", "Cloud Credits Expired ($)", "Remaining Cloud Credits ($)")]:
        add(f"{tag} credits: opening - expired - applied = remaining", close(df[o] - df[e] - df[a], df[rem]))
        add(f"{tag} credits: opening(t+1) = remaining(t); month 1 = input", close(df[o].iloc[1:], df[rem].iloc[:-1]) and abs(float(df[o].iloc[0]) - start) < 1e-6)
    add("Credits applied never exceed eligible cost",
        bool((df["AI Credits Applied ($)"] <= df["API Cost ($)"] * p["ai_elig"] + 1e-6).all()
             and (df["Cloud Credits Applied ($)"] <= (df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"]) * p["cloud_elig"] + 1e-6).all()))
    add("Cash payable = COGS - credits applied", close(df["Estimated Cash Payable ($)"], df["Total Modeled COGS ($)"] - df["Total Credits Applied ($)"]))
    add("Cash reconciliation: cash paid + outstanding invoices at horizon = cash payable",
        close(df["Estimated Cash Paid This Month ($)"].sum() + df["Vendor Invoices Outstanding ($)"].iloc[-1], df["Estimated Cash Payable ($)"].sum()),
        f"lag = {int(p['invoice_lag'])} month(s)")
    add("Implementation revenue: billed - recognized = deferred balance",
        close(df["Implementation Fees Billed ($)"].cumsum() - df["Implementation Revenue ($)"].cumsum(), df["Deferred Implementation Revenue ($)"]))
    paid_cost, free_cost = cost_allocation(df)
    pool = float((df["API Cost ($)"] + df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"]).sum())
    add("Cost allocation: paid + free traffic cost = API + vector + infra", abs(paid_cost + free_cost - pool) < 1e-4 * max(1.0, pool))
    must_be_nonneg = ["DAU", "Beginning Pro Subs", "Ending Pro Subs", "Ending Ent Accounts", "Ending Paid Accounts", "Monthly Prompts",
                      "Free Prompts", "Recognized Revenue ($)", "API Cost ($)", "Vector DB Cost ($)", "Fixed Infrastructure ($)",
                      "Total Modeled COGS ($)", "Remaining AI Credits ($)", "Remaining Cloud Credits ($)", "Estimated Cash Payable ($)",
                      "Estimated Cash Paid This Month ($)", "Vendor Invoices Outstanding ($)", "Deferred Implementation Revenue ($)"]
    add("No negative balances, counts, prompts, costs or revenues", bool((df[must_be_nonneg] >= -1e-9).all().all()))
    # Usage ceiling: at the ceiling, per-account margin must equal the target. Always listed; N/A (never silently skipped)
    # when no finite ceiling exists for these inputs (e.g. zero traffic, or fees leave no headroom at any tested target).
    tested, worst = [], 0.0
    for tgt in (40.0, 60.0, 80.0):
        uc = usage_ceiling(df, p, tgt)
        if math.isfinite(uc["var"]) and not uc["no_headroom"] and uc["var_cost"] > 0:
            implied = (uc["rev"] * (1 - p["fee_pct"]) - uc["var"] * uc["var_cost"]) / uc["rev"] * 100
            tested.append(tgt)
            worst = max(worst, abs(implied - tgt))
    add("Usage ceiling solver: per-account margin at the ceiling equals the target",
        worst < 1e-6, f"tested at targets {tested}; worst gap {worst:.2e}pp" if tested else "no finite ceiling for these inputs",
        applicable=bool(tested))
    if base_totals is not None:
        t = totals(df)
        add("Scenario base case reconciles to monthly detail (revenue, COGS, ARR, cash)",
            all(abs(base_totals[k] - t[k]) < 1e-6 * max(1.0, abs(t[k])) for k in ("rev", "cogs", "arr", "payable", "paid")))
    if pricing_df is not None:
        ref = pricing_df[pricing_df["Pricing Archetype"].str.startswith("0.")]
        add("Pricing reference row revenue = model subscription revenue", close(ref["Simulated Revenue ($)"], [df["Subscription Revenue ($)"].sum()]))
        add("Pricing rows share one cost basis (paid-usage cost identical)", float(pricing_df["Paid-Usage Cost ($)"].nunique()) == 1.0)
    return pd.DataFrame(checks)


# ==========================================
# 3. RUN, GUARDS, KPIs & BRIEFING
# ==========================================
_act = P["paid_dau_factor"]
_start_active_paid = starting_pro_subs * _act + starting_ent_accounts * P["ent_seats"] * _act
if _start_active_paid > starting_dau:
    st.warning(f"Active paid users implied by your inputs (about {_start_active_paid:,.0f}: Pro subscribers plus enterprise seats, "
               f"each x {_act * 100:.0f}% daily activity) exceed month-1 DAU ({starting_dau:,.0f}). "
               "Confirm that DAU, Pro subscribers and enterprise seats share one definition; the free pool is floored at zero.")
if starting_dau == 0 and starting_pro_subs + starting_ent_accounts > 0:
    st.warning("You have paid accounts but zero DAU. Confirm that DAU and paid-account counts use compatible definitions.")
if advanced_mode and (cfg["p1_vol"] < baseline_prompts or cfg["p2_vol"] < baseline_prompts):
    st.sidebar.warning(f"A peak volume is below baseline ({baseline_prompts}); peaks are floored at baseline.")

try:
    df = run_model_simulation(P).copy()
except ValueError as e:
    st.error(f"⚠️ Model Configuration Error: {e}")
    st.stop()
except Exception as e:  # unexpected calculation error: show it plainly instead of a raw traceback
    st.error(f"⚠️ Unexpected calculation error: {type(e).__name__}: {e}")
    st.stop()

df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = (df["Total Modeled COGS ($)"] - budget_monthly) if budget_monthly > 0 else np.nan
_prompts_safe = df["Monthly Prompts"].replace(0, np.nan)
df["API Cost / 1,000 Prompts ($)"] = df["API Cost ($)"] / _prompts_safe * 1000
df["Blended COGS / 1,000 Prompts ($)"] = df["Total Modeled COGS ($)"] / _prompts_safe * 1000
df["Blended COGS / Paid Account ($)"] = df["Total Modeled COGS ($)"] / df["Ending Paid Accounts"].replace(0, np.nan)

m_last = df.iloc[-1]
tot = totals(df)
last_arr = tot["arr"]
period_gm = tot["gm"]
last_margin = float(m_last["Modeled Gross Margin (%)"])
H = len(df)
h_mult_used = heavy_multiplier(P)
ceil = usage_ceiling(df, P, margin_target)
paid_cost_alloc, free_cost_alloc = cost_allocation(df)
free_share = float(df["Free Prompts"].sum() / df["Monthly Prompts"].sum()) if df["Monthly Prompts"].sum() > 0 else 0.0


def fmt_prompts(v):
    return "No cap (zero marginal cost)" if not math.isfinite(v) else f"{v:,.0f} prompts/mo"


c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{H})", f"${last_arr:,.0f} ARR", f"Forecast-period revenue: ${tot['rev']:,.0f}")
c2.metric("Forecast-Period Gross Margin", "n/a (no revenue)" if pd.isna(period_gm) else f"{period_gm:.1f}%",
          f"Target: {margin_target}%" if pd.isna(period_gm) else f"{period_gm - margin_target:+.1f}pp vs {margin_target}% target")
c3.metric(f"Est. Pro Usage Ceiling @ {margin_target}% GM (M{H})",
          "No headroom" if ceil["no_headroom"] else fmt_prompts(ceil["var"]),
          "Estimate, not a safe limit" if ceil["no_headroom"] else f"Fully loaded: {fmt_prompts(ceil['loaded'])} (estimate)",
          help="Estimated from the ending month's company-wide cost per prompt. Capacity steps, customer mix and real routing can move marginal cost; do not treat it as a guaranteed safe usage limit.")
c4.metric("Heavy-User Cohort", f"{P['heavy_share'] * 100:.0f}% of Pro users", f"{h_mult_used:.1f}x typical prompts")

bullets = []
if pd.isna(period_gm):
    bullets.append("<b>Gross Margin Status:</b> No recognized revenue in the forecast horizon, so margin is undefined. Direct COGS is incurred with no revenue offset.")
elif period_gm < margin_target:
    bullets.append(f"<b>Margin Alert:</b> Forecast-period gross margin <b>{period_gm:.1f}%</b> is below the <b>{margin_target}%</b> target (ending month: "
                   f"{'n/a' if pd.isna(last_margin) else f'{last_margin:.1f}%'}). <b>Action:</b> test a usage cap, overage price or lower-cost routing.")
else:
    bullets.append(f"<b>Margin Compliant:</b> Forecast-period gross margin <b>{period_gm:.1f}%</b> meets the <b>{margin_target}%</b> target "
                   f"(ending month: {'n/a' if pd.isna(last_margin) else f'{last_margin:.1f}%'}).")
breach = df.loc[df["Modeled Gross Margin (%)"] < margin_target, "Month"]
if len(breach):
    bullets.append(f"<b>First Monthly Breach:</b> Monthly margin first falls below target in <b>{breach.iloc[0]}</b> ({len(breach)} of {H} months below target).")
if budget_monthly > 0:
    over = df.loc[df["Budget Variance ($)"] > 0, "Month"]
    if len(over):
        bullets.append(f"<b>Budget Overrun:</b> Monthly COGS exceeds the <b>${budget_monthly:,.0f}</b> cap from <b>{over.iloc[0]}</b> (cumulative variance +${df['Budget Variance ($)'].clip(lower=0).sum():,.0f}).")
    else:
        bullets.append(f"<b>Budget:</b> Monthly COGS stays within the ${budget_monthly:,.0f} cap in all {H} months.")
ai_stat = credit_status(df, P["ai_credits"], "Opening AI Credits ($)", "AI Credits Expired ($)", "Remaining AI Credits ($)")
cloud_stat = credit_status(df, P["cloud_credits"], "Opening Cloud Credits ($)", "Cloud Credits Expired ($)", "Remaining Cloud Credits ($)")
bullets.append(f"<b>Credit Runway:</b> AI credits: {ai_stat}. Cloud credits: {cloud_stat}.")
due_total = float(df["Ent Accounts Up For Renewal"].sum())
if due_total > 0:
    bullets.append(f"<b>Enterprise Renewals:</b> {due_total:,.1f} account-renewals fall due in the horizon; at {P['ent_renewal_rate'] * 100:.0f}% renewal about "
                   f"{df['Ent Non-Renewals'].sum():,.1f} accounts are lost.")
if df["Monthly Prompts"].sum() > 0:
    bullets.append(f"<b>Free-Tier Cost-to-Serve:</b> Free users generate {free_share * 100:.0f}% of prompts and ≈${free_cost_alloc:,.0f} of API + vector + infrastructure cost "
                   f"({free_cost_alloc / max(1.0, paid_cost_alloc + free_cost_alloc) * 100:.0f}% of the pool), carried by paid revenue.")
st.markdown(f"""
<div class="cfo-commentary">
    <h4>💡 Deal Desk & Contract Pricing Governance Briefing</h4>
    <ul>{"".join(f"<li>{b}</li>" for b in bullets)}</ul>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Revenue & Margin", "💳 Credits & Cash", "📋 Monthly Detail",
    "📈 Scenario Matrix", "📑 Audit Register", "🎯 Value & Pricing Strategy",
])

with tab1:
    st.subheader("Forecast Recognized Revenue, COGS & Gross Margin")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    if budget_monthly > 0:
        fig.add_trace(go.Scatter(x=df["Month"], y=[budget_monthly] * H, name="COGS Budget Cap ($)", mode="lines", line=dict(color="#dd6b20", width=1.5, dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["Month"], y=[margin_target] * H, name=f"Target Margin ({margin_target}%)", mode="lines", line=dict(color="#a0aec0", width=1.5, dash="dot")), secondary_y=True)
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="USD ($ / Month)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True)
    show_chart(fig)
    st.caption("Gross margin here is a management metric: payment processing fees and implementation delivery cost are included in COGS.")

    st.markdown("#### Founder Unit-Economics Snapshot (ending month)")
    k1, k2, k3, k4 = st.columns(4)
    end_paid = float(m_last["Ending Paid Accounts"])
    cogs_pp = float(m_last["Total Modeled COGS ($)"]) / end_paid if end_paid > 0 else 0.0
    rev_pp = float(m_last["Recognized Revenue ($)"]) / end_paid if end_paid > 0 else 0.0
    k1.metric("COGS / paid account", f"${cogs_pp:,.2f}")
    k2.metric("Revenue / paid account", f"${rev_pp:,.2f}")
    k3.metric("Gross profit / paid account", f"${rev_pp - cogs_pp:,.2f}")
    k4.metric("Modeled COGS", f"${float(m_last['Total Modeled COGS ($)']):,.0f}")
    st.caption("Blended company-level averages across Pro subscribers and enterprise accounts (an enterprise account is a contract, not a user).")

    st.markdown("#### Inference Unit Cost Telemetry")
    cu1, cu2 = st.columns(2)
    fu1 = go.Figure()
    fu1.add_trace(go.Scatter(x=df["Month"], y=df["API Cost / 1,000 Prompts ($)"], name="API-only / 1k", mode="lines+markers", line=dict(color="#805ad5", width=2.5)))
    fu1.add_trace(go.Scatter(x=df["Month"], y=df["Blended COGS / 1,000 Prompts ($)"], name="Blended COGS / 1k", mode="lines+markers", line=dict(color="#e53e3e", width=2, dash="dash")))
    fu1.update_layout(title="API Cost vs Blended COGS per 1,000 Prompts ($)", template="plotly_white", height=250, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.18))
    with cu1:
        show_chart(fu1)
    fu2 = go.Figure()
    fu2.add_trace(go.Scatter(x=df["Month"], y=df["Blended COGS / Paid Account ($)"], mode="lines+markers", name="COGS / paid account", line=dict(color="#319795", width=2.5)))
    fu2.update_layout(title="Blended COGS / Paid Account ($)", template="plotly_white", height=250, margin=dict(l=30, r=30, t=40, b=20))
    with cu2:
        show_chart(fu2)

    st.markdown("#### COGS Component Disaggregation")
    cogs_fig = go.Figure()
    for col, nm, colr in [("API Cost ($)", "API Inference", "#4299e1"), ("Vector DB Cost ($)", "Vector DB", "#4fd1c5"),
                          ("Fixed Infrastructure ($)", "Infrastructure Step-Cost", "#ed8936"),
                          ("Payment Processing ($)", "Payment Fees", "#9f7aea"), ("Implementation Cost ($)", "Implementation Delivery", "#a0aec0")]:
        cogs_fig.add_trace(go.Bar(x=df["Month"], y=df[col], name=nm, marker_color=colr))
    cogs_fig.update_layout(template="plotly_white", barmode="stack", height=320, margin=dict(l=30, r=30, t=20, b=20), legend=dict(orientation="h", y=1.12))
    show_chart(cogs_fig)

    st.markdown("#### Enterprise Account Roll-Forward")
    fe = make_subplots(specs=[[{"secondary_y": True}]])
    fe.add_trace(go.Bar(x=df["Month"], y=df["Ending Ent Accounts"], name="Ending Accounts", marker_color="#3182ce"), secondary_y=False)
    fe.add_trace(go.Bar(x=df["Month"], y=-df["Ent Non-Renewals"], name="Non-Renewals", marker_color="#e53e3e"), secondary_y=False)
    fe.add_trace(go.Scatter(x=df["Month"], y=df["Ent Expansion Index"], name="Expansion Index (x)", mode="lines+markers", line=dict(color="#27965a", width=2)), secondary_y=True)
    fe.update_layout(template="plotly_white", barmode="relative", height=300, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.15))
    show_chart(fe)

with tab2:
    st.subheader("Credit Drawdown Velocity, Cash Payable & Cash Paid")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["AI Credits Applied ($)"], name="AI Credits Applied ($)", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Cloud Credits Applied ($)"], name="Cloud Credits Applied ($)", marker_color="#4fd1c5"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Cash Payable, accrual ($)", marker_color="#dd6b20"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Estimated Cash Paid This Month ($)"], name="Estimated Cash Paid ($)", mode="lines+markers", line=dict(color="#805ad5", width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color="#2b6cb0", width=2, dash="dash")), secondary_y=True)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Cloud Credits ($)"], name="Remaining Cloud Credits ($)", mode="lines", line=dict(color="#805ad5", width=2, dash="dot")), secondary_y=True)
    fig2.update_layout(template="plotly_white", height=380, barmode="stack", margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.15))
    fig2.update_yaxes(title_text="Monthly COGS coverage ($)", secondary_y=False, rangemode="tozero")
    fig2.update_yaxes(title_text="Credit balances ($)", secondary_y=True, rangemode="tozero")
    show_chart(fig2)
    st.caption("Credits reduce cash payable but NOT modeled gross profit (underlying unit economics). 'Cash Paid' shifts vendor invoices by the payment lag; payment fees and implementation costs are paid immediately.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI Credits", ai_stat)
    m2.metric("Cloud Credits", cloud_stat)
    m3.metric("Vendor Invoices Outstanding at Horizon", f"${float(m_last['Vendor Invoices Outstanding ($)']):,.0f}")
    m4.metric("Revenue Less Net Vendor Payables & Immediate Costs (horizon)",
              f"${float(df['Revenue Less Net Vendor Payables & Immediate Costs ($)'].sum()):,.0f}",
              help="Recognized revenue minus estimated cash payable (vendor invoices net of credits, plus payment fees and implementation costs). Not cash flow: invoice timing and customer collections are not modeled.")

with tab3:
    st.subheader("Monthly Roll-Forward Detail")
    show_df(style_frame(df), hide_index=True)
    st.download_button("Download Full Audit CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="ai_inference_deal_desk_model.csv", mime="text/csv")

with tab4:
    st.subheader("📈 Multi-Scenario Comparative Analysis Matrix")
    st.caption("Each scenario moves one driver family unless labeled combined. Prompt intensity scales ALL segments, including enterprise. Deal counts are not rounded. "
               "These are simplified sensitivities on one operating plan, not independent plans.")
    scen_rows = []
    scen_totals = {}
    for name, spec in SCENARIOS:
        try:
            sd = run_model_simulation(apply_scenario(P, spec))
        except Exception as e:
            st.error(f"Scenario '{name}' failed: {e}")
            st.stop()
        scen_totals[name] = totals(sd)
    base_t = scen_totals[SCENARIOS[0][0]]
    for idx, (name, spec) in enumerate(SCENARIOS, start=1):
        t = scen_totals[name]
        scen_rows.append({
            "ID": idx, "Scenario": name,
            "Revenue ($)": t["rev"], "Δ Revenue vs Base ($)": t["rev"] - base_t["rev"],
            "COGS ($)": t["cogs"], "Δ COGS vs Base ($)": t["cogs"] - base_t["cogs"],
            "Gross Profit ($)": t["gp"], "Gross Margin (%)": t["gm"],
            "Δ Margin vs Base (pp)": (t["gm"] - base_t["gm"]) if not (pd.isna(t["gm"]) or pd.isna(base_t["gm"])) else np.nan,
            "Ending ARR ($)": t["arr"],
            "Cash Payable, Horizon ($)": t["payable"], "Cash Paid, Horizon ($)": t["paid"],
            "Δ Cash Payable vs Base ($)": t["payable"] - base_t["payable"],
            "Invoices Outstanding at Horizon ($)": t["outstanding"],
        })
    scen_df = pd.DataFrame(scen_rows)
    show_df(style_frame(scen_df), hide_index=True)

with tab5:
    st.subheader("📑 Assumptions & Audit Register (Data Governance)")
    st.caption("Generated from the live inputs, so values cannot drift from the model. 'Status' is an honest label: nothing here has been verified against actuals by this tool.")
    PRESET_EST = "Founder-profile preset (estimate) — replace with actuals"
    ASSUME = "Model assumption — verify"
    audit = [
        ("Forecast start / horizon", f"{pd.Period(freq='M', year=start_year, month=start_month).strftime('%b %Y')} / {H} months", "FP&A planning calendar", "Planning choice"),
        ("Starting DAU", f"{starting_dau:,.0f}", "Mixpanel / Segment / PostHog (define DAU consistently)", PRESET_EST),
        ("Starting Pro subscribers", f"{starting_pro_subs:,.0f}", "Billing system (Stripe) active subscriptions", PRESET_EST),
        ("Starting enterprise accounts", f"{starting_ent_accounts:,.0f}", "Contract register / CRM", PRESET_EST),
        ("Total DAU monthly growth", f"{dau_monthly_growth * 100:.1f}%", "Acquisition model / analytics trend", PRESET_EST),
        ("Net new enterprise deals / month", f"{net_new_ent_contracts:,.2f}", "Sales pipeline (CRM) weighted forecast", PRESET_EST),
        ("Baseline prompts / active paid user / day", f"{baseline_prompts}", "LLM gateway logs (Helicone / Langfuse)", PRESET_EST),
        ("Target gross margin / COGS budget cap", f"{margin_target}% / " + (f"${budget_monthly:,.0f}" if budget_monthly > 0 else "none"), "Board / finance policy", "Policy input"),
        ("Paid-user daily activity factor", f"{P['paid_dau_factor'] * 100:.0f}%", "Product analytics (paid DAU / paid subs)", ASSUME),
        ("New-user conversion to Pro", f"{P['new_conv'] * 100:.1f}%", "Funnel analytics, cohort-based", ASSUME),
        ("Existing free-pool conversion / month", f"{P['base_conv'] * 100:.1f}%", "Funnel analytics, cohort-based", ASSUME),
        ("Pro monthly churn", f"{P['churn'] * 100:.1f}%", "Billing cohort retention", ASSUME),
        ("Pro ARPU / overage-expansion", f"${P['arpu_pro']:,.2f} / +{P['pro_expansion_pct'] * 100:.0f}%", "Price book / metered billing invoices", ASSUME),
        ("Enterprise contract value / account / month", f"${P['arpu_ent']:,.2f}", "Signed order forms", ASSUME),
        ("Enterprise term / renewal rate / in-term churn", f"{P['ent_term']} mo / {P['ent_renewal_rate'] * 100:.0f}% / {P['ent_in_term_churn'] * 100:.1f}%", "Contract register; historical renewals", ASSUME),
        ("Enterprise renewal-month exemption", "Cohorts at term end face the renewal decision only; in-term churn is not also applied to them that month", "Contract terms (termination for cause, insolvency)", ASSUME),
        ("Enterprise expansion / seats / prompts per account", f"{P['ent_expansion'] * 100:.1f}% mo / {P['ent_seats']} seats / {P['ent_prompts']:,} prompts", "Account usage telemetry", ASSUME),
        ("Implementation fee / recognition / delivery cost", f"${P['impl_fee']:,.0f} / {P['impl_rec_months']} mo / ${P['impl_cost']:,.0f}", "Order forms; revenue-recognition policy; PS cost model", ASSUME),
        ("Payment processing fee", f"{P['fee_pct'] * 100:.1f}%", "Processor merchant statement", ASSUME),
        ("Usage peaks (calendar months)", f"{calendar.month_name[P['p1_month']]} @ {P['p1_vol']} / {calendar.month_name[P['p2_month']]} @ {P['p2_vol']}; σ {P['p1_width']} / {P['p2_width']}", "Seasonality in gateway logs", ASSUME),
        ("Heavy-user share / calibration", f"{P['heavy_share'] * 100:.0f}% / " + (f"{P['heavy_mult']:.1f}x multiplier" if P["heavy_mode"] == "Multiplier" else f"Pareto {P['pareto_share'] * 100:.0f}% of prompts (derived {h_mult_used:.1f}x)"), "Per-user prompt distribution", ASSUME),
        ("Free-tier usage multiplier", f"{P['free_mult'] * 100:.0f}% of typical", "Gateway logs by plan", ASSUME),
        ("Tokens / prompt (in / out) and MoM growth", f"{P['in_tok']:,} / {P['out_tok']:,}; +{P['in_grow'] * 100:.1f}% / +{P['out_grow'] * 100:.1f}%", "LLM provider logs", ASSUME),
        ("Frontier routing share and prices", f"{P['front_mix'] * 100:.0f}%; ${P['f_in']:.2f} in / ${P['f_out']:.2f} out per 1M", "Vendor rate card (public, unverified)", ASSUME),
        ("Standard model prices", f"${P['s_in']:.4f} in / ${P['s_out']:.4f} out per 1M", "Vendor rate card (public, unverified)", ASSUME),
        ("Prompt caching", f"{P['cache_rate'] * 100:.0f}% hit / {P['cache_ratio'] * 100:.0f}% price", "Gateway cache logs; vendor terms", ASSUME),
        ("Batch routing / inference overhead", f"{P['batch_share'] * 100:.0f}% at {P['batch_ratio'] * 100:.0f}% price / +{P['overhead'] * 100:.0f}%", "Batch API telemetry", ASSUME),
        ("Base infrastructure / step block / step cost", f"${P['base_infra']:,.0f} / {P['step_thresh']:,.0f} ({P['step_type']}) / ${P['step_cost']:,.0f}", "Cloud invoices; capacity plan", PRESET_EST),
        ("Vector DB cost / 1k prompts", f"${P['vec_cost']:.3f}", "Vector vendor invoices", ASSUME),
        ("Starting credits (AI / Cloud)", f"${P['ai_credits']:,.0f} / ${P['cloud_credits']:,.0f}", "Grant agreements and vendor portals", PRESET_EST),
        ("Credit expiry month / eligible share", f"AI {P['ai_expiry'] or 'none'} @ {P['ai_elig'] * 100:.0f}% / Cloud {P['cloud_expiry'] or 'none'} @ {P['cloud_elig'] * 100:.0f}%", "Grant terms", ASSUME),
        ("Vendor invoice payment lag", f"{P['invoice_lag']} month(s)", "AP terms with vendors", ASSUME),
    ]
    show_df(pd.DataFrame(audit, columns=["Model Parameter", "Current Value", "Verify Against (System of Record)", "Status"]), hide_index=True)

    st.markdown("#### Model Integrity Checks")
    st.caption("Reconciliations that run on every recalculation. Any FAIL means the model output should not be used until investigated.")
    # computed here with the pricing frame from tab 6 defaults so checks always exist
    _default_rules = {"payg_rate": 0.008, "sub_pro_fee": 30.0, "sub_pro_allow": 1500, "sub_ent_fee": 250.0, "sub_ent_allow": 3000, "sub_over": 0.006,
                      "hyb_pro_fee": 20.0, "hyb_pro_allow": 1000, "hyb_ent_fee": 150.0, "hyb_ent_allow": 2000, "hyb_over": 0.008,
                      "prepaid_price": 1.0, "prepaid_ppc": 100.0, "prepaid_breakage": 0.10}
    checks_df = run_integrity_checks(df, P, base_totals=base_t, pricing_df=compare_pricing(df, P, _default_rules))
    n_fail = int((checks_df["Result"] == "FAIL").sum())
    n_na = int((checks_df["Result"] == "N/A").sum())
    n_pass = int((checks_df["Result"] == "PASS").sum())
    _na_txt = f" ({n_na} not applicable to these inputs)" if n_na else ""
    (st.success if n_fail == 0 else st.error)(
        f"{n_pass} of {n_pass + n_fail} applicable integrity checks passed{_na_txt}." if n_fail == 0 else f"{n_fail} integrity check(s) FAILED{_na_txt}.")
    st.caption("These checks confirm the arithmetic is internally consistent. They cannot prove an assumption is right: "
               "a consistently wrong input passes every check. Known-answer tests (hand-calculated cases) live in tests/test_engine.py.")
    show_df(checks_df, hide_index=True)

with tab6:
    st.subheader("🎯 Value-Creation Anchor & Hybrid Pricing Strategy")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 1. Value-Creation Anchor Calculator")
        value_metric_type = st.selectbox("Primary AI Value Metric", ["Automation (Hours Saved)", "Augmentation (Error Reduction)", "Connection (Network Access)", "Revenue (Tasks / Leads Generated)"])
        if value_metric_type == "Automation (Hours Saved)":
            raw_value = st.number_input("Hours Saved / User / Month", min_value=0.0, value=10.0, step=1.0) * st.number_input("Average User Hourly Wage ($)", min_value=0.0, value=50.0, step=5.0)
        elif value_metric_type == "Augmentation (Error Reduction)":
            raw_value = st.number_input("Errors / Rework Incidents Avoided / User / Month", min_value=0.0, value=5.0, step=1.0) * st.number_input("Average Cost per Error ($)", min_value=0.0, value=40.0, step=5.0)
        elif value_metric_type == "Connection (Network Access)":
            raw_value = st.number_input("Qualified Connections / Introductions / User / Month", min_value=0.0, value=4.0, step=1.0) * st.number_input("Value per Connection ($)", min_value=0.0, value=25.0, step=5.0)
        else:
            raw_value = (st.number_input("Tasks / Leads Generated / User / Month", min_value=0.0, value=8.0, step=1.0)
                         * st.slider("Conversion / Close Rate (%)", 0, 100, 10) / 100
                         * st.number_input("Gross Profit per Conversion ($)", min_value=0.0, value=500.0, step=50.0))
        attribution = st.slider("Share of Value Attributable to the AI Product (%)", 0, 100, 100) / 100
        economic_value = raw_value * attribution
        st.info(f"💡 Economic value created: **${economic_value:,.0f} / user / month**")
        price_basis = st.selectbox("Price Basis", ["Pro (per user)", "Enterprise (per seat)"])
        price_charged = P["arpu_pro"] if price_basis.startswith("Pro") else P["arpu_ent"] / max(1, P["ent_seats"])
        capture_text = f"{price_charged / economic_value * 100:.1f}%" if economic_value > 0 else "N/A (no value created)"
        roi_text = f"{economic_value / price_charged:.1f}x" if price_charged > 0 else "N/A (price is zero)"
        st.markdown(f"""
        * **Price charged ({price_basis}):** `${price_charged:,.2f} / user / mo`
        * **Value capture ratio:** `{capture_text}` of value created
        * **Customer ROI multiple:** `{roi_text}`
        """)

    with col_v2:
        st.markdown("#### 2. Fixed-Usage Pricing Archetype Comparison")
        st.caption("This is a sensitivity on constant customer behavior, not a forecast of what customers would buy: usage, plan mix and churn are held the same under every price. "
                   "All archetypes are applied to the SAME simulated customers and usage, segment by segment (typical Pro, heavy Pro, enterprise). "
                   "Cost is the same API + vector + infrastructure pool allocated by prompt volume; payment fees are recomputed on each archetype's revenue. "
                   "Free-tier traffic (and any idle capacity in months with no traffic) earns no revenue, so its cost is shown separately.")
        with st.expander("Edit comparable pricing rules", expanded=False):
            rules = {
                "payg_rate": st.number_input("Pay-as-you-go price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f"),
                "sub_pro_fee": st.number_input("Subscription: Pro base fee / account / month ($)", min_value=0.0, value=30.0, step=5.0),
                "sub_pro_allow": st.number_input("Subscription: Pro included prompts / account / month", min_value=0, value=1500, step=250),
                "sub_ent_fee": st.number_input("Subscription: Enterprise base fee / account / month ($)", min_value=0.0, value=250.0, step=25.0),
                "sub_ent_allow": st.number_input("Subscription: Enterprise included prompts / account / month", min_value=0, value=3000, step=250),
                "sub_over": st.number_input("Subscription: overage price per prompt ($)", min_value=0.0, value=0.006, step=0.001, format="%.4f"),
                "hyb_pro_fee": st.number_input("Hybrid: Pro base fee / account / month ($)", min_value=0.0, value=20.0, step=5.0),
                "hyb_pro_allow": st.number_input("Hybrid: Pro included prompts / account / month", min_value=0, value=1000, step=250),
                "hyb_ent_fee": st.number_input("Hybrid: Enterprise base fee / account / month ($)", min_value=0.0, value=150.0, step=25.0),
                "hyb_ent_allow": st.number_input("Hybrid: Enterprise included prompts / account / month", min_value=0, value=2000, step=250),
                "hyb_over": st.number_input("Hybrid: overage price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f"),
                "prepaid_price": st.number_input("Prepaid price per credit ($)", min_value=0.0, value=1.0, step=0.25),
                "prepaid_ppc": st.number_input("Prompts covered per credit", min_value=0.01, value=100.0, step=10.0),
                "prepaid_breakage": st.slider("Prepaid breakage (% of purchased credits never used)", 0, 50, 10) / 100,
            }
        pricing_df = compare_pricing(df, P, rules)
        show_df(style_frame(pricing_df), hide_index=True)
        st.caption("Billing Basis shows what each row charges for. Rows 1-4 do not include the Pro overage/expansion uplift or the enterprise expansion index that the reference row 0 carries, so row 0 is not a like-for-like benchmark for revenue; compare margins and unit economics with that in mind. "
                   "Prepaid revenue = value of credits purchased (consumed + expired breakage); cash timing and deferred-revenue treatment are not modeled. "
                   "Contribution margin is before free-tier and idle cost; gross margin after free-tier cost is the comparable company-level figure.")
        st.markdown(f"**Estimated Pro usage ceiling at {margin_target}% target margin (month {H}):** "
                    + ("no headroom at this target and fee rate." if ceil["no_headroom"]
                       else f"{fmt_prompts(ceil['var'])} on variable cost (API + vector); {fmt_prompts(ceil['loaded'])} fully loaded with allocated infrastructure. "
                            f"Current average Pro usage that month: "
                            f"{(float(m_last['Typical Pro Prompts'] + m_last['Heavy Pro Prompts']) / max(1e-9, float(m_last['Average Pro Subs']))):,.0f} prompts per subscriber."))
        st.caption("An estimate from company-wide cost per prompt, not a guaranteed safe limit. Capacity steps, customer mix and real routing can move marginal cost.")

# ==========================================
# 5. FOOTER, CHANGE LOG & LIMITATIONS
# ==========================================
st.markdown("---")
# Change log & limitations are part of the page but not shown on screen: they sit in a display:none block (still in the
# page's DOM, so it can be inspected or flipped on by changing SHOW_CHANGE_LOG). Set True to show them in an expander.
SHOW_CHANGE_LOG = False
CHANGE_LOG_MD = """
**Material changes from v1**
1. **Enterprise engine rebuilt as cohorts.** Accounts, renewals, non-renewals, in-term churn and expansion are tracked per cohort and reconcile to the account roll-forward. The starting book is a *staggered* book (evenly spread over remaining terms), so renewals arrive monthly and there is no end-of-horizon cliff. Renewal rate and in-term churn are explicit inputs; the hidden "50% of Pro churn" rule is gone.
2. **Expansion is no longer double counted.** Enterprise expansion grows contract value, seats and prompts of existing accounts. The 10% overage/expansion uplift applies to Pro revenue only.
3. **Accounts are not users.** Enterprise accounts carry contracted seats; only active seats count toward DAU and the free pool. Free-pool conversion is capped at the pool and measured against paid *active* users.
4. **Seasonality uses calendar months** (wrapping around the year), not forecast-month index. Peaks are floored at baseline. The peak controls no longer depend on the horizon, which removes the 1-month-horizon crash.
5. **Like-for-like pricing comparison.** Same customers, per-segment allowances (no pooling), enterprise priced on enterprise fees, one shared cost pool, fees recomputed per archetype, free-tier and idle-capacity cost shown separately, prepaid breakage, and a reference row for the current model.
6. **Usage-ceiling solver rebuilt.** Uses the ending month's own marginal cost per prompt and per-account Pro revenue; reports variable-cost and fully-loaded ceilings.
7. **Margin states.** No-revenue is "n/a", never "healthy". The advisory states which margin it tests (forecast-period) and also shows ending-month margin.
8. **Scenarios** move one driver family each, scale all prompt segments, do not round deals, include conversion and churn/renewal shocks, and report cash paid and outstanding invoices.
9. **Pareto skew implemented** as an alternative heavy-user calibration; **budget cap** now drives variance, chart line and commentary.
10. **Credit and cash.** One credit-status function feeds both the briefing and Tab 2 (expired is not "exhausted"). Vendor invoices lag; payment fees and implementation costs are paid immediately. Cash reconciliation is checked.
11. **Implementation fees** have an explicit recognition period (default upfront) and optional delivery cost; deferred balance is tracked.
12. **Restored from the earlier version:** CSV download, margin-target and budget lines, unit-cost telemetry, per-month breach commentary, credit status KPIs, scenario deltas.
13. **Audit register** is generated from live inputs with honest status labels; **integrity checks** run on every recalculation. Display no longer rounds model columns, so tables foot.

**v2.1 (peer-review follow-ups)**
14. **DAU convention made explicit.** The input is "Month-1 DAU", used exactly as entered; growth applies from month 2. Month 1 therefore has no new-user conversions, but free-pool conversions still occur.
15. **Cash metric renamed** to "Revenue Less Net Vendor Payables & Immediate Costs". It is accrual revenue less estimated payables, not cash profit; invoice timing and collections are not modeled.
16. **Pricing comparison relabeled** "Fixed-Usage": a sensitivity on constant customer behavior, not a forecast of what customers would buy.
17. **Enterprise expansion timing documented** (starts immediately, compounds monthly). **Usage ceiling labeled an estimate** from company-wide cost per prompt.
18. **Starting-accounts warning** now compares active paid users (Pro subscribers plus enterprise seats times daily activity) to month-1 DAU, not contract counts. **ROI and capture ratios** show N/A instead of 0 when undefined.
19. **Integrity checks** now report N/A (not silent skips) when not applicable, test the usage ceiling at three targets, and point to known-answer tests in `tests/test_engine.py`.
20. **Pricing comparison has a Billing Basis column** saying what each row charges for, and a caption that the reference row includes expansion revenue the alternatives do not. **Enterprise renewal-month exemption:** cohorts at term end are not also hit by in-term churn that month; it is listed in the assumptions register.

**Simplifications that remain**
- Net DAU growth is treated as new-user acquisition (gross sign-ups and DAU churn are not modeled); free-user conversion is a monthly rate on the free active pool, not a full cohort table. Month-1 DAU is taken as entered; growth starts in month 2.
- The pricing comparison holds usage, plan mix and churn constant under every price; it does not model demand response.
- Enterprise expansion starts immediately and compounds monthly; real expansion arrives through amendments and seat ramps.
- "Revenue Less Net Vendor Payables & Immediate Costs" is not cash flow: customer collections and invoice-lag timing on the revenue side are not modeled.
- **Renewal-month exemption (model assumption).** A cohort that reaches term end in a month faces only the renewal decision that month; in-term churn is not also applied to it. Every other cohort, and the renewed cohort in later months, is exposed to in-term churn. In-term churn defaults to 0%, so this has no effect unless you set it above zero.
- Enterprise accounts renew at one blended rate on one blended contract value; discounts, ramped usage, committed minimums, overage billing and account-level deals are not modeled.
- Token mix, caching and batch shares are identical across customer segments; costs per prompt are therefore equal across segments.
- Infrastructure steps use equal-size capacity blocks; real vendor pricing may be tiered or continuous.
- Payment fees are applied to recognized revenue; promotional credits reduce cash payable but not gross profit. Accounting treatment depends on the vendor agreement and policy.
- Presets, prices and rates are illustrative estimates until verified against billing exports and vendor portals.
"""
if SHOW_CHANGE_LOG:
    with st.expander("Change log & limitations (v2.1)"):
        st.markdown(CHANGE_LOG_MD)
else:
    # one line, escaped: a blank line would end the HTML block and leak the text onto the screen
    _hidden = html.escape(CHANGE_LOG_MD.strip()).replace("\n\n", "<br>").replace("\n", "<br>")
    st.markdown(f'<div id="change-log-and-limitations" aria-hidden="true" style="display:none">{_hidden}</div>', unsafe_allow_html=True)

st.caption("© 2026. Institutional AI Deal Desk & Contract Pricing Simulator. All rights reserved.")
