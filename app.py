"""
AI Inference, Credit Depletion & Unit Economics Model  (Streamlit UI)
---------------------------------------------------------------------
Driver-based FP&A planning model for evaluating direct AI model inference,
vector storage, and infrastructure COGS. Excludes non-COGS OpEx (payroll, marketing, G&A).
Prices and assumptions are user-editable estimates; verify against vendor billing portals.

Single-file app: the forecast engine (Assumptions dataclass + pure functions) sits at the top,
followed by the Streamlit UI.
Run with:  streamlit run app.py

Revision notes (v4):
  - Configurable forecast horizon (6/12/18/24 months); all labels and charts follow it.
  - Vector DB cost now driven by AVERAGE DAU (same basis as usage); step-cost triggers stay end-of-month (capacity).
  - Cache key now includes an engine fingerprint, so editing the engine can never serve stale cached results.
  - Optional linear vs compound token growth; "Reset to defaults" button; "Copy assumptions as JSON" expander.
  - Peak-month slider shows the calendar month it maps to; warning if starting paid subs exceed starting DAU;
    prominent warning when batch + cache discounts are both on (assumed to stack).
  - Pricing source / as-of note feeds the Assumptions Register; register now self-checks against the dataclass fields.
  - Scenario table highlights the Base Case; chart keys added; CSV rounds money to cents and names the start month.
  - Removed dead code (STEP_TRIGGERS now drives the selectbox; unused annual_cogs removed).

Revision notes (v3):
  - 37 positional engine args replaced by an Assumptions dataclass (all in this one file).
  - Default forecast start is the next full calendar month (no partly-elapsed first month).
  - Gross vs. net acquisition: new optional free-user churn input raises gross adds needed for net DAU growth.
  - Budget overrun total now sums overrun months only (underspend no longer nets out).
  - Removed unreachable "efficiency improves" commentary branch.
  - "Blended COGS / 1k prompts" renamed "Ops COGS / 1k prompts" and now excludes payment fees.
  - Softened unverified vendor facts in tooltips.
(v2: Month-1 double-count fix, unrounded subscriber carry-forward, average-sub usage basis,
     step-cost ceil fix, scenario-7 fix, overhead + payment-fee inputs.)
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ==========================================
# FORECAST ENGINE (pure functions, no Streamlit calls)
# ==========================================
STEP_TRIGGERS = ("DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)")


@dataclass(frozen=True)
class Assumptions:
    # Horizon
    start_year: int = 2026
    start_month: int = 11
    horizon: int = 12               # forecast length in months
    # Funnel (DAU = total daily active users, INCLUDING paid subscribers)
    starting_dau: float = 5000.0
    start_paid_subs: float = 250.0
    growth: float = 0.08            # net MoM DAU growth
    new_conv: float = 0.15          # conversion of gross new users (applies M2+)
    base_conv: float = 0.02         # monthly conversion of the opening free pool
    churn: float = 0.03             # paid churn
    arpu: float = 30.0
    free_churn: float = 0.0         # monthly churn of free users (raises gross adds needed for net DAU growth)
    # Usage
    base_prompts: float = 5.0
    peak_prompts: float = 35.0
    peak_month: int = 7             # forecast month index (M1..M<horizon>)
    surge_width: float = 2.0
    free_mult: float = 0.5
    in_tok: float = 1000.0
    out_tok: float = 400.0
    in_growth: float = 0.03
    out_growth: float = 0.0
    token_growth_mode: str = "Compound"   # "Compound" or "Linear"
    # Routing & prices ($ per 1M tokens)
    frontier_mix: float = 0.30
    f_in_price: float = 2.50
    f_out_price: float = 10.00
    s_in_price: float = 0.15
    s_out_price: float = 0.60
    cache_rate: float = 0.40
    cache_price_ratio: float = 0.20
    batch_share: float = 0.20
    batch_price_ratio: float = 0.50
    overhead: float = 0.0           # uplift on API cost (retries, guardrails, tool loops, embeddings)
    # Infrastructure & credits
    base_infra: float = 2500.0
    step_type: str = "DAU Threshold"
    step_threshold: float = 20000.0
    step_cost: float = 1500.0
    vec_cost: float = 0.15
    ai_credits: float = 20000.0
    cloud_credits: float = 30000.0
    ai_elig: float = 1.0
    cloud_elig: float = 1.0
    fee_pct: float = 0.029          # payment processing, % of recognized revenue


def default_start(today: date) -> tuple[int, int]:
    """First full calendar month after `today` (so the first forecast month isn't partly elapsed)."""
    if today.month == 12:
        return today.year + 1, 1
    return today.year, today.month + 1


def run_model_simulation(a: Assumptions) -> pd.DataFrame:
    n = max(1, min(int(a.horizon), 36))
    dates = [pd.Period(freq="M", year=a.start_year, month=a.start_month) + i for i in range(n)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    m_idx = np.arange(1, n + 1)

    width = max(float(a.surge_width), 1e-6)
    prompt_curve = a.base_prompts + (a.peak_prompts - a.base_prompts) * np.exp(
        -((m_idx - a.peak_month) ** 2) / (2 * width ** 2)
    )

    ai_credits_left = float(a.ai_credits)
    cloud_credits_left = float(a.cloud_credits)

    current_dau = float(a.starting_dau)
    prev_end_subs = float(a.start_paid_subs)  # unrounded carry-forward

    rows = []
    for i in range(n):
        beg_paid_subs = prev_end_subs

        if i == 0:
            dau_val = current_dau
            # Starting DAU is the OBSERVED base (already reflected in starting paid subs): there is no
            # brand-new cohort in Month 1, and growth / free-user churn start affecting the path in Month 2.
            # (The DAU path itself is an input, so free churn never shrinks DAU - it only raises gross adds.)
            gross_new_users = 0.0
            free_base = max(0.0, dau_val - beg_paid_subs)
        else:
            prev_dau = current_dau
            current_dau = prev_dau * (1 + a.growth)
            dau_val = current_dau
            prev_free = max(0.0, prev_dau - beg_paid_subs)
            free_churned = a.free_churn * prev_free
            # Gross adds = net DAU growth + free users lost to churn
            gross_new_users = max(0.0, (dau_val - prev_dau) + free_churned)
            # Design decision: base conversion applies to the OPENING free pool (last month's free users).
            # Users acquired this month can convert only via new_conv (Day-1); those who do not convert
            # stay in DAU and enter next month's opening pool, so nothing is dropped.
            free_base = prev_free

        new_user_conv_subs = gross_new_users * a.new_conv
        free_base_conv_subs = free_base * a.base_conv
        churned_subs = beg_paid_subs * a.churn
        ending_paid_subs = max(0.0, beg_paid_subs + new_user_conv_subs + free_base_conv_subs - churned_subs)
        prev_end_subs = ending_paid_subs

        # Average active paid subs drives BOTH revenue and usage (consistent basis)
        avg_active_paid_subs = (beg_paid_subs + ending_paid_subs) / 2.0
        free_users = max(0.0, dau_val - ending_paid_subs)
        free_users_avg = max(0.0, dau_val - avg_active_paid_subs)

        if a.token_growth_mode == "Linear":
            in_tokens = a.in_tok * (1 + a.in_growth * i)
            out_tokens = a.out_tok * (1 + a.out_growth * i)
        else:  # Compound
            in_tokens = a.in_tok * ((1 + a.in_growth) ** i)
            out_tokens = a.out_tok * ((1 + a.out_growth) ** i)

        paid_ppd = float(prompt_curve[i])
        free_ppd = paid_ppd * a.free_mult
        prompts_paid = avg_active_paid_subs * paid_ppd * month_days[i]
        prompts_free = free_users_avg * free_ppd * month_days[i]
        prompts_total = prompts_paid + prompts_free

        raw_in_m = prompts_total * in_tokens / 1_000_000
        raw_out_m = prompts_total * out_tokens / 1_000_000
        uncached_in_m = raw_in_m * (1 - a.cache_rate)
        cached_in_m = raw_in_m * a.cache_rate

        std_mix = 1.0 - a.frontier_mix
        f_in = (uncached_in_m * a.f_in_price + cached_in_m * a.f_in_price * a.cache_price_ratio) * a.frontier_mix
        s_in = (uncached_in_m * a.s_in_price + cached_in_m * a.s_in_price * a.cache_price_ratio) * std_mix
        f_out = raw_out_m * a.f_out_price * a.frontier_mix
        s_out = raw_out_m * a.s_out_price * std_mix

        api_before_batch = f_in + s_in + f_out + s_out
        api_cost = api_before_batch * ((1 - a.batch_share) + a.batch_share * a.batch_price_ratio) * (1 + a.overhead)
        # Average DAU over the month (same basis as usage). Month 1 has no prior month, so it uses starting DAU.
        avg_dau = dau_val if i == 0 else (prev_dau + dau_val) / 2.0
        vector_cost = avg_dau * a.vec_cost

        if a.step_type == "DAU Threshold":
            trigger = dau_val
        elif a.step_type == "Paid Subscribers Threshold":
            trigger = ending_paid_subs
        else:
            trigger = prompts_total

        # Step triggers use END-of-month capacity on purpose (infrastructure is provisioned for the peak).
        # Base infra covers Block 1; one extra step per additional block beyond it.
        if a.step_threshold > 0 and trigger > a.step_threshold:
            steps = max(0, math.ceil(trigger / a.step_threshold) - 1)
        else:
            steps = 0
        infra_cost = a.base_infra + steps * a.step_cost

        revenue = avg_active_paid_subs * a.arpu
        mrr = ending_paid_subs * a.arpu
        fees = revenue * a.fee_pct

        total_cogs = api_cost + vector_cost + infra_cost + fees
        gross_profit = revenue - total_cogs
        gm_pct = (gross_profit / revenue * 100) if revenue > 0 else np.nan

        # Dual-pool credit depletion (payment fees are not credit-eligible).
        # applied <= balance by construction, so balances can never go negative; the 0.005 test in the UI
        # is only a display tolerance for 'fully depleted'.
        ai_applied = min(ai_credits_left, api_cost * a.ai_elig)
        ai_credits_left = max(0.0, ai_credits_left - ai_applied)
        cloud_applied = min(cloud_credits_left, (vector_cost + infra_cost) * a.cloud_elig)
        cloud_credits_left = max(0.0, cloud_credits_left - cloud_applied)
        credits_applied = ai_applied + cloud_applied

        rows.append({
            "Month": month_names[i],
            "Calendar Days": month_days[i],
            "DAU": round(dau_val),
            "Gross New Users": round(gross_new_users),
            "Beginning Paid Subscribers": round(beg_paid_subs),
            "New User Conversions": round(new_user_conv_subs),
            "Free Base Conversions": round(free_base_conv_subs),
            "Churned Subscribers": round(churned_subs),
            "Ending Paid Subscribers": round(ending_paid_subs),
            "Avg Active Paid Subs": round(avg_active_paid_subs, 1),
            "Free Users": round(free_users),
            "Monthly Prompts": round(prompts_total),
            "Ending MRR Run-Rate ($)": mrr,
            "Recognized Revenue ($)": revenue,
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost,
            "Fixed Infrastructure ($)": infra_cost,
            "Payment Processing ($)": fees,
            "Total Modeled COGS ($)": total_cogs,
            "Modeled Gross Profit ($)": gross_profit,
            "Modeled Gross Margin (%)": gm_pct,
            "AI Credits Applied ($)": ai_applied,
            "Remaining AI Credits ($)": ai_credits_left,
            "Cloud Credits Applied ($)": cloud_applied,
            "Remaining Cloud Credits ($)": cloud_credits_left,
            "Total Credits Applied ($)": credits_applied,
            "Estimated Cash Payable ($)": total_cogs - credits_applied,
        })

    return pd.DataFrame(rows)


def add_kpis(df: pd.DataFrame, budget_monthly: float) -> pd.DataFrame:
    """Add budget and unit-cost telemetry columns."""
    out = df.copy()
    if budget_monthly > 0:
        variance = out["Total Modeled COGS ($)"] - budget_monthly
        out["Budget ($)"] = budget_monthly
        out["Budget Variance ($)"] = variance
        out["Budget Overrun ($)"] = variance.clip(lower=0.0)  # overrun months only; underspend does not net out
    else:
        out["Budget ($)"] = np.nan
        out["Budget Variance ($)"] = np.nan
        out["Budget Overrun ($)"] = np.nan

    prompts = out["Monthly Prompts"].replace(0, np.nan)
    ops_cogs = out["API Cost ($)"] + out["Vector DB Cost ($)"] + out["Fixed Infrastructure ($)"]
    out["API Cost / 1,000 Prompts ($)"] = out["API Cost ($)"] / prompts * 1000
    # Excludes payment fees (those scale with revenue, not prompts)
    out["Ops COGS / 1,000 Prompts ($)"] = ops_cogs / prompts * 1000
    out["Blended COGS / Paid Sub ($)"] = out["Total Modeled COGS ($)"] / out["Ending Paid Subscribers"].replace(0, np.nan)
    return out


def build_scenarios(a: Assumptions) -> list[tuple[str, Assumptions]]:
    """Eight reforecast scenarios derived from the base assumptions."""
    def tok(m: float) -> dict:
        return {"in_tok": a.in_tok * m, "out_tok": a.out_tok * m}

    def price(m: float) -> dict:
        return {
            "f_in_price": a.f_in_price * m, "f_out_price": a.f_out_price * m,
            "s_in_price": a.s_in_price * m, "s_out_price": a.s_out_price * m,
        }

    return [
        ("1. Base Case", a),
        ("2. Downside: Growth Slowdown (-50%)", replace(a, growth=a.growth * 0.5)),
        ("3. Downside: High Subscriber Churn (2x)", replace(a, churn=a.churn * 2.0)),
        ("4. Downside: Double Whammy (Low Conv + High Churn)",
         replace(a, growth=a.growth * 0.5, churn=a.churn * 2.0, new_conv=a.new_conv * 0.5, base_conv=a.base_conv * 0.5)),
        ("5. Cost Pressure: Context Expansion (+25% Tokens)", replace(a, **tok(1.25))),
        ("6. Efficiency: Prompt Pruning (-25% Tokens)", replace(a, **tok(0.75))),
        # Optimization can only lower the frontier share, never raise it above the base case
        ("7. Efficiency: Routing & Caching Optimization",
         replace(a, frontier_mix=min(a.frontier_mix, 0.20), cache_rate=min(0.90, a.cache_rate * 1.5))),
        ("8. Market Shift: Vendor Rate Cut (-25% Price)", replace(a, **price(0.75))),
    ]


st.set_page_config(
    page_title="AI Inference, Credit Depletion & Unit Economics Model",
    page_icon="⚡",
    layout="wide"
)


def _engine_fingerprint() -> str:
    """Hash of the engine source. st.cache_data only tracks the cached function's own code, so without this an
    edit to run_model_simulation / Assumptions could keep serving stale cached results."""
    try:
        text = inspect.getsource(run_model_simulation) + inspect.getsource(Assumptions)
    except (OSError, TypeError):
        return "unknown"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


ENGINE_FINGERPRINT = _engine_fingerprint()


@st.cache_data(show_spinner=False)
def _cached_run(params: dict, engine_fingerprint: str) -> pd.DataFrame:
    # engine_fingerprint is unused in the body on purpose: it is part of the cache key.
    return run_model_simulation(Assumptions(**params))


def simulate(a: Assumptions) -> pd.DataFrame:
    return _cached_run(asdict(a), ENGINE_FINGERPRINT).copy()


# Custom Soft Neutral Styling & Compact Elements
st.markdown("""
<style>
.stApp {background:#fcfbf9;color:#2d3748;font-family:'Inter', sans-serif;}
section[data-testid="stSidebar"] {background:#f5f2ed !important; border-right: 1px solid #e2d9d0;}

/* Compact Metric Cards */
div[data-testid="stMetric"] {
    background:#fff;
    border:1px solid #e2d9d0;
    padding:8px 12px !important;
    border-radius:6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {font-size:0.78rem !important; color:#615a52 !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.15rem !important; font-weight:700;}

/* Compact Header Banner */
.compact-header {
    background: linear-gradient(135deg,#f5f0eb,#ebe5df);
    padding: 10px 16px;
    border-radius: 8px;
    border: 1px solid #e2d9d0;
    margin-bottom: 10px;
}
.compact-header h1 {color:#2d3748; margin:0 0 2px 0; font-size:1.3rem; font-weight:700;}
.compact-header p {color:#615a52; margin:0; font-size:0.82rem;}

/* Executive Commentary Box */
.cfo-commentary {
    background: #ffffff;
    border: 1px solid #cbd5e0;
    border-left: 4px solid #3182ce;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 14px;
}
.cfo-commentary h4 {margin:0 0 6px 0; font-size:0.95rem; color:#2b6cb0; font-weight:700;}
.cfo-commentary ul {margin:0; padding-left:18px; font-size:0.85rem; color:#4a5568;}
.cfo-commentary li {margin-bottom:3px;}

/* Alert Cards */
.alert-card {padding:6px 10px; border-radius:5px; font-size:0.80rem; margin-bottom:10px; font-weight:500;}
.alert-warning {background:#fffaf0; border:1px solid #feebc8; color:#9c4221;}
.alert-error {background:#fff5f5; border:1px solid #fed7d7; color:#9b2c2c;}
.alert-info {background:#ebf8ff; border:1px solid #bee3f8; color:#2c5282;}
.scope-note {font-size:0.75rem; color:#718096; font-style:italic; margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# Compact Header
st.markdown("""
<div class="compact-header">
    <h1>⚡ AI Inference, Credit Depletion & Unit Economics Model</h1>
    <p>Driver-based forecast of token spend, modular step-costs, dual-funnel PLG subscriber bridge, and credit depletion.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR ASSUMPTIONS & HOVER TOOLTIPS
# ==========================================


def _reset_inputs():
    st.session_state.clear()


st.sidebar.button("↺ Reset all inputs to defaults", on_click=_reset_inputs,
                  help="Clears every sidebar input back to its default.")
st.sidebar.header("0. Rolling Forecast Horizon")
d_year, d_month = default_start(datetime.date.today())
start_year = st.sidebar.number_input(
    "Forecast Start Year", min_value=2024, max_value=2035, value=min(max(d_year, 2024), 2035),
    help="Defaults to the next full calendar month so the first forecast month is not partly elapsed."
)
start_month = st.sidebar.slider("Forecast Start Month", 1, 12, d_month)
horizon = st.sidebar.selectbox(
    "Forecast Horizon (Months)", [6, 12, 18, 24], index=1,
    help="Length of the forecast. Charts, KPIs and the scenario matrix follow this setting."
)

st.sidebar.header("1. User Acquisition & Paid Funnel")
starting_dau = st.sidebar.number_input(
    "Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500,
    help="Total daily active users at Month 1, INCLUDING paid subscribers (free users = DAU - paid subs). "
         "Source: Mixpanel, Google Analytics, or PostHog."
)
starting_paid_subs = st.sidebar.number_input(
    "Starting Paid Subscribers", min_value=0, value=250, step=50,
    help="Initial paid subscriber count. Assumed to be a subset of starting DAU."
)
if starting_paid_subs > starting_dau:
    st.sidebar.warning("⚠️ Starting paid subscribers exceed starting DAU. The model treats this as zero free users; "
                       "check that DAU includes paid subscribers.")
monthly_growth = st.sidebar.slider(
    "Monthly DAU Growth (%)", 0.0, 25.0, 8.0, 0.5,
    help="Net MoM growth of total DAU (gross adds minus free-user churn)."
) / 100

free_user_churn = st.sidebar.slider(
    "Monthly Free-User Churn (%)", 0.0, 30.0, 0.0, 0.5,
    help="Free users who stop being active each month. Raises GROSS new users needed to hit the net DAU growth above, "
         "which raises new-user conversions. 0% = net growth equals gross adds."
) / 100

new_user_conversion = st.sidebar.slider(
    "New User Conversion (%)", 0.0, 50.0, 15.0, 0.5,
    help="% of gross new users who upgrade to paid in their first month (Day-1 checkout conversion). "
         "Applies from Month 2 onward; the starting base is already reflected in starting paid subs."
) / 100

free_base_conversion = st.sidebar.slider(
    "Free Base Conversion (%)", 0.0, 10.0, 2.0, 0.1,
    help="% of the opening free-user pool (prior-month free users) who convert to paid each month."
) / 100

monthly_churn = st.sidebar.slider(
    "Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5,
    help="Percentage of paid subscribers cancelling each month."
) / 100

arpu = st.sidebar.number_input(
    "Monthly ARPU per Paid Subscriber ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average Revenue Per User per month for paid tiers."
)

payment_fee_pct = st.sidebar.slider(
    "Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1,
    help="Card-processor / app-store fees recognized in COGS. Card fees are often around 3% plus a fixed per-charge fee "
         "(not modeled); app-store fees can be far higher. Confirm with your processor. Not eligible for credits."
) / 100

st.sidebar.header("2. Usage Patterns & Token Context")
baseline_prompts = st.sidebar.slider(
    "Baseline Prompts / Paid User / Day", 1, 20, 5,
    help="Average daily prompt volume for paid power users during normal activity."
)
peak_prompts_input = st.sidebar.slider(
    "Peak Prompts / Paid User / Day", 1, 100, 35,
    help="Expected peak usage surge per paid user during heavy usage or launch event."
)

# Input Guardrail Enforcement
if peak_prompts_input < baseline_prompts:
    st.sidebar.warning(f"⚠️ Peak prompts ({peak_prompts_input}) cannot be below baseline ({baseline_prompts}). Enforcing peak = {baseline_prompts}.")
    peak_prompts = baseline_prompts
else:
    peak_prompts = peak_prompts_input

peak_month = st.sidebar.slider(
    "Peak Usage Forecast Month (M#)", 1, int(horizon), min(7, int(horizon)),
    help="Forecast month index (M1 = the forecast start month), NOT the calendar month."
)
_peak_label = (pd.Period(freq="M", year=int(start_year), month=int(start_month)) + (int(peak_month) - 1)).strftime("%b %Y")
_start_label = pd.Period(freq="M", year=int(start_year), month=int(start_month)).strftime("%b %Y")
st.sidebar.caption(f"Peak = M{peak_month} = {_peak_label} (M1 = {_start_label}).")
surge_width = st.sidebar.slider("Peak Usage Spread (Months)", 1.0, 4.0, 2.0, 0.5)

free_user_usage_mult = st.sidebar.slider(
    "Free Tier Usage Multiplier (%)", 0, 100, 50,
    help="Usage volume of free users relative to paid users."
) / 100

input_tokens_start = st.sidebar.number_input(
    "Initial Input Tokens / Prompt", min_value=0, value=1000, step=100,
    help="Average prompt context length. Includes system prompt (~300t), RAG context (~300t), and chat history (~400t)."
)
output_tokens_start = st.sidebar.number_input(
    "Initial Output Tokens / Prompt", min_value=0, value=400, step=50,
    help="Average billed completion tokens per prompt. Include hidden reasoning/thinking tokens if your models bill them as output."
)
input_growth = st.sidebar.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
output_growth = st.sidebar.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100
token_growth_mode = st.sidebar.selectbox(
    "Token Growth Mode", ["Compound", "Linear"],
    help="Compound: tokens x (1 + g)^month. Linear: tokens x (1 + g x month)."
)

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider("Frontier Model Traffic Share (%)", 0, 100, 30, help="Simplified routing assumption for premium models.") / 100
pricing_note = st.sidebar.text_input(
    "Pricing source (model names + as-of date)", value="", placeholder="e.g. vendor list price, as of YYYY-MM-DD",
    help="Recorded in the Assumptions Register. Leave blank to flag the prices below as unverified placeholders."
)

frontier_input_price = st.sidebar.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25,
                                               help="Placeholder default. Replace with the current rate card for the specific model you route to.")
frontier_output_price = st.sidebar.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50)
standard_input_price = st.sidebar.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
standard_output_price = st.sidebar.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f")

cache_hit_rate = st.sidebar.slider(
    "Eligible Input Tokens Cached (%)", 0, 90, 40,
    help="Share of input tokens billed at the cached rate. Providers typically require a minimum cacheable prefix "
         "(commonly on the order of 1,000+ tokens; confirm per model), so a short static system prompt may not qualify. "
         "Cache-write premiums are not modeled."
) / 100
cache_price_ratio = st.sidebar.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
batch_share = st.sidebar.slider(
    "Traffic Using Batch Pricing (%)", 0, 80, 20,
    help="Batch APIs are typically asynchronous (results can take hours). Only count background / non-interactive traffic. "
         "Batch and caching discounts are assumed to stack; confirm with your vendor."
) / 100
batch_price_ratio = st.sidebar.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100
if batch_share > 0 and cache_hit_rate > 0:
    st.sidebar.warning("⚠️ Batch and caching discounts are assumed to STACK on the same tokens. Many vendors do not "
                       "allow both; verify with your vendor or set one of them to 0%.")

overhead_pct = st.sidebar.slider(
    "Inference Overhead (%)", 0, 100, 0,
    help="Uplift on API cost for retries, guardrail/moderation calls, tool-call loops, and embeddings not captured in token counts. "
         "Defaults to 0%; set it from your gateway logs."
) / 100

st.sidebar.header("4. Infrastructure Step-Costs & Credits")
base_fixed_infra = st.sidebar.number_input(
    "Base Fixed Monthly Infra ($)", min_value=0.0, value=2500.0, step=500.0,
    help="Base hosting cost covering Tier 1 capacity (Block 1 up to threshold). Additional step costs apply beyond threshold."
)

step_trigger_type = st.sidebar.selectbox(
    "Step-Cost Trigger Basis",
    list(STEP_TRIGGERS),
    help="Select basis for triggering modular infrastructure upgrades beyond base capacity (measured at end of month)."
)

if step_trigger_type == "DAU Threshold":
    step_threshold_input = st.sidebar.number_input("DAU Capacity Block (Users)", min_value=1000, value=20000, step=5000)
    raw_step_threshold = float(step_threshold_input)
    step_block_label = f"{step_threshold_input:,.0f} DAU"
elif step_trigger_type == "Paid Subscribers Threshold":
    step_threshold_input = st.sidebar.number_input("Subscriber Capacity Block (Subs)", min_value=500, value=5000, step=1000)
    raw_step_threshold = float(step_threshold_input)
    step_block_label = f"{step_threshold_input:,.0f} paid subscribers"
else:
    step_threshold_input = st.sidebar.number_input("Monthly Prompt Block (Millions)", min_value=1, value=20, step=5)
    raw_step_threshold = float(step_threshold_input * 1_000_000)
    step_block_label = f"{step_threshold_input:,.0f}M monthly prompts"

step_cost_increment = st.sidebar.number_input("Infra Step-Up Cost ($ per block)", min_value=0.0, value=1500.0, step=250.0)

vector_cost_per_user = st.sidebar.number_input(
    "Vector DB Cost / DAU / Month ($)", min_value=0.0, value=0.15, step=0.05,
    help="Monthly vector database storage and query cost per daily active user."
)

starting_ai_credits = st.sidebar.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=20000.0, step=2500.0,
                                              help="Credit expiry dates are not modeled; confirm grant terms.")
starting_cloud_credits = st.sidebar.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=30000.0, step=5000.0)
ai_credit_eligible_share = st.sidebar.slider("AI Credit Eligible Share (%)", 0, 100, 100) / 100
cloud_credit_eligible_share = st.sidebar.slider("Cloud Credit Eligible Share (%)", 0, 100, 100) / 100

margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

assumptions = Assumptions(
    start_year=int(start_year), start_month=int(start_month), horizon=int(horizon),
    starting_dau=float(starting_dau), start_paid_subs=float(starting_paid_subs),
    growth=monthly_growth, new_conv=new_user_conversion, base_conv=free_base_conversion,
    churn=monthly_churn, arpu=float(arpu), free_churn=free_user_churn,
    base_prompts=float(baseline_prompts), peak_prompts=float(peak_prompts), peak_month=int(peak_month),
    surge_width=float(surge_width), free_mult=free_user_usage_mult,
    in_tok=float(input_tokens_start), out_tok=float(output_tokens_start),
    in_growth=input_growth, out_growth=output_growth, token_growth_mode=token_growth_mode,
    frontier_mix=frontier_mix,
    f_in_price=float(frontier_input_price), f_out_price=float(frontier_output_price),
    s_in_price=float(standard_input_price), s_out_price=float(standard_output_price),
    cache_rate=cache_hit_rate, cache_price_ratio=cache_price_ratio,
    batch_share=batch_share, batch_price_ratio=batch_price_ratio, overhead=overhead_pct,
    base_infra=float(base_fixed_infra), step_type=step_trigger_type,
    step_threshold=raw_step_threshold, step_cost=float(step_cost_increment),
    vec_cost=float(vector_cost_per_user),
    ai_credits=float(starting_ai_credits), cloud_credits=float(starting_cloud_credits),
    ai_elig=ai_credit_eligible_share, cloud_elig=cloud_credit_eligible_share,
    fee_pct=payment_fee_pct,
)

with st.sidebar.expander("Copy assumptions as JSON"):
    st.code(
        json.dumps({**asdict(assumptions), "margin_target_pct": margin_target, "budget_monthly": budget_monthly,
                    "pricing_source": pricing_note}, indent=2),
        language="json",
    )

# Run Baseline Simulation
df = add_kpis(simulate(assumptions), budget_monthly)

# ==========================================
# 3. CFO DASHBOARD KPI PANEL & DECISION ENGINE
# ==========================================
n = len(df)
horizon_revenue = df["Recognized Revenue ($)"].sum()
last = df.iloc[-1]
last_arr = last["Ending MRR Run-Rate ($)"] * 12

ai_dep_idx = df.index[df["Remaining AI Credits ($)"] <= 0.005].tolist() if starting_ai_credits > 0 else []
cloud_dep_idx = df.index[df["Remaining Cloud Credits ($)"] <= 0.005].tolist() if starting_cloud_credits > 0 else []

ai_dep_str = f"Depleted in {df.loc[ai_dep_idx[0], 'Month']}" if ai_dep_idx else ("None ($0)" if starting_ai_credits <= 0 else f"Active through M{n}")
cloud_dep_str = f"Depleted in {df.loc[cloud_dep_idx[0], 'Month']}" if cloud_dep_idx else ("None ($0)" if starting_cloud_credits <= 0 else f"Active through M{n}")

last_margin = last["Modeled Gross Margin (%)"]
if pd.isna(last_margin):
    margin_kpi_str = "Pre-Revenue"
    margin_sub_str = "Gross Loss ($0 Rev)"
else:
    margin_var = last_margin - margin_target
    margin_kpi_str = f"{last_margin:.1f}%"
    margin_sub_str = f"{margin_var:+.1f}% vs {margin_target}% Target"

c_first_api = df.loc[0, "API Cost / 1,000 Prompts ($)"]
c_last_api = last["API Cost / 1,000 Prompts ($)"]
if pd.notna(c_first_api) and pd.notna(c_last_api) and c_first_api > 0:
    prompt_api_cost_delta = (c_last_api - c_first_api) / c_first_api * 100
else:
    prompt_api_cost_delta = 0.0
api_kpi_str = f"${c_last_api:.4f} / 1k" if pd.notna(c_last_api) else "n/a (no prompts)"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"1. ARR Run-Rate (M{n})", f"${last_arr:,.0f} ARR", f"Recognized {n}M Rev: ${horizon_revenue:,.0f}")
c2.metric(f"2. Gross Margin (M{n})", margin_kpi_str, margin_sub_str)
c3.metric("3. Pure API Efficiency", api_kpi_str, f"{prompt_api_cost_delta:+.1f}% M1 to M{n}")
c4.metric("4. Credit Depletion Horizon", f"AI: {ai_dep_str}", f"Cloud: {cloud_dep_str}")
st.caption("ARR run-rate = ending paid subscribers x ARPU x 12. Recognized revenue uses AVERAGE paid subscribers each month, "
           "so the two diverge when growth or churn is high.")

# DYNAMIC CFO COMMENTARY ENGINE
bullets = []
valid_margins = df["Modeled Gross Margin (%)"].dropna()
if valid_margins.empty:
    bullets.append("<b>Gross Margin Status:</b> The model is generating a gross loss in a pre-revenue phase ($0 revenue). Direct COGS exceed revenues.")
else:
    breach = df.index[df["Modeled Gross Margin (%)"].notna() & (df["Modeled Gross Margin (%)"] < margin_target)].tolist()
    if breach:
        breach_m = df.loc[breach[0], "Month"]
        bullets.append(f"<b>Margin Alert:</b> Gross margin drops below the <b>{margin_target}% target</b> in <b>{breach_m}</b>. Potential drivers include context token growth and infrastructure step-costs.")
    else:
        bullets.append(f"<b>Margin Compliant:</b> Gross margin remains above the <b>{margin_target}% target</b> across all {n} forecast months.")

# Budget Breach Commentary (overrun months only)
if budget_monthly > 0:
    b_breach = df.index[df["Budget Overrun ($)"] > 0].tolist()
    if b_breach:
        total_overrun = df["Budget Overrun ($)"].sum()
        bullets.append(f"<b>Budget Overrun:</b> Monthly COGS exceeds the <b>${budget_monthly:,.0f} cap</b> starting in <b>{df.loc[b_breach[0], 'Month']}</b> ({len(b_breach)} of {n} months over; total overrun in those months: <b>+${total_overrun:,.0f}</b>).")

if starting_ai_credits > 0 and ai_dep_idx and (not cloud_dep_idx or ai_dep_idx[0] <= cloud_dep_idx[0]):
    bullets.append(f"<b>Credit Depletion:</b> <b>AI Model Credits</b> reach zero in <b>{df.loc[ai_dep_idx[0], 'Month']}</b>. Direct cash payable for token inference begins thereafter.")
elif starting_cloud_credits > 0 and cloud_dep_idx:
    bullets.append(f"<b>Credit Depletion:</b> <b>Cloud Infrastructure Credits</b> reach zero in <b>{df.loc[cloud_dep_idx[0], 'Month']}</b>. Hosting spend hits cash directly thereafter.")
else:
    bullets.append(f"<b>Credit Depletion:</b> Configured credit balances remain active without depleting within the {n}-month forecast horizon.")

# Per-prompt API cost can only rise (token-growth inputs are >= 0), so only the expansion case is reported.
if prompt_api_cost_delta > 5.0:
    bullets.append(f"<b>API Inference Efficiency:</b> Pure API cost per 1,000 prompts expands by <b>+{prompt_api_cost_delta:.1f}%</b> from M1 to M{n}, driven by {token_growth_mode.lower()} token growth (input {input_growth*100:.1f}% / output {output_growth*100:.1f}% MoM).")

commentary_html = f"""
<div class="cfo-commentary">
    <h4>💡 Dynamic Executive Briefing & Decision Support</h4>
    <ul>
        {"".join([f"<li>{b}</li>" for b in bullets])}
    </ul>
</div>
"""
st.markdown(commentary_html, unsafe_allow_html=True)

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue & Margin",
    "💳 Credits & Cash",
    "🎛️ Scenario Matrix",
    "📋 Monthly Detail",
    "📑 Assumptions Register"
])

with tab1:
    st.subheader(f"{n}-Month Recognized Revenue, Modeled COGS & Gross Margin")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["Month"], y=[margin_target] * n, name=f"Target Margin ({margin_target}%)",
        mode="lines", line=dict(color="#a0aec0", width=1.5, dash="dot")
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", height=360,
        legend=dict(orientation="h", y=1.12, x=0.01),
        margin=dict(l=40, r=60, t=10, b=30)
    )
    fig.update_yaxes(title_text="USD ($ / Month)", secondary_y=False, rangemode="tozero", gridcolor="#edf2f7")

    if not valid_margins.empty:
        min_m, max_m = float(valid_margins.min()), float(valid_margins.max())
        y_min = min(-50.0, min_m - 10.0)
        y_max = max(100.0, max_m + 10.0)
        fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True, range=[y_min, y_max], gridcolor="#edf2f7")
    else:
        fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True, range=[-100, 100], gridcolor="#edf2f7")

    st.plotly_chart(fig, use_container_width=True, key="chart_revenue_margin")

    st.subheader("Inference Unit Cost Telemetry")
    col_u1, col_u2 = st.columns(2)

    fig_u1 = go.Figure()
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["API Cost / 1,000 Prompts ($)"], name="API-Only Cost / 1k", mode="lines+markers", line=dict(color="#805ad5", width=2.5)))
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["Ops COGS / 1,000 Prompts ($)"], name="Ops COGS / 1k (API + vector + infra)", mode="lines+markers", line=dict(color="#e53e3e", width=2, dash="dash")))
    fig_u1.update_layout(title="API Cost vs Ops COGS per 1,000 Prompts ($)", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.15))
    col_u1.plotly_chart(fig_u1, use_container_width=True, key="chart_unit_cost")

    fig_u2 = go.Figure()
    fig_u2.add_trace(go.Scatter(x=df["Month"], y=df["Blended COGS / Paid Sub ($)"], mode="lines+markers", line=dict(color="#319795", width=2.5)))
    fig_u2.update_layout(title="Blended COGS / Paid Subscriber ($)", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20))
    col_u2.plotly_chart(fig_u2, use_container_width=True, key="chart_cogs_per_sub")

with tab2:
    st.subheader("Cloud Credit Depletion vs. Out-of-Pocket Cash Payable")
    st.caption("Stacked bars represent total modeled COGS consumption offset by credit grants.")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["AI Credits Applied ($)"], name="AI Credits Applied ($)", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Cloud Credits Applied ($)"], name="Cloud Credits Applied ($)", marker_color="#4fd1c5"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Cash Payable ($)", marker_color="#dd6b20"), secondary_y=False)

    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color="#2b6cb0", width=2, dash="dash")), secondary_y=True)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Cloud Credits ($)"], name="Remaining Cloud Credits ($)", mode="lines", line=dict(color="#805ad5", width=2)), secondary_y=True)

    fig2.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", height=380,
        barmode="stack", legend=dict(orientation="h", y=1.12, x=0.01),
        margin=dict(l=40, r=60, t=10, b=30)
    )
    fig2.update_yaxes(title_text="Monthly COGS Coverage ($)", secondary_y=False, rangemode="tozero", gridcolor="#edf2f7")
    fig2.update_yaxes(title_text="Credit Balances ($)", secondary_y=True, rangemode="tozero", gridcolor="#edf2f7")
    st.plotly_chart(fig2, use_container_width=True, key="chart_credits_cash")

with tab3:
    st.subheader(f"Multi-Variable Scenario Matrix (Full {n}M Reforecast)")
    st.caption("Full 8-scenario matrix comparing revenue, COGS, gross margin, unit telemetry, and out-of-pocket cash payable against Base Case.")

    col_rev, col_cogs, col_gp = f"{n}M Revenue ($)", f"{n}M COGS ($)", f"{n}M Gross Profit ($)"
    col_cash, col_api, col_sub = f"{n}M Cash Payable ($)", f"M{n} API Cost / 1k ($)", f"M{n} COGS / Sub ($)"

    base_rev = df["Recognized Revenue ($)"].sum()
    base_cogs = df["Total Modeled COGS ($)"].sum()
    base_gp = base_rev - base_cogs
    base_gm = (base_gp / base_rev * 100) if base_rev > 0 else np.nan
    base_cash = df["Estimated Cash Payable ($)"].sum()

    scen_results = []
    for label, scen_assumptions in build_scenarios(assumptions):
        s_df = simulate(scen_assumptions)
        s_last = s_df.iloc[-1]

        s_rev = s_df["Recognized Revenue ($)"].sum()
        s_cogs = s_df["Total Modeled COGS ($)"].sum()
        s_gp = s_rev - s_cogs
        s_gm = (s_gp / s_rev * 100) if s_rev > 0 else np.nan
        s_cash = s_df["Estimated Cash Payable ($)"].sum()

        s_last_api_cost_1k = s_last["API Cost ($)"] / s_last["Monthly Prompts"] * 1000 if s_last["Monthly Prompts"] > 0 else np.nan
        s_last_cogs_sub = s_last["Total Modeled COGS ($)"] / s_last["Ending Paid Subscribers"] if s_last["Ending Paid Subscribers"] > 0 else np.nan

        scen_results.append({
            "Scenario": label,
            col_rev: s_rev,
            "Δ Revenue vs Base ($)": s_rev - base_rev,
            col_cogs: s_cogs,
            "Δ COGS vs Base ($)": s_cogs - base_cogs,
            col_gp: s_gp,
            "Gross Margin (%)": s_gm,
            "Δ Margin (pp)": (s_gm - base_gm) if (not pd.isna(s_gm) and not pd.isna(base_gm)) else np.nan,
            col_api: s_last_api_cost_1k,
            col_sub: s_last_cogs_sub,
            col_cash: s_cash,
            "Δ Cash Payable ($)": s_cash - base_cash,
        })

    scen_df = pd.DataFrame(scen_results)

    def _highlight_base(row):
        style = "font-weight: 700; background-color: #ebf8ff" if str(row["Scenario"]).startswith("1.") else ""
        return [style] * len(row)

    st.dataframe(
        scen_df.style.apply(_highlight_base, axis=1).format({
            col_rev: "${:,.0f}",
            "Δ Revenue vs Base ($)": "${:+,.0f}",
            col_cogs: "${:,.0f}",
            "Δ COGS vs Base ($)": "${:+,.0f}",
            col_gp: "${:,.0f}",
            "Gross Margin (%)": "{:.1f}%",
            "Δ Margin (pp)": "{:+.1f}pp",
            col_api: "${:,.4f}",
            col_sub: "${:,.2f}",
            col_cash: "${:,.0f}",
            "Δ Cash Payable ($)": "${:+,.0f}",
        }, na_rep="—"),
        use_container_width=True
    )

with tab4:
    st.subheader(f"{n}-Month Detailed Financial & Operational Roll-Forward")
    st.dataframe(
        df.style.format({
            "DAU": "{:,.0f}",
            "Gross New Users": "{:,.0f}",
            "Beginning Paid Subscribers": "{:,.0f}",
            "New User Conversions": "{:,.0f}",
            "Free Base Conversions": "{:,.0f}",
            "Churned Subscribers": "{:,.0f}",
            "Ending Paid Subscribers": "{:,.0f}",
            "Avg Active Paid Subs": "{:,.1f}",
            "Free Users": "{:,.0f}",
            "Monthly Prompts": "{:,.0f}",
            "Ending MRR Run-Rate ($)": "${:,.0f}",
            "Recognized Revenue ($)": "${:,.0f}",
            "API Cost ($)": "${:,.0f}",
            "Vector DB Cost ($)": "${:,.0f}",
            "Fixed Infrastructure ($)": "${:,.0f}",
            "Payment Processing ($)": "${:,.0f}",
            "Total Modeled COGS ($)": "${:,.0f}",
            "Modeled Gross Profit ($)": "${:,.0f}",
            "Modeled Gross Margin (%)": "{:.1f}%",
            "AI Credits Applied ($)": "${:,.0f}",
            "Remaining AI Credits ($)": "${:,.0f}",
            "Cloud Credits Applied ($)": "${:,.0f}",
            "Remaining Cloud Credits ($)": "${:,.0f}",
            "Total Credits Applied ($)": "${:,.0f}",
            "Estimated Cash Payable ($)": "${:,.0f}",
            "Budget ($)": "${:,.0f}",
            "Budget Variance ($)": "${:,.0f}",
            "Budget Overrun ($)": "${:,.0f}",
            "API Cost / 1,000 Prompts ($)": "${:,.4f}",
            "Ops COGS / 1,000 Prompts ($)": "${:,.4f}",
            "Blended COGS / Paid Sub ($)": "${:,.2f}",
        }, na_rep="—"),
        use_container_width=True
    )

    # Export only: round money columns to cents (the engine keeps full precision so identities reconcile exactly)
    csv_df = df.copy()
    money_cols = [c for c in csv_df.columns if "($)" in c]
    csv_df[money_cols] = csv_df[money_cols].round(2)
    csv = csv_df.to_csv(index=False).encode("utf-8")
    start_tag = f"{int(start_year)}-{int(start_month):02d}"
    st.download_button("Download Full Audit CSV", data=csv, file_name=f"ai_inference_cogs_{start_tag}_{n}m.csv", mime="text/csv")

with tab5:
    st.subheader("📑 Complete Source & Assumptions Audit Register")
    st.caption("Governance register detailing data provenance, verification status, and owner across UI inputs.")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    start_period_str = pd.Period(freq='M', year=int(start_year), month=int(start_month)).strftime("%b %Y")
    price_status = f"Source: {pricing_note.strip()}" if pricing_note.strip() else "Placeholder default — unverified (set the pricing source in the sidebar)"
    price_evidence = "Vendor public rate card (record model name + as-of date)"

    def reg(param, value, unit, source, status, owner, covers):
        return {"Parameter": param, "Current Value": value, "Unit / Scale": unit,
                "Primary Source / Evidence Required": source, "Verification Status": status,
                "Owner": owner, "Last Updated": today_str, "_fields": covers}

    assumptions_data = [
        reg("1. Forecast Start Date", start_period_str, "Calendar Period", "FP&A Planning Horizon Config", "User Estimate — defaults to next full month", "Finance", ["start_year", "start_month"]),
        reg("2. Starting DAU & Paid Subs", f"{starting_dau:,} DAU / {starting_paid_subs:,} Subs", "Active Accounts", "Mixpanel & Stripe Billing Baseline", "User Estimate — DAU includes paid subs; pending actuals upload", "Growth / Finance", ["starting_dau", "start_paid_subs"]),
        reg("3. Monthly DAU Growth (net)", f"{monthly_growth*100:.1f}%", "% MoM Growth", "Acquisition Model", "Management Estimate", "Marketing", ["growth"]),
        reg("4. New User Day-1 Conversion", f"{new_user_conversion*100:.1f}%", "% of gross new users", "Stripe Checkout Day-1 Upgrade Analytics", "Illustrative Baseline — Source Required (applies M2+)", "Growth", ["new_conv"]),
        reg("5. Free Base PLG Conversion", f"{free_base_conversion*100:.1f}%", "% Opening Free Pool / Mo", "In-App Product Funnel Telemetry", "Illustrative Baseline — Source Required", "Product", ["base_conv"]),
        reg("6. Paid Subscriber Churn", f"{monthly_churn*100:.1f}%", "% Monthly Churn", "Stripe Billing Dashboard", "User Estimate", "Finance", ["churn"]),
        reg("7. Monthly ARPU", f"${arpu:,.2f}", "$ / Paid Sub / Mo", "Subscription Plan Tier Card", "User Estimate — Public Tier Assumption", "Finance", ["arpu"]),
        reg("8. Daily Prompts (Paid / Free)", f"{baseline_prompts} / {baseline_prompts*free_user_usage_mult:.1f}", "Prompts / Day", "Helicone / Langfuse API Logging", "Telemetry Estimate", "Engineering", ["base_prompts", "free_mult"]),
        reg("9. Usage Surge Curve", f"Peak {peak_prompts} Prompts (M{peak_month} = {_peak_label}, σ={surge_width})", "Prompts / Day", "Product Event Capacity Model", "Management Estimate (M = forecast month index)", "Engineering", ["peak_prompts", "peak_month", "surge_width"]),
        reg("10. Token Context & Growth", f"{input_tokens_start:,} in / {output_tokens_start:,} out; {token_growth_mode.lower()} growth", "Tokens / Prompt", "Datadog APM / LLM Provider Logs", "Includes System + RAG Overhead", "Engineering", ["in_tok", "out_tok", "in_growth", "out_growth", "token_growth_mode"]),
        reg("11. Frontier Model Pricing & Share", f"${frontier_input_price:.2f} in / ${frontier_output_price:.2f} out; {frontier_mix*100:.0f}% of traffic", "$ / 1M Tokens", price_evidence, price_status, "DevOps", ["f_in_price", "f_out_price", "frontier_mix"]),
        reg("12. Standard Model Pricing", f"${standard_input_price:.4f} in / ${standard_output_price:.4f} out", "$ / 1M Tokens", price_evidence, price_status, "DevOps", ["s_in_price", "s_out_price"]),
        reg("13. Prompt Caching Rules", f"{cache_hit_rate*100:.0f}% hit / {cache_price_ratio*100:.0f}% cost", "% Hit / % Cost", "LLM Gateway Caching Logs", "Telemetry Estimate (check minimum cacheable prefix; assumed to stack with batch)", "Engineering", ["cache_rate", "cache_price_ratio"]),
        reg("14. Async Batch Routing", f"{batch_share*100:.0f}% traffic / {batch_price_ratio*100:.0f}% cost", "% Async Share", "Batch API Route Telemetry", "Management Estimate (non-interactive traffic only; assumed to stack with caching)", "Engineering", ["batch_share", "batch_price_ratio"]),
        reg("15. Base Fixed Infra Spend", f"${base_fixed_infra:,.0f}/mo", "$ / Month", "AWS / GCP Monthly Invoices", "Covers Tier 1 Base Capacity (Block 1)", "DevOps", ["base_infra"]),
        reg("16. Infra Step-Up Trigger", f"${step_cost_increment:,.0f} per {step_block_label}", "$ / Capacity Block", "DevOps Infrastructure Capacity Plan", "Triggers on end-of-month capacity beyond Block 1", "Engineering", ["step_type", "step_threshold", "step_cost"]),
        reg("17. Vector DB Unit Cost", f"${vector_cost_per_user:.2f} / avg DAU / mo", "$ / DAU / Month", "Pinecone / Qdrant Invoice Rate", "Simplified Storage + Query Driver (applied to average DAU)", "DevOps", ["vec_cost"]),
        reg("18. Starting Credit Balances", f"${starting_ai_credits:,.0f} AI / ${starting_cloud_credits:,.0f} Cloud (eligible {ai_credit_eligible_share*100:.0f}% / {cloud_credit_eligible_share*100:.0f}%)", "$ Total Grant", "AWS Activate & OpenAI Portal Grants", "User Estimate — Pending Grant Terms (expiry not modeled)", "Finance", ["ai_credits", "cloud_credits", "ai_elig", "cloud_elig"]),
        reg("19. Inference Overhead", f"{overhead_pct*100:.0f}%", "% uplift on API cost", "Gateway logs: retries, guardrails, tool loops, embeddings", "Management Estimate (default 0%)", "Engineering", ["overhead"]),
        reg("20. Payment Processing Fee", f"{payment_fee_pct*100:.1f}% of revenue", "% of Recognized Revenue", "Processor / App Store fee schedule", "Rate assumption (fixed per-charge fee not modeled)", "Finance", ["fee_pct"]),
        reg("21. Free-User Churn", f"{free_user_churn*100:.1f}% / mo", "% of free users", "Product Analytics (free-tier retention)", "Management Estimate (default 0%; affects M2+)", "Product", ["free_churn"]),
        reg("22. Forecast Horizon", f"{n} months", "Months", "FP&A Planning Horizon Config", "User Selection", "Finance", ["horizon"]),
    ]

    # Drift guard: every Assumptions field must be covered by a register row
    covered = {f for r in assumptions_data for f in r["_fields"]}
    missing_fields = [f.name for f in fields(Assumptions) if f.name not in covered]
    if missing_fields:
        st.warning(f"Assumptions Register is out of date - no row covers: {', '.join(missing_fields)}")

    register_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_fields"} for r in assumptions_data])
    st.table(register_df)

# MODEL SCOPE FOOTNOTE
st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
st.markdown("""
<div style="font-size:0.75rem; color:#a0aec0; line-height:1.3;">
<b>Model Financial Scope & Perimeter Disclaimer:</b> This tool models direct AI inference, vector storage, infrastructure, and payment-processing COGS (Direct Cost of Goods Sold). It evaluates <b>Cloud-Credit Depletion Horizons</b> and out-of-pocket cash payable for direct production expenses. It does NOT calculate total corporate cash burn or startup runway, which requires non-COGS operating expenses (payroll, marketing, sales CAC, G&A, legal), starting cash balances, cash receipts timing, and equity/debt financing activities.
</div>
""", unsafe_allow_html=True)
