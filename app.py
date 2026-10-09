"""
AI Inference, Credit Depletion & Unit Economics Model  (Streamlit UI)
---------------------------------------------------------------------
Driver-based FP&A planning model for evaluating direct AI model inference,
vector storage, and infrastructure COGS. Excludes non-COGS OpEx (payroll, marketing, G&A).
Prices and assumptions are user-editable estimates; verify against vendor billing portals.

Single-file app: the forecast engine (Assumptions dataclass + pure functions) sits at the top,
followed by the Streamlit UI.
Run with:  streamlit run app.py

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
import math
from dataclasses import asdict, dataclass, replace
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
    peak_month: int = 7             # forecast month index (M1..M12)
    surge_width: float = 2.0
    free_mult: float = 0.5
    in_tok: float = 1000.0
    out_tok: float = 400.0
    in_growth: float = 0.03
    out_growth: float = 0.0
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
    dates = [pd.Period(freq="M", year=a.start_year, month=a.start_month) + i for i in range(12)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    m_idx = np.arange(1, 13)

    width = max(float(a.surge_width), 1e-6)
    prompt_curve = a.base_prompts + (a.peak_prompts - a.base_prompts) * np.exp(
        -((m_idx - a.peak_month) ** 2) / (2 * width ** 2)
    )

    ai_credits_left = float(a.ai_credits)
    cloud_credits_left = float(a.cloud_credits)

    current_dau = float(a.starting_dau)
    prev_end_subs = float(a.start_paid_subs)  # unrounded carry-forward

    rows = []
    for i in range(12):
        beg_paid_subs = prev_end_subs

        if i == 0:
            dau_val = current_dau
            # Starting DAU is the existing base (already reflected in starting paid subs):
            # no brand-new cohort in Month 1.
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
        vector_cost = dau_val * a.vec_cost

        if a.step_type == "DAU Threshold":
            trigger = dau_val
        elif a.step_type == "Paid Subscribers Threshold":
            trigger = ending_paid_subs
        else:
            trigger = prompts_total

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

        # Dual-pool credit depletion (payment fees are not credit-eligible)
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


@st.cache_data(show_spinner=False)
def _cached_run(params: dict) -> pd.DataFrame:
    return run_model_simulation(Assumptions(**params))


def simulate(a: Assumptions) -> pd.DataFrame:
    return _cached_run(asdict(a)).copy()


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
st.sidebar.header("0. Rolling Forecast Horizon")
d_year, d_month = default_start(datetime.date.today())
start_year = st.sidebar.number_input(
    "Forecast Start Year", min_value=2024, max_value=2035, value=min(max(d_year, 2024), 2035),
    help="Defaults to the next full calendar month so the first forecast month is not partly elapsed."
)
start_month = st.sidebar.slider("Forecast Start Month", 1, 12, d_month)

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
    "Peak Usage Forecast Month (1-12)", 1, 12, 7,
    help="Forecast month index (M1-M12), not calendar month. M1 = the forecast start month."
)
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

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider("Frontier Model Traffic Share (%)", 0, 100, 30, help="Simplified routing assumption for premium models.") / 100

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
    ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"],
    help="Select basis for triggering modular infrastructure upgrades beyond base capacity."
)

if step_trigger_type == "DAU Threshold":
    step_threshold_input = st.sidebar.number_input("DAU Capacity Block (Users)", min_value=1000, value=20000, step=5000)
    raw_step_threshold = float(step_threshold_input)
elif step_trigger_type == "Paid Subscribers Threshold":
    step_threshold_input = st.sidebar.number_input("Subscriber Capacity Block (Subs)", min_value=500, value=5000, step=1000)
    raw_step_threshold = float(step_threshold_input)
else:
    step_threshold_input = st.sidebar.number_input("Monthly Prompt Block (Millions)", min_value=1, value=20, step=5)
    raw_step_threshold = float(step_threshold_input * 1_000_000)

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
    start_year=int(start_year), start_month=int(start_month),
    starting_dau=float(starting_dau), start_paid_subs=float(starting_paid_subs),
    growth=monthly_growth, new_conv=new_user_conversion, base_conv=free_base_conversion,
    churn=monthly_churn, arpu=float(arpu), free_churn=free_user_churn,
    base_prompts=float(baseline_prompts), peak_prompts=float(peak_prompts), peak_month=int(peak_month),
    surge_width=float(surge_width), free_mult=free_user_usage_mult,
    in_tok=float(input_tokens_start), out_tok=float(output_tokens_start),
    in_growth=input_growth, out_growth=output_growth,
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

# Run Baseline Simulation
df = add_kpis(simulate(assumptions), budget_monthly)

# ==========================================
# 3. CFO DASHBOARD KPI PANEL & DECISION ENGINE
# ==========================================
annual_revenue = df["Recognized Revenue ($)"].sum()
annual_cogs = df["Total Modeled COGS ($)"].sum()
m12 = df.iloc[-1]
m12_arr = m12['Ending MRR Run-Rate ($)'] * 12

ai_dep_idx = df.index[df["Remaining AI Credits ($)"] <= 0.005].tolist() if starting_ai_credits > 0 else []
cloud_dep_idx = df.index[df["Remaining Cloud Credits ($)"] <= 0.005].tolist() if starting_cloud_credits > 0 else []

ai_dep_str = f"Depleted in {df.loc[ai_dep_idx[0], 'Month']}" if ai_dep_idx else ("None ($0)" if starting_ai_credits <= 0 else "Active through M12")
cloud_dep_str = f"Depleted in {df.loc[cloud_dep_idx[0], 'Month']}" if cloud_dep_idx else ("None ($0)" if starting_cloud_credits <= 0 else "Active through M12")

m12_margin = m12["Modeled Gross Margin (%)"]
if pd.isna(m12_margin):
    margin_kpi_str = "Pre-Revenue"
    margin_sub_str = "Gross Loss ($0 Rev)"
else:
    margin_var = m12_margin - margin_target
    margin_kpi_str = f"{m12_margin:.1f}%"
    margin_sub_str = f"{margin_var:+.1f}% vs {margin_target}% Target"

c_m1_api = df.loc[0, "API Cost / 1,000 Prompts ($)"]
c_m12_api = df.loc[11, "API Cost / 1,000 Prompts ($)"]
if pd.notna(c_m1_api) and pd.notna(c_m12_api) and c_m1_api > 0:
    prompt_api_cost_delta = (c_m12_api - c_m1_api) / c_m1_api * 100
else:
    prompt_api_cost_delta = 0.0
api_kpi_str = f"${c_m12_api:.4f} / 1k" if pd.notna(c_m12_api) else "n/a (no prompts)"

c1, c2, c3, c4 = st.columns(4)
c1.metric("1. ARR Run-Rate (M12)", f"${m12_arr:,.0f} ARR", f"Recognized 12M Rev: ${annual_revenue:,.0f}")
c2.metric("2. Gross Margin (M12)", margin_kpi_str, margin_sub_str)
c3.metric("3. Pure API Efficiency", api_kpi_str, f"{prompt_api_cost_delta:+.1f}% M1 to M12")
c4.metric("4. Credit Depletion Horizon", f"AI: {ai_dep_str}", f"Cloud: {cloud_dep_str}")

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
        bullets.append(f"<b>Margin Compliant:</b> Gross margin remains above the <b>{margin_target}% target</b> across all 12 forecast months.")

# Budget Breach Commentary (overrun months only)
if budget_monthly > 0:
    b_breach = df.index[df["Budget Overrun ($)"] > 0].tolist()
    if b_breach:
        total_overrun = df["Budget Overrun ($)"].sum()
        bullets.append(f"<b>Budget Overrun:</b> Monthly COGS exceeds the <b>${budget_monthly:,.0f} cap</b> starting in <b>{df.loc[b_breach[0], 'Month']}</b> ({len(b_breach)} of 12 months over; total overrun in those months: <b>+${total_overrun:,.0f}</b>).")

if starting_ai_credits > 0 and ai_dep_idx and (not cloud_dep_idx or ai_dep_idx[0] <= cloud_dep_idx[0]):
    bullets.append(f"<b>Credit Depletion:</b> <b>AI Model Credits</b> reach zero in <b>{df.loc[ai_dep_idx[0], 'Month']}</b>. Direct cash payable for token inference begins thereafter.")
elif starting_cloud_credits > 0 and cloud_dep_idx:
    bullets.append(f"<b>Credit Depletion:</b> <b>Cloud Infrastructure Credits</b> reach zero in <b>{df.loc[cloud_dep_idx[0], 'Month']}</b>. Hosting spend hits cash directly thereafter.")
else:
    bullets.append("<b>Credit Depletion:</b> Configured credit balances remain active without depleting within the 12-month forecast horizon.")

# Per-prompt API cost can only rise (token-growth inputs are >= 0), so only the expansion case is reported.
if prompt_api_cost_delta > 5.0:
    bullets.append(f"<b>API Inference Efficiency:</b> Pure API cost per 1,000 prompts expands by <b>+{prompt_api_cost_delta:.1f}%</b> from M1 to M12, driven by compound token growth (input {input_growth*100:.1f}% / output {output_growth*100:.1f}% MoM).")

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
    st.subheader("12-Month Recognized Revenue, Modeled COGS & Gross Margin")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["Month"], y=[margin_target]*12, name=f"Target Margin ({margin_target}%)",
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

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inference Unit Cost Telemetry")
    col_u1, col_u2 = st.columns(2)

    fig_u1 = go.Figure()
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["API Cost / 1,000 Prompts ($)"], name="API-Only Cost / 1k", mode="lines+markers", line=dict(color="#805ad5", width=2.5)))
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["Ops COGS / 1,000 Prompts ($)"], name="Ops COGS / 1k (API + vector + infra)", mode="lines+markers", line=dict(color="#e53e3e", width=2, dash="dash")))
    fig_u1.update_layout(title="API Cost vs Ops COGS per 1,000 Prompts ($)", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.15))
    col_u1.plotly_chart(fig_u1, use_container_width=True)

    fig_u2 = go.Figure()
    fig_u2.add_trace(go.Scatter(x=df["Month"], y=df["Blended COGS / Paid Sub ($)"], mode="lines+markers", line=dict(color="#319795", width=2.5)))
    fig_u2.update_layout(title="Blended COGS / Paid Subscriber ($)", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20))
    col_u2.plotly_chart(fig_u2, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Multi-Variable Scenario Matrix (Full 12M Reforecast)")
    st.caption("Full 8-scenario matrix comparing revenue, COGS, gross margin, unit telemetry, and out-of-pocket cash payable against Base Case.")

    base_rev = df["Recognized Revenue ($)"].sum()
    base_cogs = df["Total Modeled COGS ($)"].sum()
    base_gp = base_rev - base_cogs
    base_gm = (base_gp / base_rev * 100) if base_rev > 0 else np.nan
    base_cash = df["Estimated Cash Payable ($)"].sum()

    scen_results = []
    for label, scen_assumptions in build_scenarios(assumptions):
        s_df = simulate(scen_assumptions)

        s_rev = s_df["Recognized Revenue ($)"].sum()
        s_cogs = s_df["Total Modeled COGS ($)"].sum()
        s_gp = s_rev - s_cogs
        s_gm = (s_gp / s_rev * 100) if s_rev > 0 else np.nan
        s_cash = s_df["Estimated Cash Payable ($)"].sum()

        s_m12_api_cost_1k = s_df.loc[11, "API Cost ($)"] / s_df.loc[11, "Monthly Prompts"] * 1000 if s_df.loc[11, "Monthly Prompts"] > 0 else np.nan
        s_m12_cogs_sub = s_df.loc[11, "Total Modeled COGS ($)"] / s_df.loc[11, "Ending Paid Subscribers"] if s_df.loc[11, "Ending Paid Subscribers"] > 0 else np.nan

        scen_results.append({
            "Scenario": label,
            "12M Revenue ($)": s_rev,
            "Δ Revenue vs Base ($)": s_rev - base_rev,
            "12M COGS ($)": s_cogs,
            "Δ COGS vs Base ($)": s_cogs - base_cogs,
            "12M Gross Profit ($)": s_gp,
            "Gross Margin (%)": s_gm,
            "Δ Margin (pp)": (s_gm - base_gm) if (not pd.isna(s_gm) and not pd.isna(base_gm)) else np.nan,
            "M12 API Cost / 1k ($)": s_m12_api_cost_1k,
            "M12 COGS / Sub ($)": s_m12_cogs_sub,
            "12M Cash Payable ($)": s_cash,
            "Δ Cash Payable ($)": s_cash - base_cash,
        })

    scen_df = pd.DataFrame(scen_results)
    st.dataframe(
        scen_df.style.format({
            "12M Revenue ($)": "${:,.0f}",
            "Δ Revenue vs Base ($)": "${:+,.0f}",
            "12M COGS ($)": "${:,.0f}",
            "Δ COGS vs Base ($)": "${:+,.0f}",
            "12M Gross Profit ($)": "${:,.0f}",
            "Gross Margin (%)": "{:.1f}%",
            "Δ Margin (pp)": "{:+.1f}pp",
            "M12 API Cost / 1k ($)": "${:,.4f}",
            "M12 COGS / Sub ($)": "${:,.2f}",
            "12M Cash Payable ($)": "${:,.0f}",
            "Δ Cash Payable ($)": "${:+,.0f}",
        }, na_rep="—"),
        use_container_width=True
    )

with tab4:
    st.subheader("12-Month Detailed Financial & Operational Roll-Forward")
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

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Audit CSV", data=csv, file_name="ai_inference_cogs_model.csv", mime="text/csv")

with tab5:
    st.subheader("📑 Complete Source & Assumptions Audit Register")
    st.caption("Governance register detailing data provenance, verification status, and owner across UI inputs.")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    start_period_str = pd.Period(freq='M', year=int(start_year), month=int(start_month)).strftime("%b %Y")

    assumptions_data = [
        {"Parameter": "1. Forecast Start Date", "Current Value": f"{start_period_str}", "Unit / Scale": "Calendar Period", "Primary Source / Evidence Required": "FP&A Planning Horizon Config", "Verification Status": "User Estimate — defaults to next full month", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "2. Starting DAU & Paid Subs", "Current Value": f"{starting_dau:,} DAU / {starting_paid_subs:,} Subs", "Unit / Scale": "Active Accounts", "Primary Source / Evidence Required": "Mixpanel & Stripe Billing Baseline", "Verification Status": "User Estimate — DAU includes paid subs; pending actuals upload", "Owner": "Growth / Finance", "Last Updated": today_str},
        {"Parameter": "3. Monthly DAU Growth (net)", "Current Value": f"{monthly_growth*100:.1f}%", "Unit / Scale": "% MoM Growth", "Primary Source / Evidence Required": "Acquisition Model", "Verification Status": "Management Estimate", "Owner": "Marketing", "Last Updated": today_str},
        {"Parameter": "4. New User Day-1 Conversion", "Current Value": f"{new_user_conversion*100:.1f}%", "Unit / Scale": "% of gross new users", "Primary Source / Evidence Required": "Stripe Checkout Day-1 Upgrade Analytics", "Verification Status": "Illustrative Baseline — Source Required (applies M2+)", "Owner": "Growth", "Last Updated": today_str},
        {"Parameter": "5. Free Base PLG Conversion", "Current Value": f"{free_base_conversion*100:.1f}%", "Unit / Scale": "% Opening Free Pool / Mo", "Primary Source / Evidence Required": "In-App Product Funnel Telemetry", "Verification Status": "Illustrative Baseline — Source Required", "Owner": "Product", "Last Updated": today_str},
        {"Parameter": "6. Paid Subscriber Churn", "Current Value": f"{monthly_churn*100:.1f}%", "Unit / Scale": "% Monthly Churn", "Primary Source / Evidence Required": "Stripe Billing Dashboard", "Verification Status": "User Estimate", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "7. Monthly ARPU", "Current Value": f"${arpu:,.2f}", "Unit / Scale": "$ / Paid Sub / Mo", "Primary Source / Evidence Required": "Subscription Plan Tier Card", "Verification Status": "User Estimate — Public Tier Assumption", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "8. Daily Prompts (Paid / Free)", "Current Value": f"{baseline_prompts} / {baseline_prompts*free_user_usage_mult:.1f}", "Unit / Scale": "Prompts / Day", "Primary Source / Evidence Required": "Helicone / Langfuse API Logging", "Verification Status": "Telemetry Estimate", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "9. Usage Surge Curve", "Current Value": f"Peak {peak_prompts} Prompts (M{peak_month}, σ={surge_width})", "Unit / Scale": "Prompts / Day", "Primary Source / Evidence Required": "Product Event Capacity Model", "Verification Status": "Management Estimate (M = forecast month index)", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "10. Token Context & Growth", "Current Value": f"{input_tokens_start:,} in / {output_tokens_start:,} out", "Unit / Scale": "Tokens / Prompt", "Primary Source / Evidence Required": "Datadog APM / LLM Provider Logs", "Verification Status": "Includes System + RAG Overhead", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "11. Frontier Model Pricing", "Current Value": f"${frontier_input_price:.2f} in / ${frontier_output_price:.2f} out", "Unit / Scale": "$ / 1M Tokens", "Primary Source / Evidence Required": "Vendor public rate card (record model name + as-of date)", "Verification Status": "Placeholder default — unverified", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "12. Standard Model Pricing", "Current Value": f"${standard_input_price:.4f} in / ${standard_output_price:.4f} out", "Unit / Scale": "$ / 1M Tokens", "Primary Source / Evidence Required": "Vendor public rate card (record model name + as-of date)", "Verification Status": "Placeholder default — unverified", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "13. Prompt Caching Rules", "Current Value": f"{cache_hit_rate*100:.0f}% hit / {cache_price_ratio*100:.0f}% cost", "Unit / Scale": "% Hit / % Cost", "Primary Source / Evidence Required": "LLM Gateway Caching Logs", "Verification Status": "Telemetry Estimate (check minimum cacheable prefix)", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "14. Async Batch Routing", "Current Value": f"{batch_share*100:.0f}% traffic / {batch_price_ratio*100:.0f}% cost", "Unit / Scale": "% Async Share", "Primary Source / Evidence Required": "Batch API Route Telemetry", "Verification Status": "Management Estimate (non-interactive traffic only)", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "15. Base Fixed Infra Spend", "Current Value": f"${base_fixed_infra:,.0f}/mo", "Unit / Scale": "$ / Month", "Primary Source / Evidence Required": "AWS / GCP Monthly Invoices", "Verification Status": "Covers Tier 1 Base Capacity (Block 1)", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "16. Infra Step-Up Trigger", "Current Value": f"${step_cost_increment:,.0f} per {step_threshold_input:,.0f} {step_trigger_type}", "Unit / Scale": "$ / Capacity Block", "Primary Source / Evidence Required": "DevOps Infrastructure Capacity Plan", "Verification Status": "Triggers on excess capacity beyond Block 1", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "17. Vector DB Unit Cost", "Current Value": f"${vector_cost_per_user:.2f} / DAU / mo", "Unit / Scale": "$ / DAU / Month", "Primary Source / Evidence Required": "Pinecone / Qdrant Invoice Rate", "Verification Status": "Simplified Storage + Query Driver", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "18. Starting Credit Balances", "Current Value": f"${starting_ai_credits:,.0f} AI / ${starting_cloud_credits:,.0f} Cloud", "Unit / Scale": "$ Total Grant", "Primary Source / Evidence Required": "AWS Activate & OpenAI Portal Grants", "Verification Status": "User Estimate — Pending Grant Terms (expiry not modeled)", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "19. Inference Overhead", "Current Value": f"{overhead_pct*100:.0f}%", "Unit / Scale": "% uplift on API cost", "Primary Source / Evidence Required": "Gateway logs: retries, guardrails, tool loops, embeddings", "Verification Status": "Management Estimate (default 0%)", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "20. Payment Processing Fee", "Current Value": f"{payment_fee_pct*100:.1f}% of revenue", "Unit / Scale": "% of Recognized Revenue", "Primary Source / Evidence Required": "Processor / App Store fee schedule", "Verification Status": "Rate assumption (fixed per-charge fee not modeled)", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "21. Free-User Churn", "Current Value": f"{free_user_churn*100:.1f}% / mo", "Unit / Scale": "% of free users", "Primary Source / Evidence Required": "Product Analytics (free-tier retention)", "Verification Status": "Management Estimate (default 0%)", "Owner": "Product", "Last Updated": today_str},
    ]

    st.table(pd.DataFrame(assumptions_data))

# MODEL SCOPE FOOTNOTE
st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
st.markdown("""
<div style="font-size:0.75rem; color:#a0aec0; line-height:1.3;">
<b>Model Financial Scope & Perimeter Disclaimer:</b> This tool models direct AI inference, vector storage, infrastructure, and payment-processing COGS (Direct Cost of Goods Sold). It evaluates <b>Cloud-Credit Depletion Horizons</b> and out-of-pocket cash payable for direct production expenses. It does NOT calculate total corporate cash burn or startup runway, which requires non-COGS operating expenses (payroll, marketing, sales CAC, G&A, legal), starting cash balances, cash receipts timing, and equity/debt financing activities.
</div>
""", unsafe_allow_html=True)
