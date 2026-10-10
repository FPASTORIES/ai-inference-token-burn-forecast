"""
AI Inference, Credit Depletion & Unit Economics Model
------------------------------------------------------
Driver-based FP&A planning model for evaluating direct AI model inference,
vector storage, and infrastructure COGS. Excludes non-COGS OpEx (payroll, marketing, G&A).
Prices and assumptions are user-editable estimates; verify against vendor billing portals.
"""

import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference, Credit Depletion & Unit Economics Model",
    page_icon="⚡",
    layout="wide"
)

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

# Compact Header Banner
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
current_now = datetime.datetime.now()
start_year = st.sidebar.number_input("Forecast Start Year", min_value=2024, max_value=2035, value=current_now.year)
start_month = st.sidebar.slider("Forecast Start Month", 1, 12, current_now.month)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 36, 12)

st.sidebar.header("1. User Acquisition & Paid Funnel")
starting_dau = st.sidebar.number_input(
    "Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500,
    help="Initial active user count at Month 1."
)
starting_paid_subs = st.sidebar.number_input(
    "Starting Paid Subscribers", min_value=0, value=250, step=50,
    help="Initial paid subscriber count. Assumed to be a subset of starting DAU."
)
monthly_growth = st.sidebar.slider(
    "Monthly DAU Growth (%)", 0.0, 25.0, 8.0, 0.5,
    help="Assumed MoM growth for top-of-funnel acquisition (modeled as a simplified net DAU growth proxy)."
) / 100

paid_dau_factor = st.sidebar.slider(
    "Paid Subscribers Active Daily (%)", 0, 100, 70, 5,
    help="Average daily active paid users as a share of paid subscribers."
) / 100

new_user_conversion = st.sidebar.slider(
    "New User Conversion (%)", 0.0, 50.0, 15.0, 0.5,
    help="% of brand-new acquired users who upgrade to paid in their first month (applied to net DAU growth proxy)."
) / 100

free_base_conversion = st.sidebar.slider(
    "Free-User Proxy Conversion (%)", 0.0, 10.0, 2.0, 0.1,
    help="% of active free-user proxy pool converting to paid each month."
) / 100

monthly_churn = st.sidebar.slider(
    "Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5,
    help="Percentage of paid subscribers cancelling each month."
) / 100

arpu = st.sidebar.number_input(
    "Monthly ARPU per Paid Subscriber ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average Revenue Per User per month for paid tiers."
)

expansion_arpu_pct = st.sidebar.slider(
    "Expansion / Overage Revenue (% of ARPU)", 0.0, 50.0, 10.0, 1.0,
    help="Incremental revenue from token overages and credit top-ups as a % of base ARPU."
) / 100

payment_fee_pct = st.sidebar.slider(
    "Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1,
    help="Card processor / app-store fees recognized in COGS (modeling assumption)."
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

if peak_prompts_input < baseline_prompts:
    st.sidebar.warning(f"⚠️ Peak prompts ({peak_prompts_input}) cannot be below baseline ({baseline_prompts}). Enforcing peak = {baseline_prompts}.")
    peak_prompts = baseline_prompts
else:
    peak_prompts = peak_prompts_input

peak_month = st.sidebar.slider("Peak Usage Forecast Month", 1, forecast_horizon, min(7, forecast_horizon))
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
    help="Average billed completion tokens per prompt."
)
input_growth = st.sidebar.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
output_growth = st.sidebar.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider("Frontier Model Traffic Share (%)", 0, 100, 30) / 100
standard_mix = 1.0 - frontier_mix

frontier_input_price = st.sidebar.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25)
frontier_output_price = st.sidebar.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50)
standard_input_price = st.sidebar.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
standard_output_price = st.sidebar.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f")

cache_hit_rate = st.sidebar.slider("Eligible Input Tokens Cached (%)", 0, 90, 40) / 100
cache_price_ratio = st.sidebar.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
batch_share = st.sidebar.slider("Traffic Using Batch Pricing (%)", 0, 80, 20) / 100
batch_price_ratio = st.sidebar.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100

overhead_pct = st.sidebar.slider("Inference Overhead (%)", 0, 100, 0) / 100

st.sidebar.header("4. Infrastructure Step-Costs & Credits")
base_fixed_infra = st.sidebar.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=2500.0, step=500.0)

step_trigger_type = st.sidebar.selectbox(
    "Step-Cost Trigger Basis",
    ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"]
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

vector_cost_per_user = st.sidebar.number_input("Vector DB Cost / DAU / Month ($)", min_value=0.0, value=0.15, step=0.05)

starting_ai_credits = st.sidebar.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=20000.0, step=2500.0)
starting_cloud_credits = st.sidebar.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=30000.0, step=5000.0)
ai_credit_eligible_share = st.sidebar.slider("AI Credit Eligible Share (%)", 0, 100, 100) / 100
cloud_credit_eligible_share = st.sidebar.slider("Cloud Credit Eligible Share (%)", 0, 100, 100) / 100

margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

# ==========================================
# 2. ENGINE-LEVEL VALIDATION & FORECAST FUNCTION
# ==========================================
def validate_assumptions(
    p_start_month, p_horizon, p_peak_m, p_dau, p_start_paid_subs, p_growth, p_churn,
    p_paid_dau_factor, p_new_conv, p_base_conv, p_arpu, p_front_mix, p_cache_rate,
    p_batch_share, p_ai_elig, p_cloud_elig, p_overhead, p_fee_pct,
    p_in_tok, p_out_tok, p_base_infra, p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred,
    p_cache_ratio, p_batch_price_ratio, p_step_thresh_raw, p_expansion_pct
):
    """Reject invalid inputs before forecasting."""
    if not 1 <= p_start_month <= 12:
        raise ValueError("start_month must be between 1 and 12.")
    if not 1 <= p_horizon <= 36:
        raise ValueError("horizon must be between 1 and 36 months.")
    if not 1 <= p_peak_m <= p_horizon:
        raise ValueError("peak_month must fall within the forecast horizon.")
    if p_dau < 0 or p_start_paid_subs < 0:
        raise ValueError("Starting DAU and paid subscribers cannot be negative.")
    if p_start_paid_subs > p_dau:
        raise ValueError("Starting paid subscribers cannot exceed starting DAU.")
    if p_growth < 0 or p_churn < 0:
        raise ValueError("Growth and churn rates cannot be negative.")
    if not (0 <= p_paid_dau_factor <= 1 and 0 <= p_new_conv <= 1 and 0 <= p_base_conv <= 1 and 0 <= p_churn <= 1):
        raise ValueError("Rates, conversion factors, and churn must be between 0% and 100%.")
    if not (0 <= p_front_mix <= 1 and 0 <= p_cache_rate <= 1 and 0 <= p_batch_share <= 1):
        raise ValueError("Mix, caching, and batch share ratios must be between 0% and 100%.")
    if not (0 <= p_cache_ratio <= 1 and 0 <= p_batch_price_ratio <= 1):
        raise ValueError("Cache price ratio and batch price ratio must be between 0% and 100%.")
    if not (0 <= p_ai_elig <= 1 and 0 <= p_cloud_elig <= 1 and 0 <= p_overhead <= 1 and 0 <= p_fee_pct <= 1 and 0 <= p_expansion_pct <= 1):
        raise ValueError("Credit eligibility, overhead, fee, and expansion percentages must be between 0% and 100%.")
    if p_arpu < 0 or p_in_tok < 0 or p_out_tok < 0 or p_base_infra < 0 or p_step_cost < 0 or p_vec_cost < 0 or p_ai_cred < 0 or p_cloud_cred < 0 or p_step_thresh_raw < 0:
        raise ValueError("Token counts, prices, costs, ARPU, credits, and thresholds cannot be negative.")

@st.cache_data(show_spinner=False)
def run_model_simulation(
    p_start_year, p_start_month, p_horizon, p_dau, p_start_paid_subs, p_growth, p_paid_dau_factor,
    p_new_conv, p_base_conv, p_churn, p_arpu, p_expansion_pct,
    p_base_prompts, p_peak_prompts, p_peak_m, p_surge_w, p_free_mult,
    p_in_tok, p_out_tok, p_in_grow, p_out_grow, p_front_mix,
    p_f_in_p, p_f_out_p, p_s_in_p, p_s_out_p, p_cache_rate, p_cache_ratio,
    p_batch_share, p_batch_price_ratio, p_base_infra, p_step_type, p_step_thresh_raw,
    p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred, p_ai_elig, p_cloud_elig,
    p_overhead, p_fee_pct
):
    validate_assumptions(
        p_start_month, p_horizon, p_peak_m, p_dau, p_start_paid_subs, p_growth, p_churn,
        p_paid_dau_factor, p_new_conv, p_base_conv, p_arpu, p_front_mix, p_cache_rate,
        p_batch_share, p_ai_elig, p_cloud_elig, p_overhead, p_fee_pct,
        p_in_tok, p_out_tok, p_base_infra, p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred,
        p_cache_ratio, p_batch_price_ratio, p_step_thresh_raw, p_expansion_pct
    )
    n = int(p_horizon)

    dates = [pd.Period(freq='M', year=int(p_start_year), month=int(p_start_month)) + i for i in range(n)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    m_indices = np.arange(1, n + 1)

    prompt_curve = p_base_prompts + (p_peak_prompts - p_base_prompts) * np.exp(-((m_indices - p_peak_m) ** 2) / (2 * p_surge_w ** 2))

    ai_credits_left = float(p_ai_cred)
    cloud_credits_left = float(p_cloud_cred)

    current_dau = float(p_dau)
    prev_end_subs = float(p_start_paid_subs)

    sim_rows = []

    for i in range(n):
        beg_paid_subs = prev_end_subs

        if i == 0:
            dau_val = current_dau
            new_users_acquired = 0.0
            free_base = max(0.0, dau_val - beg_paid_subs)
            prev_dau_val = current_dau
        else:
            prev_dau_val = current_dau
            current_dau = prev_dau_val * (1 + p_growth)
            dau_val = current_dau
            new_users_acquired = dau_val - prev_dau_val
            free_base = max(0.0, prev_dau_val - beg_paid_subs)

        new_user_conv_subs = new_users_acquired * p_new_conv
        free_base_conv_subs = free_base * p_base_conv
        total_new_conversions = new_user_conv_subs + free_base_conv_subs
        churned_subs = beg_paid_subs * p_churn
        
        raw_ending_subs = beg_paid_subs + total_new_conversions - churned_subs
        if raw_ending_subs > dau_val + 1e-6:
            raise ValueError(
                f"Paid subscriber forecast ({raw_ending_subs:,.0f}) "
                f"exceeds DAU ({dau_val:,.0f}) in month {month_names[i]}. "
                "Review conversion assumptions and user-population definitions."
            )
        ending_paid_subs = max(0.0, raw_ending_subs)
        prev_end_subs = ending_paid_subs

        avg_active_paid_subs = (beg_paid_subs + ending_paid_subs) / 2.0
        avg_dau = (prev_dau_val + dau_val) / 2.0
        
        paid_dau = avg_active_paid_subs * p_paid_dau_factor
        free_dau_avg = max(0.0, avg_dau - paid_dau)
        free_users_snapshot = max(0.0, dau_val - ending_paid_subs)

        input_tokens_val = p_in_tok * ((1 + p_in_grow) ** i)
        output_tokens_val = p_out_tok * ((1 + p_out_grow) ** i)

        prompts_per_paid_user_day = float(prompt_curve[i])
        prompts_per_free_user_day = prompts_per_paid_user_day * p_free_mult

        monthly_prompts_paid = paid_dau * prompts_per_paid_user_day * month_days[i]
        monthly_prompts_free = free_dau_avg * prompts_per_free_user_day * month_days[i]
        monthly_prompts_total = monthly_prompts_paid + monthly_prompts_free

        raw_input_m = monthly_prompts_total * input_tokens_val / 1_000_000
        raw_output_m = monthly_prompts_total * output_tokens_val / 1_000_000

        uncached_input_m = raw_input_m * (1 - p_cache_rate)
        cached_input_m = raw_input_m * p_cache_rate

        std_mix = 1.0 - p_front_mix
        frontier_input_cost = (uncached_input_m * p_f_in_p + cached_input_m * p_f_in_p * p_cache_ratio) * p_front_mix
        standard_input_cost = (uncached_input_m * p_s_in_p + cached_input_m * p_s_in_p * p_cache_ratio) * std_mix
        frontier_output_cost = raw_output_m * p_f_out_p * p_front_mix
        standard_output_cost = raw_output_m * p_s_out_p * std_mix

        api_before_batch = frontier_input_cost + standard_input_cost + frontier_output_cost + standard_output_cost
        api_cost = api_before_batch * ((1 - p_batch_share) + p_batch_share * p_batch_price_ratio) * (1 + p_overhead)

        vector_cost = avg_dau * p_vec_cost

        if p_step_type == "DAU Threshold":
            trigger_metric = dau_val
        elif p_step_type == "Paid Subscribers Threshold":
            trigger_metric = ending_paid_subs
        else:
            trigger_metric = monthly_prompts_total

        if p_step_thresh_raw > 0 and trigger_metric > p_step_thresh_raw:
            step_multiplier = max(0, math.ceil(trigger_metric / p_step_thresh_raw) - 1)
        else:
            step_multiplier = 0

        infrastructure_cost = p_base_infra + (step_multiplier * p_step_cost)

        base_revenue = avg_active_paid_subs * p_arpu
        expansion_revenue = base_revenue * p_expansion_pct
        recognized_revenue = base_revenue + expansion_revenue
        ending_mrr_runrate = ending_paid_subs * p_arpu * (1 + p_expansion_pct)
        payment_fees = recognized_revenue * p_fee_pct

        total_modeled_cogs = api_cost + vector_cost + infrastructure_cost + payment_fees

        gross_profit = recognized_revenue - total_modeled_cogs
        gross_margin_pct = (gross_profit / recognized_revenue * 100) if recognized_revenue > 0 else np.nan

        eligible_api_cost = api_cost * p_ai_elig
        ai_credits_applied = min(ai_credits_left, eligible_api_cost)
        ai_credits_left = max(0.0, ai_credits_left - ai_credits_applied)

        eligible_cloud_cost = (vector_cost + infrastructure_cost) * p_cloud_elig
        cloud_credits_applied = min(cloud_credits_left, eligible_cloud_cost)
        cloud_credits_left = max(0.0, cloud_credits_left - cloud_credits_applied)

        total_credits_applied = ai_credits_applied + cloud_credits_applied
        estimated_cash_payable = total_modeled_cogs - total_credits_applied

        # CFO KPI telemetry calculations
        total_tokens_m = (raw_input_m + raw_output_m)
        blended_cost_per_m_tokens = (api_cost / total_tokens_m) if total_tokens_m > 0 else 0.0
        compute_copu = (api_cost + infrastructure_cost) / avg_dau if avg_dau > 0 else 0.0
        vector_copu = vector_cost / avg_dau if avg_dau > 0 else 0.0
        credit_burn_velocity = total_credits_applied

        sim_rows.append({
            "Month": month_names[i],
            "Calendar Days": month_days[i],
            "DAU": round(dau_val),
            "Beginning Paid Subscribers": round(beg_paid_subs),
            "New User Conversions": round(new_user_conv_subs),
            "Free Base Conversions": round(free_base_conv_subs),
            "Churned Subscribers": round(churned_subs),
            "Ending Paid Subscribers": round(ending_paid_subs),
            "Avg Active Paid Subs": round(avg_active_paid_subs, 1),
            "Ending Free User Proxy": round(free_users_snapshot),
            "Monthly Prompts": monthly_prompts_total,
            "Ending MRR Run-Rate ($)": ending_mrr_runrate,
            "Recognized Revenue ($)": recognized_revenue,
            "Base Subscription Revenue ($)": base_revenue,
            "Expansion / Overage Revenue ($)": expansion_revenue,
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost,
            "Fixed Infrastructure ($)": infrastructure_cost,
            "Payment Processing ($)": payment_fees,
            "Total Modeled COGS ($)": total_modeled_cogs,
            "Modeled Gross Profit ($)": gross_profit,
            "Modeled Gross Margin (%)": gross_margin_pct,
            "AI Credits Applied ($)": ai_credits_applied,
            "Remaining AI Credits ($)": ai_credits_left,
            "Cloud Credits Applied ($)": cloud_credits_applied,
            "Remaining Cloud Credits ($)": cloud_credits_left,
            "Total Credits Applied ($)": total_credits_applied,
            "Estimated Cash Payable ($)": estimated_cash_payable,
            "Blended Cost / 1M Tokens ($)": blended_cost_per_m_tokens,
            "Compute COPU / DAU ($)": compute_copu,
            "Vector COPU / DAU ($)": vector_copu,
            "Credit Burn Velocity ($)": credit_burn_velocity,
        })

    sim_df = pd.DataFrame(sim_rows)

    if not sim_df.empty:
        assert np.allclose(
            sim_df["Total Modeled COGS ($)"],
            sim_df["API Cost ($)"] + sim_df["Vector DB Cost ($)"] + sim_df["Fixed Infrastructure ($)"] + sim_df["Payment Processing ($)"],
            atol=1e-2
        ), "COGS component reconciliation failed."
        assert np.allclose(
            sim_df["Estimated Cash Payable ($)"] + sim_df["Total Credits Applied ($)"],
            sim_df["Total Modeled COGS ($)"],
            atol=1e-2
        ), "Cash payable and credits reconciliation failed."
        assert (sim_df["Remaining AI Credits ($)"] >= -0.01).all()
        assert (sim_df["Remaining Cloud Credits ($)"] >= -0.01).all()
        assert (sim_df["Estimated Cash Payable ($)"] >= -0.01).all()

    return sim_df

try:
    df = run_model_simulation(
        start_year, start_month, forecast_horizon, starting_dau, starting_paid_subs, monthly_growth, paid_dau_factor,
        new_user_conversion, free_base_conversion, monthly_churn, arpu, expansion_arpu_pct,
        baseline_prompts, peak_prompts, peak_month, surge_width, free_user_usage_mult,
        input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
        frontier_input_price, frontier_output_price, standard_input_price, standard_output_price,
        cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
        step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
        starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share,
        overhead_pct, payment_fee_pct
    ).copy()
except ValueError as e:
    st.error(f"⚠️ Model Configuration Error: {e}")
    st.stop()

df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = (df["Total Modeled COGS ($)"] - budget_monthly) if budget_monthly > 0 else np.nan

df["API Cost / 1,000 Prompts ($)"] = df["API Cost ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Blended COGS / 1,000 Prompts ($)"] = df["Total Modeled COGS ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Blended COGS / Paid Sub ($)"] = df["Total Modeled COGS ($)"] / df["Ending Paid Subscribers"].replace(0, np.nan)

# ==========================================
# 3. CFO DASHBOARD KPI PANEL & DECISION ENGINE
# ==========================================
annual_revenue = df["Recognized Revenue ($)"].sum()
annual_cogs = df["Total Modeled COGS ($)"].sum()
m_last = df.iloc[-1]
last_arr = m_last['Ending MRR Run-Rate ($)'] * 12

ai_dep_idx = df.index[df["Remaining AI Credits ($)"] <= 0.005].tolist() if starting_ai_credits > 0 else []
cloud_dep_idx = df.index[df["Remaining Cloud Credits ($)"] <= 0.005].tolist() if starting_cloud_credits > 0 else []

ai_dep_str = f"Depleted in {df.loc[ai_dep_idx[0], 'Month']}" if ai_dep_idx else ("None ($0)" if starting_ai_credits <= 0 else "Active through Horizon")
cloud_dep_str = f"Depleted in {df.loc[cloud_dep_idx[0], 'Month']}" if cloud_dep_idx else ("None ($0)" if starting_cloud_credits <= 0 else "Active through Horizon")

last_margin = m_last["Modeled Gross Margin (%)"]
if pd.isna(last_margin):
    margin_kpi_str = "Pre-Revenue"
    margin_sub_str = "Undefined ($0 Rev)"
else:
    margin_var = last_margin - margin_target
    margin_kpi_str = f"{last_margin:.1f}%"
    margin_sub_str = f"{margin_var:+.1f}% vs {margin_target}% Target"

c_m1_api = df.loc[0, "API Cost / 1,000 Prompts ($)"]
c_last_api = df.loc[len(df) - 1, "API Cost / 1,000 Prompts ($)"]

if pd.notna(c_m1_api) and pd.notna(c_last_api) and c_m1_api > 0:
    prompt_api_cost_delta = (c_last_api - c_m1_api) / c_m1_api * 100
    api_delta_str = f"{prompt_api_cost_delta:+.1f}% M1 to M{forecast_horizon}"
else:
    api_delta_str = "Change unavailable: baseline is zero or n/a"

api_kpi_str = f"${c_last_api:.4f} / 1k" if pd.notna(c_last_api) else "n/a"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"1. Ending ARR Run-Rate (M{forecast_horizon})", f"${last_arr:,.0f} ARR", f"Recognized Horizon Rev: ${annual_revenue:,.0f}")
c2.metric(f"2. Gross Margin (Before Credits)", margin_kpi_str, margin_sub_str)
c3.metric("3. Pure API Efficiency", api_kpi_str, api_delta_str)
c4.metric("4. Credit Depletion Horizon", f"AI: {ai_dep_str}", f"Cloud: {cloud_dep_str}")

bullets = []
valid_margins = df["Modeled Gross Margin (%)"].dropna()
if valid_margins.empty:
    bullets.append("<b>Gross Margin Status:</b> No recognized revenue; gross margin is undefined for this period.")
else:
    breach = df.index[df["Modeled Gross Margin (%)"].notna() & (df["Modeled Gross Margin (%)"] < margin_target)].tolist()
    if breach:
        breach_m = df.loc[breach[0], "Month"]
        bullets.append(f"<b>Margin Alert:</b> Gross margin drops below the <b>{margin_target}% target</b> in <b>{breach_m}</b>.")
    else:
        bullets.append(f"<b>Margin Compliant:</b> Gross margin remains above the <b>{margin_target}% target</b> across all forecast months.")

if budget_monthly > 0:
    b_breach = df.index[df["Budget Variance ($)"] > 0].tolist()
    if b_breach:
        bullets.append(f"<b>Budget Overrun:</b> Monthly COGS exceeds the <b>${budget_monthly:,.0f} cap</b> starting in <b>{df.loc[b_breach[0], 'Month']}</b>.")

if starting_ai_credits > 0 and ai_dep_idx and starting_cloud_credits > 0 and cloud_dep_idx:
    bullets.append(f"<b>Credit Depletion:</b> AI credits exhausted in <b>{df.loc[ai_dep_idx[0], 'Month']}</b>; Cloud credits exhausted in <b>{df.loc[cloud_dep_idx[0], 'Month']}</b>.")
elif starting_ai_credits > 0 and ai_dep_idx:
    bullets.append(f"<b>Credit Depletion:</b> AI credits are exhausted during <b>{df.loc[ai_dep_idx[0], 'Month']}</b>; eligible inference costs not covered by the remaining balance flow to cash payable.")
elif starting_cloud_credits > 0 and cloud_dep_idx:
    bullets.append(f"<b>Credit Depletion:</b> Cloud credits are exhausted during <b>{df.loc[cloud_dep_idx[0], 'Month']}</b>; eligible infrastructure costs not covered by the remaining balance hit cash.")
else:
    bullets.append("<b>Credit Depletion:</b> Configured credit balances remain active without depleting within the forecast horizon.")

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
    st.subheader(f"Forecast Recognized Revenue, Modeled COGS & Gross Margin — Before Credits ({forecast_horizon}-Month)")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($) [Base + Overages]", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%) — Before Credits", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["Month"], y=[margin_target]*len(df), name=f"Target Margin ({margin_target}%)",
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
        fig.update_yaxes(title_text="Gross Margin (%) — Before Credits", secondary_y=True, range=[y_min, y_max], gridcolor="#edf2f7")
    else:
        fig.update_yaxes(title_text="Gross Margin (%) — Before Credits", secondary_y=True, range=[-100, 100], gridcolor="#edf2f7")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CFO Telemetry: Token Cost Efficiency & COPU Disaggregation")
    col_u1, col_u2 = st.columns(2)

    fig_u1 = go.Figure()
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["Blended Cost / 1M Tokens ($)"], name="Blended Cost / 1M Tokens", mode="lines+markers", line=dict(color="#805ad5", width=2.5)))
    fig_u1.update_layout(title="Effective Blended Cost per 1M Tokens ($)", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.15))
    col_u1.plotly_chart(fig_u1, use_container_width=True)

    fig_u2 = go.Figure()
    fig_u2.add_trace(go.Scatter(x=df["Month"], y=df["Compute COPU / DAU ($)"], name="Compute COPU / DAU", mode="lines+markers", line=dict(color="#319795", width=2.5)))
    fig_u2.add_trace(go.Scatter(x=df["Month"], y=df["Vector COPU / DAU ($)"], name="Vector Storage COPU / DAU", mode="lines+markers", line=dict(color="#dd6b20", width=2, dash="dash")))
    fig_u2.update_layout(title="Infrastructure COPU per DAU ($) — Compute vs Vector", template="plotly_white", height=240, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.15))
    col_u2.plotly_chart(fig_u2, use_container_width=True)

with tab2:
    st.subheader("Cloud Credit Burn Velocity & Out-of-Pocket Cash Payable")
    st.caption("Stacked bars reconcile directly to modeled COGS consumption offset by credit burn velocity.")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["AI Credits Applied ($)"], name="AI Credits Burned ($)", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Cloud Credits Applied ($)"], name="Cloud Credits Burned ($)", marker_color="#4fd1c5"), secondary_y=False)
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
    st.subheader("Multi-Variable Scenario Matrix (Full Reforecast)")
    st.caption("Full 8-scenario matrix comparing revenue, COGS, gross margin, unit telemetry, and out-of-pocket cash payable against Base Case.")

    scenarios_config = [
        ("1. Base Case", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 1.0, frontier_mix, cache_hit_rate),
        ("2. Downside: Growth Slowdown (-50%)", monthly_growth * 0.5, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 1.0, frontier_mix, cache_hit_rate),
        ("3. Downside: High Subscriber Churn (2x)", monthly_growth, monthly_churn * 2.0, new_user_conversion, free_base_conversion, 1.0, 1.0, frontier_mix, cache_hit_rate),
        ("4. Downside: Double Whammy (Low Conv + High Churn)", monthly_growth * 0.5, monthly_churn * 2.0, new_user_conversion * 0.5, free_base_conversion * 0.5, 1.0, 1.0, frontier_mix, cache_hit_rate),
        ("5. Cost Pressure: Context Expansion (+25% Tokens)", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.25, 1.0, frontier_mix, cache_hit_rate),
        ("6. Efficiency: Token Volume Reduction (-25%)", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 0.75, 1.0, frontier_mix, cache_hit_rate),
        ("7. Efficiency: Routing & Caching Optimization", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 1.0, min(frontier_mix, 0.20), min(0.90, cache_hit_rate * 1.5)),
        ("8. Market Shift: Vendor Rate Cut (-25% Price)", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 0.75, frontier_mix, cache_hit_rate),
    ]

    base_rev = df["Recognized Revenue ($)"].sum()
    base_cogs = df["Total Modeled COGS ($)"].sum()
    base_gp = base_rev - base_cogs
    base_gm = (base_gp / base_rev * 100) if base_rev > 0 else np.nan
    base_cash = df["Estimated Cash Payable ($)"].sum()

    scen_results = []
    for label, s_growth, s_churn, s_nconv, s_bconv, s_tok_mult, s_price_mult, s_fmix, s_cache in scenarios_config:
        try:
            s_df = run_model_simulation(
                int(start_year), int(start_month), int(forecast_horizon), float(starting_dau), float(starting_paid_subs), float(s_growth), float(paid_dau_factor), float(s_nconv), float(s_bconv), float(s_churn), float(arpu), float(expansion_arpu_pct),
                float(baseline_prompts), float(peak_prompts), float(peak_month), float(surge_width), float(free_user_usage_mult),
                float(input_tokens_start * s_tok_mult), float(output_tokens_start * s_tok_mult), float(input_growth), float(output_growth), float(s_fmix),
                float(frontier_input_price * s_price_mult), float(frontier_output_price * s_price_mult),
                float(standard_input_price * s_price_mult), float(standard_output_price * s_price_mult),
                float(s_cache), float(cache_price_ratio), float(batch_share), float(batch_price_ratio), float(base_fixed_infra),
                str(step_trigger_type), float(raw_step_threshold), float(step_cost_increment), float(vector_cost_per_user),
                float(starting_ai_credits), float(starting_cloud_credits), float(ai_credit_eligible_share), float(cloud_credit_eligible_share),
                float(overhead_pct), float(payment_fee_pct)
            )
            s_rev = s_df["Recognized Revenue ($)"].sum()
            s_cogs = s_df["Total Modeled COGS ($)"].sum()
            s_gp = s_rev - s_cogs
            s_gm = (s_gp / s_rev * 100) if s_rev > 0 else np.nan
            s_cash = s_df["Estimated Cash Payable ($)"].sum()

            last_idx = len(s_df) - 1
            s_last_api_cost_1k = s_df.loc[last_idx, "API Cost ($)"] / s_df.loc[last_idx, "Monthly Prompts"] * 1000 if s_df.loc[last_idx, "Monthly Prompts"] > 0 else np.nan
            s_last_cogs_sub = s_df.loc[last_idx, "Total Modeled COGS ($)"] / s_df.loc[last_idx, "Ending Paid Subscribers"] if s_df.loc[last_idx, "Ending Paid Subscribers"] > 0 else np.nan

            scen_results.append({
                "Scenario": label,
                "Recognized Revenue ($)": s_rev,
                "Δ Revenue vs Base ($)": s_rev - base_rev,
                "Total COGS ($)": s_cogs,
                "Δ COGS vs Base ($)": s_cogs - base_cogs,
                "Gross Profit ($)": s_gp,
                "Gross Margin (%)": s_gm,
                "Δ Margin (pp)": (s_gm - base_gm) if (not pd.isna(s_gm) and not pd.isna(base_gm)) else np.nan,
                "Final API Cost / 1k ($)": s_last_api_cost_1k,
                "Final COGS / Sub ($)": s_last_cogs_sub,
                "Cash Payable ($)": s_cash,
                "Δ Cash Payable ($)": s_cash - base_cash,
            })
        except ValueError as e:
            scen_results.append({
                "Scenario": f"{label} [Error: {str(e)}]",
                "Recognized Revenue ($)": np.nan,
                "Δ Revenue vs Base ($)": np.nan,
                "Total COGS ($)": np.nan,
                "Δ COGS vs Base ($)": np.nan,
                "Gross Profit ($)": np.nan,
                "Gross Margin (%)": np.nan,
                "Δ Margin (pp)": np.nan,
                "Final API Cost / 1k ($)": np.nan,
                "Final COGS / Sub ($)": np.nan,
                "Cash Payable ($)": np.nan,
                "Δ Cash Payable ($)": np.nan,
            })

    scen_df = pd.DataFrame(scen_results)
    st.dataframe(
        scen_df.style.format({
            "Recognized Revenue ($)": "${:,.0f}",
            "Δ Revenue vs Base ($)": "${:+,.0f}",
            "Total COGS ($)": "${:,.0f}",
            "Δ COGS vs Base ($)": "${:+,.0f}",
            "Gross Profit ($)": "${:,.0f}",
            "Gross Margin (%)": "{:.1f}%",
            "Δ Margin (pp)": "{:+.1f}pp",
            "Final API Cost / 1k ($)": "${:,.4f}",
            "Final COGS / Sub ($)": "${:,.2f}",
            "Cash Payable ($)": "${:,.0f}",
            "Δ Cash Payable ($)": "${:+,.0f}",
        }, na_rep="—"),
        use_container_width=True
    )

with tab4:
    st.subheader("Detailed Financial & Operational Roll-Forward")
    st.dataframe(
        df.style.format({
            "DAU": "{:,.0f}",
            "Beginning Paid Subscribers": "{:,.0f}",
            "New User Conversions": "{:,.0f}",
            "Free Base Conversions": "{:,.0f}",
            "Churned Subscribers": "{:,.0f}",
            "Ending Paid Subscribers": "{:,.0f}",
            "Avg Active Paid Subs": "{:,.1f}",
            "Ending Free User Proxy": "{:,.0f}",
            "Monthly Prompts": "{:,.0f}",
            "Ending MRR Run-Rate ($)": "${:,.0f}",
            "Recognized Revenue ($)": "${:,.0f}",
            "Base Subscription Revenue ($)": "${:,.0f}",
            "Expansion / Overage Revenue ($)": "${:,.0f}",
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
            "Blended Cost / 1M Tokens ($)": "${:,.4f}",
            "Compute COPU / DAU ($)": "${:,.2f}",
            "Vector COPU / DAU ($)": "${:,.2f}",
            "Credit Burn Velocity ($)": "${:,.0f}",
            "Budget ($)": "${:,.0f}",
            "Budget Variance ($)": "${:,.0f}",
            "API Cost / 1,000 Prompts ($)": "${:,.4f}",
            "Blended COGS / 1,000 Prompts ($)": "${:,.4f}",
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

    if step_trigger_type == "DAU Threshold":
        step_threshold_display = f"{step_threshold_input:,.0f} users"
    elif step_trigger_type == "Paid Subscribers Threshold":
        step_threshold_display = f"{step_threshold_input:,.0f} subs"
    else:
        step_threshold_display = f"{step_threshold_input:,.0f}M prompts"

    assumptions_data = [
        {"Parameter": "1. Forecast Horizon & Start", "Current Value": f"{start_period_str} ({forecast_horizon} Months)", "Unit / Scale": "Calendar Period", "Primary Source / Evidence Required": "FP&A Planning Horizon Config", "Verification Status": "User Estimate — Dynamic Calendar Roll-Forward", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "2. Starting DAU & Paid Subs", "Current Value": f"{starting_dau:,} DAU / {starting_paid_subs:,} Subs", "Unit / Scale": "Active Accounts", "Primary Source / Evidence Required": "Mixpanel & Stripe Billing Baseline", "Verification Status": "User Estimate — Pending Actuals Upload", "Owner": "Growth / Finance", "Last Updated": today_str},
        {"Parameter": "3. Monthly DAU Growth", "Current Value": f"{monthly_growth*100:.1f}%", "Unit / Scale": "% MoM Growth", "Primary Source / Evidence Required": "Acquisition Model (Simplified Net DAU Proxy)", "Verification Status": "Management Estimate", "Owner": "Marketing", "Last Updated": today_str},
        {"Parameter": "4. Paid Subscriber Daily Activity", "Current Value": f"{paid_dau_factor*100:.0f}%", "Unit / Scale": "% of paid subscribers", "Primary Source / Evidence Required": "Product analytics / daily activity cohort data", "Verification Status": "Management Estimate — replace with observed paid DAU", "Owner": "Product / Finance", "Last Updated": today_str},
        {"Parameter": "5. New User Day-1 Conversion", "Current Value": f"{new_user_conversion*100:.1f}%", "Unit / Scale": "% Day-1 Signup", "Primary Source / Evidence Required": "Stripe Checkout Day-1 Upgrade Analytics", "Verification Status": "Illustrative Baseline — Source Required (applies M2+)", "Owner": "Growth", "Last Updated": today_str},
        {"Parameter": "6. Free-User Proxy Conversion", "Current Value": f"{free_base_conversion*100:.1f}%", "Unit / Scale": "% Active Free Pool / Mo", "Primary Source / Evidence Required": "In-App Product Funnel Telemetry", "Verification Status": "DAU-based Active Proxy Conversion (prev_dau - beg_paid_subs)", "Owner": "Product", "Last Updated": today_str},
        {"Parameter": "7. Paid Subscriber Churn", "Current Value": f"{monthly_churn*100:.1f}%", "Unit / Scale": "% Monthly Churn", "Primary Source / Evidence Required": "Stripe Billing Dashboard", "Verification Status": "User Estimate", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "8. Monthly ARPU & Expansion", "Current Value": f"${arpu:,.2f} base / {expansion_arpu_pct*100:.0f}% expansion", "Unit / Scale": "$ / Paid Sub / Mo", "Primary Source / Evidence Required": "Subscription Plan & Overage Metering", "Verification Status": "User Estimate — Token Overages & Top-Ups", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "9. Daily Prompts (Paid / Free)", "Current Value": f"{baseline_prompts} / {baseline_prompts*free_user_usage_mult:.1f}", "Unit / Scale": "Prompts / Day", "Primary Source / Evidence Required": "Helicone / Langfuse API Logging", "Verification Status": "Telemetry Estimate", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "10. Usage Surge Curve", "Current Value": f"Peak {peak_prompts} Prompts (Month {peak_month}, σ={surge_width})", "Unit / Scale": "Prompts / Day", "Primary Source / Evidence Required": "Product Event Capacity Model", "Verification Status": "Management Estimate", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "11. Token Context & Growth", "Current Value": f"{input_tokens_start:,} in / {output_tokens_start:,} out", "Unit / Scale": "Tokens / Prompt", "Primary Source / Evidence Required": "Datadog APM / LLM Provider Logs", "Verification Status": "Includes System + RAG Overhead", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "12. Frontier Model Pricing", "Current Value": f"${frontier_input_price:.2f} in / ${frontier_output_price:.2f} out", "Unit / Scale": "$ / 1M Tokens", "Primary Source / Evidence Required": "OpenAI / Anthropic Public Rate Card", "Verification Status": "Vendor Rate Card (Unverified Public Rate)", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "13. Standard Model Pricing", "Current Value": f"${standard_input_price:.4f} in / ${standard_output_price:.4f} out", "Unit / Scale": "$ / 1M Tokens", "Primary Source / Evidence Required": "Vendor Rate Card", "Verification Status": "Vendor Rate Card (Unverified Public Rate)", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "14. Prompt Caching Rules", "Current Value": f"{cache_hit_rate*100:.0f}% hit / {cache_price_ratio*100:.0f}% cost", "Unit / Scale": "% Hit / % Cost", "Primary Source / Evidence Required": "LLM Gateway Caching Logs", "Verification Status": "Telemetry Estimate", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "15. Async Batch Routing", "Current Value": f"{batch_share*100:.0f}% traffic / {batch_price_ratio*100:.0f}% cost", "Unit / Scale": "% Async Share", "Primary Source / Evidence Required": "Batch API Route Telemetry", "Verification Status": "Management Estimate", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "16. Base Fixed Infra Spend", "Current Value": f"${base_fixed_infra:,.0f}/mo", "Unit / Scale": "$ / Month", "Primary Source / Evidence Required": "AWS / GCP Monthly Invoices", "Verification Status": "Covers Tier 1 Base Capacity (Block 1)", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "17. Infra Step-Up Trigger", "Current Value": f"${step_cost_increment:,.0f} per {step_threshold_display}", "Unit / Scale": "$ / Capacity Block", "Primary Source / Evidence Required": "DevOps Infrastructure Capacity Plan", "Verification Status": "Triggers on excess capacity beyond Block 1", "Owner": "Engineering", "Last Updated": today_str},
        {"Parameter": "18. Vector DB Unit Cost", "Current Value": f"${vector_cost_per_user:.2f} / DAU / mo", "Unit / Scale": "$ / DAU / Month", "Primary Source / Evidence Required": "Pinecone / Qdrant Invoice Rate", "Verification Status": "Simplified Storage + Query Driver (scaled on avg DAU)", "Owner": "DevOps", "Last Updated": today_str},
        {"Parameter": "19. Starting Credit Balances & Burn", "Current Value": f"${starting_ai_credits:,.0f} AI / ${starting_cloud_credits:,.0f} Cloud", "Unit / Scale": "$ Total Grant & Burn", "Primary Source / Evidence Required": "AWS Activate & OpenAI Portal Grants", "Verification Status": "Tracked via Monthly Credit Burn Velocity", "Owner": "Finance", "Last Updated": today_str},
        {"Parameter": "20. Inference Overhead & Fees", "Current Value": f"{overhead_pct*100:.0f}% overhead / {payment_fee_pct*100:.1f}% fee", "Unit / Scale": "% Uplift / % Rev", "Primary Source / Evidence Required": "Gateway Logs & Stripe/App Store Schedule", "Verification Status": "Management Estimate", "Owner": "Engineering / Finance", "Last Updated": today_str},
        {"Parameter": "21. Budget Cap & Target Margin", "Current Value": f"${budget_monthly:,.0f}/mo budget / {margin_target}% margin", "Unit / Scale": "$ Cap / % Margin", "Primary Source / Evidence Required": "FP&A Target Guardrails", "Verification Status": "Management Policy", "Owner": "Finance", "Last Updated": today_str},
    ]

    st.table(pd.DataFrame(assumptions_data))

# MODEL SCOPE FOOTNOTE
st.markdown("---")
st.caption("© 2026. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
st.markdown("""
<div style="font-size:0.75rem; color:#a0aec0; line-height:1.3;">
<b>Model Financial Scope & Perimeter Disclaimer:</b> This tool models direct AI inference, vector storage, infrastructure, and payment-processing COGS (Direct Cost of Goods Sold). It evaluates <b>Cloud-Credit Depletion Horizons</b> and out-of-pocket cash payable for direct production expenses. It does NOT calculate total corporate cash burn or startup runway, which requires non-COGS operating expenses (payroll, marketing, sales CAC, G&A, legal), starting cash balances, cash receipts timing, and equity/debt financing activities.
</div>
""", unsafe_allow_html=True)
