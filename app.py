"""
AI Inference, Credit Depletion & Contract Pricing Deal Desk Model
-----------------------------------------------------------------
Institutional-grade FP&A planning model for evaluating direct AI model inference,
vector storage, multi-tier contract pricing, Pareto skew, and credit burn velocity.
"""

import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference Deal Desk & Unit Economics Model",
    page_icon="⚡",
    layout="wide"
)

# Custom Soft Neutral Styling & Ultra-Compact Sidebar Elements
st.markdown("""
<style>
.stApp {background:#fcfbf9;color:#2d3748;font-family:'Inter', sans-serif;}
section[data-testid="stSidebar"] {background:#f5f2ed !important; border-right: 1px solid #e2d9d0;}

/* Tighten Sidebar Element Spacing */
section[data-testid="stSidebar"] div.stBlock {gap: 0.35rem !important;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    margin-bottom: -0.15rem !important;
    padding-top: 0rem !important;
}

/* Compact Sidebar Headers */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important;
    margin-top: 6px !important;
    margin-bottom: 2px !important;
    padding-bottom: 0px !important;
    font-weight: 700 !important;
}

/* Compact Widget Labels & Padding */
section[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    margin-bottom: 1px !important;
    color: #4a5568 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
}
section[data-testid="stSidebar"] hr {
    margin-top: 6px !important;
    margin-bottom: 6px !important;
}

/* Compact Metric Cards */
div[data-testid="stMetric"] {
    background:#fff;
    border:1px solid #e2d9d0;
    padding:6px 10px !important;
    border-radius:6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {font-size:0.75rem !important; color:#615a52 !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.1rem !important; font-weight:700;}

/* Compact Header Banner */
.compact-header {
    background: linear-gradient(135deg,#f5f0eb,#ebe5df);
    padding: 8px 14px;
    border-radius: 6px;
    border: 1px solid #e2d9d0;
    margin-bottom: 8px;
}
.compact-header h1 {color:#2d3748; margin:0 0 2px 0; font-size:1.25rem; font-weight:700;}
.compact-header p {color:#615a52; margin:0; font-size:0.8rem;}

/* CFO Commentary Box */
.cfo-commentary {
    background: #ffffff;
    border: 1px solid #cbd5e0;
    border-left: 4px solid #3182ce;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.cfo-commentary h4 {margin:0 0 4px 0; font-size:0.9rem; color:#2b6cb0; font-weight:700;}
.cfo-commentary ul {margin:0; padding-left:16px; font-size:0.82rem; color:#4a5568;}
.cfo-commentary li {margin-bottom:2px;}
</style>
""", unsafe_allow_html=True)

# Compact Header Banner
st.markdown("""
<div class="compact-header">
    <h1>⚡ AI Deal Desk & Unit Economics Simulator</h1>
    <p>Founder onboarding presets, enterprise contract economics, typical/heavy-user costs, comparable pricing simulations, and credit runway.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. SIDEBAR ASSUMPTIONS & UX GOVERNANCE
# ==========================================
st.sidebar.header("0. Founder Onboarding")
founder_stage = st.sidebar.selectbox(
    "Choose a starting profile",
    ["Early traction", "Pre-launch", "Scaling"],
    index=0,
    help="Presets populate starting values only. All assumptions remain editable below."
)
FOUNDER_PRESETS = {
    "Pre-launch": {"dau": 100, "pro": 0, "ent": 0, "growth": 5.0, "deals": 0, "prompts": 3, "infra": 500.0, "ai_credits": 0.0, "cloud_credits": 0.0},
    "Early traction": {"dau": 5000, "pro": 200, "ent": 50, "growth": 8.0, "deals": 3, "prompts": 5, "infra": 2500.0, "ai_credits": 20000.0, "cloud_credits": 30000.0},
    "Scaling": {"dau": 50000, "pro": 2500, "ent": 250, "growth": 10.0, "deals": 8, "prompts": 8, "infra": 10000.0, "ai_credits": 50000.0, "cloud_credits": 100000.0},
}
preset = FOUNDER_PRESETS[founder_stage]
current_now = datetime.datetime.now()
start_year = st.sidebar.number_input("Forecast Start Year", min_value=2024, max_value=2035, value=current_now.year)
start_month = st.sidebar.slider("Forecast Start Month", 1, 12, current_now.month)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 36, 12)

# UX Toggle: Simple vs. Advanced Mode
st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("🛠️ Enable Advanced Deal Desk Parameters", value=False, help="Start with the simpler founder view. Turn this on to access granular model assumptions.")

# --- CORE / DEFAULT INPUTS (Always Visible) ---
st.sidebar.header("2. Core Growth & Funnel")
starting_dau = st.sidebar.number_input("Starting Daily Active Users (DAU)", min_value=0, value=preset["dau"], step=500, key=f"dau_{founder_stage}")
starting_pro_subs = st.sidebar.number_input("Starting Pro Subscribers", min_value=0, value=preset["pro"], step=50, key=f"pro_{founder_stage}")
starting_ent_subs = st.sidebar.number_input("Starting Enterprise Contracts", min_value=0, value=preset["ent"], step=10, key=f"ent_{founder_stage}")
starting_paid_subs = starting_pro_subs + starting_ent_subs

# PLG Motion vs. Sales-Led Motion
pro_monthly_growth = st.sidebar.slider("Pro Tier Monthly DAU Growth (%)", 0.0, 25.0, preset["growth"], 0.5, key=f"growth_{founder_stage}") / 100
net_new_ent_contracts = st.sidebar.number_input("Net New Enterprise Deals / Month", min_value=0, value=preset["deals"], step=1, help="Sales pipeline velocity independent of PLG web traffic.")
baseline_prompts = st.sidebar.slider("Baseline Prompts / Paid User / Day", 1, 20, preset["prompts"], key=f"prompts_{founder_stage}")

# --- ADVANCED INPUTS (Inside Collapsible Expanders if Advanced Mode is True) ---
if advanced_mode:
    with st.sidebar.expander("📂 1. Subscriber Funnel & Conversion Levers"):
        paid_dau_factor = st.slider("Paid Subscribers Active Daily (%)", 0, 100, 70, 5) / 100
        new_user_conversion = st.slider("New PLG User Conversion to Pro (%)", 0.0, 50.0, 15.0, 0.5) / 100
        free_base_conversion = st.slider("Free-User Proxy Conversion (%)", 0.0, 10.0, 2.0, 0.1) / 100
        ent_monthly_growth = st.slider("Enterprise Account Expansion (%)", 0.0, 25.0, 3.0, 0.5) / 100
        monthly_churn = st.slider("Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5) / 100
        arpu_pro = st.number_input("Pro Tier Monthly ARPU ($)", min_value=0.0, value=30.0, step=5.0)
        arpu_ent = st.number_input("Enterprise Tier Monthly ARPU ($)", min_value=0.0, value=250.0, step=25.0)
        expansion_arpu_pct = st.slider("Expansion / Overage Revenue (% of ARPU)", 0.0, 50.0, 10.0, 1.0) / 100
        payment_fee_pct = st.slider("Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1) / 100
        enterprise_contract_months = st.number_input("New Enterprise Contract Term (months)", min_value=1, max_value=120, value=12, step=1)
        enterprise_implementation_fee = st.number_input("Implementation Fee per New Enterprise Deal ($)", min_value=0.0, value=0.0, step=500.0)
        enterprise_prompts_per_account = st.number_input("Enterprise Prompts per Account / Month", min_value=0, value=3000, step=500)
    
    with st.sidebar.expander("📂 2. Multi-Peak Usage & Pareto Skew"):
        st.markdown("**Peak 1 (Spring / Launch)**")
        peak_prompts_1 = st.slider("Peak 1 Volume", 1, 100, 35)
        peak_month_1 = st.slider("Peak 1 Month", 1, forecast_horizon, min(3, forecast_horizon))
        surge_width_1 = st.slider("Peak 1 Spread ($\\sigma$)", 0.5, 4.0, 1.5, 0.5)
        
        st.markdown("**Peak 2 (Fall / Holiday)**")
        peak_prompts_2 = st.slider("Peak 2 Volume", 1, 100, 45)
        peak_month_2 = st.slider("Peak 2 Month", 1, forecast_horizon, min(11, forecast_horizon))
        surge_width_2 = st.slider("Peak 2 Spread ($\\sigma$)", 0.5, 4.0, 2.0, 0.5)
        
        pareto_skew_share = st.slider("Power-User Token Skew Share (%) — reference only", 50, 95, 80, 5) / 100
        heavy_user_share = st.slider("Heavy Users as Share of Pro Users (%)", 0, 50, 20, 5) / 100
        heavy_user_multiplier = st.slider("Heavy-User Prompt Multiplier (vs. Typical)", 1.0, 20.0, 4.0, 0.5)
        free_user_usage_mult = st.slider("Free Tier Usage Multiplier (%)", 0, 100, 50) / 100
        input_tokens_start = st.number_input("Initial Input Tokens / Prompt", min_value=0, value=1000, step=100)
        output_tokens_start = st.number_input("Initial Output Tokens / Prompt", min_value=0, value=400, step=50)
        input_growth = st.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
        output_growth = st.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

    with st.sidebar.expander("📂 3. Token Routing & Pricing Rules"):
        frontier_mix = st.slider("Frontier Model Traffic Share (%)", 0, 100, 30) / 100
        frontier_input_price = st.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25)
        frontier_output_price = st.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50)
        standard_input_price = st.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
        standard_output_price = st.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f")
        cache_hit_rate = st.slider("Eligible Input Tokens Cached (%)", 0, 90, 40) / 100
        cache_price_ratio = st.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
        batch_share = st.slider("Traffic Using Batch Pricing (%)", 0, 80, 20) / 100
        batch_price_ratio = st.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100
        overhead_pct = st.slider("Inference Overhead (%)", 0, 100, 0) / 100

    with st.sidebar.expander("📂 4. Infrastructure & Credit Drawdown"):
        base_fixed_infra = st.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=preset["infra"], step=500.0)
        step_trigger_type = st.selectbox("Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"])
        if step_trigger_type == "DAU Threshold":
            step_threshold_input = st.number_input("DAU Capacity Block (Users)", min_value=1000, value=20000, step=5000)
            raw_step_threshold = float(step_threshold_input)
        elif step_trigger_type == "Paid Subscribers Threshold":
            step_threshold_input = st.number_input("Subscriber Capacity Block (Subs)", min_value=500, value=5000, step=1000)
            raw_step_threshold = float(step_threshold_input)
        else:
            step_threshold_input = st.number_input("Monthly Prompt Block (Millions)", min_value=1, value=20, step=5)
            raw_step_threshold = float(step_threshold_input * 1_000_000)
        step_cost_increment = st.number_input("Infra Step-Up Cost ($ per block)", min_value=0.0, value=1500.0, step=250.0)
        vector_cost_per_user = st.number_input("Vector DB Cost / DAU / Month ($)", min_value=0.0, value=0.15, step=0.05)
        starting_ai_credits = st.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=preset["ai_credits"], step=2500.0)
        starting_cloud_credits = st.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=preset["cloud_credits"], step=5000.0)
        ai_credit_expiry_month = st.number_input("AI Credits Expire After Forecast Month (0 = no expiry in horizon)", min_value=0, max_value=36, value=0, step=1)
        cloud_credit_expiry_month = st.number_input("Cloud Credits Expire After Forecast Month (0 = no expiry in horizon)", min_value=0, max_value=36, value=0, step=1)
        invoice_lag_months = st.selectbox("Invoice Payment Lag (months)", [0, 1, 2, 3], index=0, help="Cash paid is shifted by this many forecast months; invoices unpaid at the horizon are shown as outstanding.")
        ai_credit_eligible_share = st.slider("AI Credit Eligible Share (%)", 0, 100, 100) / 100
        cloud_credit_eligible_share = st.slider("Cloud Credit Eligible Share (%)", 0, 100, 100) / 100
        margin_target = st.slider("Target Gross Margin (%)", 0, 90, 60)
        budget_monthly = st.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)
else:
    paid_dau_factor = 0.70
    new_user_conversion = 0.15
    free_base_conversion = 0.02
    ent_monthly_growth = 0.03
    monthly_churn = 0.03
    arpu_pro = 30.0
    arpu_ent = 250.0
    expansion_arpu_pct = 0.10
    payment_fee_pct = 0.029
    enterprise_contract_months = 12
    enterprise_implementation_fee = 0.0
    enterprise_prompts_per_account = 3000
    heavy_user_share = 0.20
    heavy_user_multiplier = 4.0
    ai_credit_expiry_month = 0
    cloud_credit_expiry_month = 0
    invoice_lag_months = 0
    peak_prompts_1 = 35
    peak_month_1 = min(3, forecast_horizon)
    surge_width_1 = 1.5
    peak_prompts_2 = 45
    peak_month_2 = min(11, forecast_horizon)
    surge_width_2 = 2.0
    pareto_skew_share = 0.80
    free_user_usage_mult = 0.50
    input_tokens_start = 1000
    output_tokens_start = 400
    input_growth = 0.03
    output_growth = 0.0
    frontier_mix = 0.30
    frontier_input_price = 2.50
    frontier_output_price = 10.00
    standard_input_price = 0.15
    standard_output_price = 0.60
    cache_hit_rate = 0.40
    cache_price_ratio = 0.20
    batch_share = 0.20
    batch_price_ratio = 0.50
    overhead_pct = 0.0
    base_fixed_infra = preset["infra"]
    step_trigger_type = "DAU Threshold"
    raw_step_threshold = 20000.0
    step_cost_increment = 1500.0
    vector_cost_per_user = 0.15
    starting_ai_credits = preset["ai_credits"]
    starting_cloud_credits = preset["cloud_credits"]
    ai_credit_eligible_share = 1.0
    cloud_credit_eligible_share = 1.0
    margin_target = 60
    budget_monthly = 0.0


# ==========================================
# 2. CORE ENGINE SIMULATION FUNCTION
# ==========================================
@st.cache_data(show_spinner=False)
def run_model_simulation(
    p_start_year, p_start_month, p_horizon, p_dau, p_pro_subs, p_ent_subs, p_pro_growth, p_net_new_ent, p_paid_dau_factor,
    p_new_conv, p_base_conv, p_ent_expansion, p_churn, p_arpu_pro, p_arpu_ent, p_expansion_pct,
    p_base_prompts, p_p1_vol, p_p1_m, p_s1_w, p_p2_vol, p_p2_m, p_s2_w, p_free_mult,
    p_in_tok, p_out_tok, p_in_grow, p_out_grow, p_front_mix,
    p_f_in_p, p_f_out_p, p_s_in_p, p_s_out_p, p_cache_rate, p_cache_ratio,
    p_batch_share, p_batch_price_ratio, p_base_infra, p_step_type, p_step_thresh_raw,
    p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred, p_ai_elig, p_cloud_elig,
    p_overhead, p_fee_pct, p_ent_term=12, p_impl_fee=0.0, p_ent_prompts=3000,
    p_heavy_share=0.20, p_heavy_mult=4.0, p_ai_expiry_month=0, p_cloud_expiry_month=0, p_invoice_lag=0
):
    n = int(p_horizon)
    dates = [pd.Period(freq='M', year=int(p_start_year), month=int(p_start_month)) + i for i in range(n)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    m_indices = np.arange(1, n + 1)

    curve_1 = (p_p1_vol - p_base_prompts) * np.exp(-((m_indices - p_p1_m) ** 2) / (2 * p_s1_w ** 2))
    curve_2 = (p_p2_vol - p_base_prompts) * np.exp(-((m_indices - p_p2_m) ** 2) / (2 * p_s2_w ** 2))
    prompt_curve = p_base_prompts + curve_1 + curve_2

    ai_credits_left = float(p_ai_cred)
    cloud_credits_left = float(p_cloud_cred)
    # Newly signed enterprise contracts are tracked by cohort so they stop contributing
    # after their modeled term. Starting contracts are treated as pre-existing/renewed.
    enterprise_cohorts = []
    accrued_payables = []

    current_dau = float(p_dau)
    prev_pro_subs = float(p_pro_subs)
    prev_ent_subs = float(p_ent_subs)

    sim_rows = []

    for i in range(n):
        beg_pro = prev_pro_subs
        beg_ent = prev_ent_subs
        beg_paid_subs = beg_pro + beg_ent

        # Motion 1: PLG Pro Tier (Driven by organic DAU funnel)
        if i == 0:
            dau_val = current_dau
            new_users_acquired = 0.0
            free_base = max(0.0, dau_val - beg_paid_subs)
            prev_dau_val = current_dau
        else:
            prev_dau_val = current_dau
            current_dau = prev_dau_val * (1 + p_pro_growth)
            dau_val = current_dau
            new_users_acquired = dau_val - prev_dau_val
            free_base = max(0.0, prev_dau_val - beg_paid_subs)

        # 100% of PLG web funnel conversions feed Pro
        new_pro_conv = (new_users_acquired * p_new_conv) + (free_base * p_base_conv)
        churned_pro = beg_pro * p_churn
        ending_pro = max(0.0, beg_pro + new_pro_conv - churned_pro)

        # Motion 2: Sales-Led Enterprise Tier (Decoupled sales velocity)
        new_ent_conv = float(p_net_new_ent)
        # New deals are signed in each forecast month. Each cohort expires after
        # its configured term; starting contracts are treated as pre-existing.
        expired_new_contracts = sum(count for age, count in enterprise_cohorts if age + 1 >= int(p_ent_term))
        enterprise_cohorts = [(age + 1, count * (1 + p_ent_expansion)) for age, count in enterprise_cohorts if age + 1 < int(p_ent_term)]
        if new_ent_conv > 0:
            enterprise_cohorts.append((0, new_ent_conv))
        churned_ent = beg_ent * (p_churn * 0.5)
        ending_ent = max(0.0, (beg_ent * (1 + p_ent_expansion)) + new_ent_conv - churned_ent - expired_new_contracts)

        ending_paid_subs = ending_pro + ending_ent

        prev_pro_subs = ending_pro
        prev_ent_subs = ending_ent

        avg_pro = (beg_pro + ending_pro) / 2.0
        avg_ent = (beg_ent + ending_ent) / 2.0
        avg_dau = (prev_dau_val + dau_val) / 2.0
        
        paid_dau = (avg_pro + avg_ent) * p_paid_dau_factor
        free_dau_avg = max(0.0, avg_dau - paid_dau)
        free_users_snapshot = max(0.0, dau_val - ending_paid_subs)

        input_tokens_val = p_in_tok * ((1 + p_in_grow) ** i)
        output_tokens_val = p_out_tok * ((1 + p_out_grow) ** i)

        prompts_per_typical_user_day = max(0.0, float(prompt_curve[i]))
        prompts_per_heavy_user_day = prompts_per_typical_user_day * max(1.0, p_heavy_mult)
        weighted_pro_prompts_day = prompts_per_typical_user_day * (1 - p_heavy_share) + prompts_per_heavy_user_day * p_heavy_share
        prompts_per_free_user_day = prompts_per_typical_user_day * p_free_mult

        pro_paid_dau = avg_pro * p_paid_dau_factor
        monthly_prompts_typical = pro_paid_dau * (1 - p_heavy_share) * prompts_per_typical_user_day * month_days[i]
        monthly_prompts_heavy = pro_paid_dau * p_heavy_share * prompts_per_heavy_user_day * month_days[i]
        monthly_prompts_enterprise = avg_ent * max(0.0, p_ent_prompts)
        monthly_prompts_paid = monthly_prompts_typical + monthly_prompts_heavy + monthly_prompts_enterprise
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

        step_multiplier = max(0, math.ceil(trigger_metric / p_step_thresh_raw) - 1) if p_step_thresh_raw > 0 and trigger_metric > p_step_thresh_raw else 0
        infrastructure_cost = p_base_infra + (step_multiplier * p_step_cost)

        base_rev_pro = avg_pro * p_arpu_pro
        base_rev_ent = avg_ent * p_arpu_ent
        base_revenue = base_rev_pro + base_rev_ent
        expansion_revenue = base_revenue * p_expansion_pct
        implementation_revenue = new_ent_conv * max(0.0, p_impl_fee)
        recognized_revenue = base_revenue + expansion_revenue + implementation_revenue
        ending_mrr_runrate = (ending_pro * p_arpu_pro + ending_ent * p_arpu_ent) * (1 + p_expansion_pct)
        payment_fees = recognized_revenue * p_fee_pct

        total_modeled_cogs = api_cost + vector_cost + infrastructure_cost + payment_fees
        gross_profit = recognized_revenue - total_modeled_cogs
        gross_margin_pct = (gross_profit / recognized_revenue * 100) if recognized_revenue > 0 else np.nan

        # A grant is usable through its stated forecast month, then any unused
        # balance expires before the following month begins.
        ai_credits_expired = 0.0
        cloud_credits_expired = 0.0
        if p_ai_expiry_month > 0 and (i + 1) > p_ai_expiry_month and ai_credits_left > 0:
            ai_credits_expired = ai_credits_left
            ai_credits_left = 0.0
        if p_cloud_expiry_month > 0 and (i + 1) > p_cloud_expiry_month and cloud_credits_left > 0:
            cloud_credits_expired = cloud_credits_left
            cloud_credits_left = 0.0

        eligible_api_cost = api_cost * p_ai_elig
        ai_credits_applied = min(ai_credits_left, eligible_api_cost)
        ai_credits_left = max(0.0, ai_credits_left - ai_credits_applied)

        eligible_cloud_cost = (vector_cost + infrastructure_cost) * p_cloud_elig
        cloud_credits_applied = min(cloud_credits_left, eligible_cloud_cost)
        cloud_credits_left = max(0.0, cloud_credits_left - cloud_credits_applied)

        total_credits_applied = ai_credits_applied + cloud_credits_applied
        estimated_cash_payable = max(0.0, total_modeled_cogs - total_credits_applied)
        accrued_payables.append(estimated_cash_payable)
        cash_paid_this_month = accrued_payables[i - int(p_invoice_lag)] if i >= int(p_invoice_lag) else 0.0
        outstanding_invoices = sum(accrued_payables[max(0, i - int(p_invoice_lag) + 1):i + 1]) if p_invoice_lag > 0 else 0.0

        sim_rows.append({
            "Month": month_names[i],
            "DAU": round(dau_val),
            "Ending Pro Subs": round(ending_pro),
            "Ending Ent Subs": round(ending_ent),
            "Ending Paid Subscribers": round(ending_paid_subs),
            "Average Paid Subscribers": avg_pro + avg_ent,
            "New Enterprise Deals": new_ent_conv,
            "Implementation Revenue ($)": implementation_revenue,
            "Typical Pro Prompts": monthly_prompts_typical,
            "Heavy Pro Prompts": monthly_prompts_heavy,
            "Enterprise Prompts": monthly_prompts_enterprise,
            "Monthly Prompts": monthly_prompts_total,
            "Ending MRR Run-Rate ($)": ending_mrr_runrate,
            "Recognized Revenue ($)": recognized_revenue,
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost,
            "Fixed Infrastructure ($)": infrastructure_cost,
            "Payment Processing ($)": payment_fees,
            "Total Modeled COGS ($)": total_modeled_cogs,
            "Modeled Gross Profit ($)": gross_profit,
            "Modeled Gross Margin (%)": gross_margin_pct,
            "Remaining AI Credits ($)": ai_credits_left,
            "Remaining Cloud Credits ($)": cloud_credits_left,
            "AI Credits Expired ($)": ai_credits_expired,
            "Cloud Credits Expired ($)": cloud_credits_expired,
            "AI Credits Applied ($)": ai_credits_applied,
            "Cloud Credits Applied ($)": cloud_credits_applied,
            "Estimated Cash Payable ($)": estimated_cash_payable,
            "Estimated Cash Paid This Month ($)": cash_paid_this_month,
            "Outstanding Invoices at Horizon ($)": outstanding_invoices,
        })

    return pd.DataFrame(sim_rows)

try:
    df = run_model_simulation(
        start_year, start_month, forecast_horizon, starting_dau, starting_pro_subs, starting_ent_subs, 
        pro_monthly_growth, net_new_ent_contracts, paid_dau_factor,
        new_user_conversion, free_base_conversion, ent_monthly_growth, monthly_churn, arpu_pro, arpu_ent, expansion_arpu_pct,
        baseline_prompts, peak_prompts_1, peak_month_1, surge_width_1, peak_prompts_2, peak_month_2, surge_width_2, free_user_usage_mult,
        input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
        frontier_input_price, frontier_output_price, standard_input_price, standard_output_price,
        cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
        step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
        starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share,
        overhead_pct, payment_fee_pct, enterprise_contract_months, enterprise_implementation_fee,
        enterprise_prompts_per_account, heavy_user_share, heavy_user_multiplier,
        ai_credit_expiry_month, cloud_credit_expiry_month, invoice_lag_months
    ).copy()
except ValueError as e:
    st.error(f"⚠️ Model Configuration Error: {e}")
    st.stop()

# Input sanity checks: warn without silently changing the founder's assumptions.
if starting_pro_subs + starting_ent_subs > starting_dau:
    st.warning("Starting paid subscribers exceed DAU. The model can still run, but free-user estimates and usage allocation may be misleading; check that these metrics use compatible definitions.")
if starting_dau == 0 and starting_pro_subs + starting_ent_subs > 0:
    st.warning("You have paid subscribers but zero DAU. Confirm whether DAU and paid-subscriber counts refer to the same product and period.")


# ==========================================
# 3. REVERSE-MARGIN SOLVER & CFO BRIEFING
# ==========================================
m_last = df.iloc[-1]
last_arr = m_last['Ending MRR Run-Rate ($)'] * 12
annual_rev = df["Recognized Revenue ($)"].sum()
annual_cogs = df["Total Modeled COGS ($)"].sum()
annual_gp = annual_rev - annual_cogs
annual_gm = (annual_gp / annual_rev * 100) if annual_rev > 0 else 0.0
last_margin = m_last["Modeled Gross Margin (%)"]

# Approximate per-Pro usage ceiling at the selected margin target.
# Allocate non-API modeled costs across ending paid subscribers, then solve for
# the remaining API-cost allowance. This is a planning estimate, not a contract guarantee.
blended_token_cost_m1 = (df.loc[0, "API Cost ($)"] / df.loc[0, "Monthly Prompts"]) if df.loc[0, "Monthly Prompts"] > 0 else 0.0
pro_arpu_total = arpu_pro * (1 + expansion_arpu_pct)
target_cogs_per_pro = pro_arpu_total * (1 - margin_target / 100)
ending_paid_for_allocation = max(float(m_last["Ending Paid Subscribers"]), 1.0)
non_api_cost_per_paid = (
    float(m_last["Vector DB Cost ($)"] + m_last["Fixed Infrastructure ($)"] + m_last["Payment Processing ($)"])
    / ending_paid_for_allocation
)
allowable_api_cost_per_pro = max(0.0, target_cogs_per_pro - non_api_cost_per_paid)
max_prompts_target = (allowable_api_cost_per_pro / blended_token_cost_m1) if blended_token_cost_m1 > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{forecast_horizon})", f"${last_arr:,.0f} ARR", f"Forecast-period revenue: ${annual_rev:,.0f}")
c2.metric("Forecast-Period Gross Margin", f"{annual_gm:.1f}%", f"Target: {margin_target}%")
c3.metric(f"Estimated Pro Usage Ceiling ({margin_target}% GM)", f"{max_prompts_target:,.0f} prompts/mo", "Approximate per-customer limit")
c4.metric("Heavy-User Cohort", f"{heavy_user_share*100:.0f}% of Pro users", f"{heavy_user_multiplier:.1f}× typical prompts; modeled")

policy_recs = []
if last_margin < margin_target:
    policy_recs.append(f"<b>Deal Desk Advisory:</b> Forecast-period gross margin ({last_margin:.1f}%) breaches target ({margin_target}%). <b>Action:</b> Test a usage cap, overage price, or lower-cost model routing. The estimated Pro usage ceiling is about {max_prompts_target:,.0f} prompts/month under the current assumptions; validate it with customer-level usage data before enforcing it.")
else:
    policy_recs.append(f"<b>Deal Desk Status:</b> Gross margin remains robust at {last_margin:.1f}%, comfortably meeting the {margin_target}% hurdle rate.")

if m_last["Remaining AI Credits ($)"] <= 0:
    ai_exhaustion_months = df.loc[(df["Remaining AI Credits ($)"].shift(1, fill_value=starting_ai_credits) > 0) & (df["Remaining AI Credits ($)"] <= 0), "Month"]
    ai_timing = ai_exhaustion_months.iloc[0] if len(ai_exhaustion_months) else "before or at forecast start"
    policy_recs.append(f"<b>Credit Depletion Warning:</b> AI credits are exhausted by {ai_timing}. Check the Credits & Cash tab for expiry and invoice-lag effects.")
else:
    policy_recs.append(f"<b>Credit Runway:</b> ${m_last['Remaining AI Credits ($)']:,.0f} in AI vendor credits remain at horizon end.")

st.markdown(f"""
<div class="cfo-commentary">
    <h4>💡 Deal Desk & Contract Pricing Governance Briefing</h4>
    <ul>
        {"".join([f"<li>{rec}</li>" for rec in policy_recs])}
    </ul>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. DASHBOARD TABS (FULL 6-TAB STRUCTURE)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Revenue & Margin", 
    "💳 Credits & Cash", 
    "📋 Monthly Detail", 
    "📈 Scenario Matrix", 
    "📑 Audit Register",
    "🎯 Value & Pricing Strategy"
])

with tab1:
    st.subheader("Forecast Recognized Revenue, COGS & Gross Margin (Multi-Tier)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Founder Unit-Economics Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    end_paid = float(m_last["Ending Paid Subscribers"])
    end_cogs_per_paid = float(m_last["Total Modeled COGS ($)"]) / end_paid if end_paid > 0 else 0.0
    end_rev_per_paid = float(m_last["Recognized Revenue ($)"]) / end_paid if end_paid > 0 else 0.0
    end_gp_per_paid = end_rev_per_paid - end_cogs_per_paid
    k1.metric("Ending-month COGS / paid subscriber", f"${end_cogs_per_paid:,.2f}")
    k2.metric("Ending-month revenue / paid subscriber", f"${end_rev_per_paid:,.2f}")
    k3.metric("Ending-month GP / paid subscriber", f"${end_gp_per_paid:,.2f}")
    k4.metric("Ending-month modeled COGS", f"${float(m_last['Total Modeled COGS ($)']):,.0f}")
    st.caption("These are blended company-level averages, not individual-customer profitability. Enterprise and Pro customers may have very different usage and costs.")

    st.markdown("#### COGS Component Disaggregation")
    cogs_fig = go.Figure()
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["API Cost ($)"], name="API Inference Cost", marker_color="#4299e1"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Vector DB Cost ($)"], name="Vector DB Cost", marker_color="#4fd1c5"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Fixed Infrastructure ($)"], name="Infrastructure Step-Cost", marker_color="#ed8936"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Payment Processing ($)"], name="Payment Processing Fees", marker_color="#9f7aea"))
    cogs_fig.update_layout(template="plotly_white", barmode="stack", height=320, margin=dict(l=30, r=30, t=20, b=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(cogs_fig, use_container_width=True)

with tab2:
    st.subheader("Credit Drawdown Velocity & Estimated Cash Payable")
    st.caption("Accrued cash payable is modeled COGS less eligible promotional credits. Cash paid is shifted by the selected invoice lag; credit expiry is modeled by forecast month. Taxes and non-modeled expenses remain excluded.")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Accrued Cash Payable ($)", marker_color="#dd6b20"))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Estimated Cash Paid This Month ($)"], name="Estimated Cash Paid ($)", mode="lines+markers", line=dict(color="#805ad5", width=2)))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color="#38a169", width=2)))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Cloud Credits ($)"], name="Remaining Cloud Credits ($)", mode="lines", line=dict(color="#3182ce", width=2, dash="dot")))
    fig2.update_layout(template="plotly_white", height=340, barmode="group", margin=dict(l=30, r=30, t=30, b=20), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, use_container_width=True)
    ai_exhausted = df.loc[(df["Remaining AI Credits ($)"].shift(1, fill_value=starting_ai_credits) > 0) & (df["Remaining AI Credits ($)"] <= 0) & (df["AI Credits Expired ($)"] == 0), "Month"]
    cloud_exhausted = df.loc[(df["Remaining Cloud Credits ($)"].shift(1, fill_value=starting_cloud_credits) > 0) & (df["Remaining Cloud Credits ($)"] <= 0) & (df["Cloud Credits Expired ($)"] == 0), "Month"]
    ai_expired_total = float(df["AI Credits Expired ($)"].sum())
    cloud_expired_total = float(df["Cloud Credits Expired ($)"].sum())
    ai_expiry_event = df.loc[df["AI Credits Expired ($)"] > 0, "Month"]
    cloud_expiry_event = df.loc[df["Cloud Credits Expired ($)"] > 0, "Month"]
    m1, m2, m3 = st.columns(3)
    ai_status = ai_exhausted.iloc[0] if len(ai_exhausted) else (f"Expired by {ai_expiry_event.iloc[0]}" if len(ai_expiry_event) else ("Balance remains" if df["Remaining AI Credits ($)"].iloc[-1] > 0 else "No starting balance"))
    cloud_status = cloud_exhausted.iloc[0] if len(cloud_exhausted) else (f"Expired by {cloud_expiry_event.iloc[0]}" if len(cloud_expiry_event) else ("Balance remains" if df["Remaining Cloud Credits ($)"].iloc[-1] > 0 else "No starting balance"))
    m1.metric("AI Credits Run Out / Expire", ai_status)
    m2.metric("Cloud Credits Run Out / Expire", cloud_status)
    horizon_outstanding = float(df["Outstanding Invoices at Horizon ($)"].iloc[-1])
    m3.metric("Outstanding Invoices at Horizon", f"${horizon_outstanding:,.0f}")
    if ai_expired_total or cloud_expired_total:
        st.warning(f"Unused credits expiring during forecast: AI ${ai_expired_total:,.0f}; cloud ${cloud_expired_total:,.0f}.")

with tab3:
    st.subheader("Monthly Roll-Forward Detail Table")
    st.dataframe(df.style.format({
        "DAU": "{:,.0f}", "Ending Pro Subs": "{:,.0f}", "Ending Ent Subs": "{:,.0f}",
        "Ending Paid Subscribers": "{:,.0f}", "Average Paid Subscribers": "{:,.0f}", "Monthly Prompts": "{:,.0f}",
        "Typical Pro Prompts": "{:,.0f}", "Heavy Pro Prompts": "{:,.0f}", "Enterprise Prompts": "{:,.0f}",
        "Implementation Revenue ($)": "${:,.0f}", "New Enterprise Deals": "{:,.0f}",
        "Ending MRR Run-Rate ($)": "${:,.0f}", "Recognized Revenue ($)": "${:,.0f}",
        "API Cost ($)": "${:,.0f}", "Vector DB Cost ($)": "${:,.0f}",
        "Fixed Infrastructure ($)": "${:,.0f}", "Payment Processing ($)": "${:,.0f}",
        "Total Modeled COGS ($)": "${:,.0f}", "Modeled Gross Profit ($)": "${:,.0f}",
        "Modeled Gross Margin (%)": "{:.1f}%", "Remaining AI Credits ($)": "${:,.0f}",
        "Remaining Cloud Credits ($)": "${:,.0f}", "AI Credits Expired ($)": "${:,.0f}",
        "Cloud Credits Expired ($)": "${:,.0f}", "AI Credits Applied ($)": "${:,.0f}",
        "Cloud Credits Applied ($)": "${:,.0f}", "Estimated Cash Payable ($)": "${:,.0f}",
        "Estimated Cash Paid This Month ($)": "${:,.0f}", "Outstanding Invoices at Horizon ($)": "${:,.0f}"
    }, na_rep="—"), use_container_width=True)

with tab4:
    st.subheader("📈 Multi-Scenario Comparative Analysis Matrix")
    st.markdown("Evaluating baseline assumptions against growth, usage, and price scenarios. Note: each scenario changes only the drivers shown in its name; the API price-reduction case does not change cache or batch settings.")

    def run_scenario(name, g_mult, prompt_mult, price_mult):
        scen_df = run_model_simulation(
            start_year, start_month, forecast_horizon, starting_dau, starting_pro_subs, starting_ent_subs,
            pro_monthly_growth * g_mult, round(net_new_ent_contracts * g_mult), paid_dau_factor, 
            new_user_conversion, free_base_conversion, ent_monthly_growth, monthly_churn,
            arpu_pro, arpu_ent, expansion_arpu_pct,
            baseline_prompts * prompt_mult, peak_prompts_1 * prompt_mult, peak_month_1, surge_width_1,
            peak_prompts_2 * prompt_mult, peak_month_2, surge_width_2, free_user_usage_mult,
            input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
            frontier_input_price * price_mult, frontier_output_price * price_mult,
            standard_input_price * price_mult, standard_output_price * price_mult,
            cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
            step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
            starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share,
            overhead_pct, payment_fee_pct, enterprise_contract_months, enterprise_implementation_fee,
            enterprise_prompts_per_account, heavy_user_share, heavy_user_multiplier,
            ai_credit_expiry_month, cloud_credit_expiry_month, invoice_lag_months
        )
        tot_rev = scen_df["Recognized Revenue ($)"].sum()
        tot_cogs = scen_df["Total Modeled COGS ($)"].sum()
        tot_gp = tot_rev - tot_cogs
        gm = (tot_gp / tot_rev * 100) if tot_rev > 0 else 0
        end_arr = scen_df.iloc[-1]["Ending MRR Run-Rate ($)"] * 12
        end_cash = scen_df.iloc[-1]["Estimated Cash Payable ($)"]
        return {
            "Scenario": name,
            "Total Revenue ($)": tot_rev,
            "Total COGS ($)": tot_cogs,
            "Gross Profit ($)": tot_gp,
            "Gross Margin (%)": gm,
            "Ending ARR ($)": end_arr,
            "Ending Monthly Cash Payable ($)": end_cash
        }

    scen_results = [
        run_scenario("Base Case", 1.0, 1.0, 1.0),
        run_scenario("High Growth (1.25x)", 1.25, 1.0, 1.0),
        run_scenario("Low Growth (0.75x)", 0.75, 1.0, 1.0),
        run_scenario("High Prompt Intensity (1.5x)", 1.0, 1.5, 1.0),
        run_scenario("Low Prompt Intensity (0.7x)", 1.0, 0.7, 1.0),
        run_scenario("API Price Inflation (+25%)", 1.0, 1.0, 1.25),
        run_scenario("API Price Reduction (-20%)", 1.0, 1.0, 0.8),
        run_scenario("Stress Test (High Usage + Inflation)", 1.1, 1.4, 1.2)
    ]
    scen_summary_df = pd.DataFrame(scen_results)
    scen_summary_df.insert(0, "ID", range(1, len(scen_summary_df) + 1))
    
    st.dataframe(scen_summary_df.style.format({
        "ID": "{:,.0f}",
        "Total Revenue ($)": "${:,.0f}", "Total COGS ($)": "${:,.0f}",
        "Gross Profit ($)": "${:,.0f}", "Gross Margin (%)": "{:.1f}%",
        "Ending ARR ($)": "${:,.0f}", "Ending Monthly Cash Payable ($)": "${:,.0f}"
    }), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("📑 Assumptions & Audit Register (Data Governance)")
    st.markdown("Input register for review. Source-system labels are suggested sources, not verified connections; replace estimates with actual company data where available.")

    audit_data = [
        ("Starting Daily Active Users (DAU)", f"{starting_dau:,.0f}", "Mixpanel / Segment Event Pipeline", "Management Baseline Estimate"),
        ("Starting Pro Subscribers ($30/mo)", f"{starting_pro_subs:,.0f}", "Stripe Billing Subscriptions Table", "Actual Active Pro Subscriptions"),
        ("Starting Enterprise Contracts ($250/mo)", f"{starting_ent_subs:,.0f}", "Stripe Billing Subscriptions Table", "Actual Active Enterprise Contracts"),
        ("Pro Tier Monthly DAU Growth", f"{pro_monthly_growth*100:.1f}% MoM", "Self-Serve PLG Growth Model", "Organic Web/App Growth Engine"),
        ("Net New Enterprise Deals / Month", f"{net_new_ent_contracts} deals/mo", "Sales Pipeline Sales Velocity", "Decoupled Sales-Led Pipeline Target"),
        ("Enterprise Account Expansion Rate", f"{ent_monthly_growth*100:.1f}% MoM", "Enterprise Net Retention Model", "Account Expansion & Up-Sells"),
        ("Paid DAU Activity Factor", f"{paid_dau_factor*100:.1f}%", "Mixpanel Daily Active Engagement Ratio", "Product Telemetry Benchmark"),
        ("New PLG User Conversion Rate", f"{new_user_conversion*100:.1f}%", "Stripe Checkout & Funnel Analytics", "PLG Self-Serve Conversion Rate"),
        ("Free-User Proxy Conversion Rate", f"{free_base_conversion*100:.1f}% MoM", "Product-Led Growth Funnel Report", "Estimated Conversion Benchmark"),
        ("Monthly Paid Subscriber Churn", f"{monthly_churn*100:.1f}% MoM", "Stripe Churn & Retention Analytics", "Historical Cohort Average"),
        ("Pro Tier Monthly ARPU", f"${arpu_pro:.2f} / mo", "Stripe Pricing Plan Definitions", "Standard Self-Serve Plan"),
        ("Enterprise Tier Monthly ARPU", f"${arpu_ent:.2f} / mo", "Stripe Contract Agreements", "Custom Enterprise Tier"),
        ("Expansion / Overage Revenue", f"{expansion_arpu_pct*100:.1f}% of ARPU", "Stripe Metered Billing Invoices", "Usage-Based Overage Proxy"),
        ("Payment Processing Fee", f"{payment_fee_pct*100:.1f}%", "Stripe Merchant Statement", "Standard Gateway Fee Schedule"),
        ("Baseline Prompts / User / Day", f"{baseline_prompts}", "Helicone / Langfuse API Telemetry", "Observed Mean Daily Prompt Volume"),
        ("Peak 1 Prompts / User / Day", f"{peak_prompts_1} (Month {peak_month_1})", "Capacity Planning Stress Model", "Spring Seasonal Peak Assumption"),
        ("Peak 2 Prompts / User / Day", f"{peak_prompts_2} (Month {peak_month_2})", "Capacity Planning Stress Model", "Fall Seasonal Peak Assumption"),
        ("Pareto Power-User Skew Share", f"{pareto_skew_share*100:.1f}%", "LLM Gateway Distribution Logs", "Top 20% User Concentration Factor"),
        ("Frontier Model Traffic Share", f"{frontier_mix*100:.1f}%", "LLM Gateway Router Logs", "Routing Policy Configuration"),
        ("Frontier Input Token Price", f"${frontier_input_price:.2f} / 1M", "OpenAI / Anthropic Developer Pricing", "Official Published List Price"),
        ("Frontier Output Token Price", f"${frontier_output_price:.2f} / 1M", "OpenAI / Anthropic Developer Pricing", "Official Published List Price"),
        ("Standard Input Token Price", f"${standard_input_price:.4f} / 1M", "OpenAI / Anthropic Developer Pricing", "Official Published List Price"),
        ("Standard Output Token Price", f"${standard_output_price:.4f} / 1M", "OpenAI / Anthropic Developer Pricing", "Official Published List Price"),
        ("Prompt Cache Hit Rate", f"{cache_hit_rate*100:.1f}%", "Helicone Caching Analytics", "Enterprise Caching Benchmark"),
        ("Fixed Base Infrastructure", f"${base_fixed_infra:,.0f} / mo", "AWS / GCP Cloud Billing Dashboard", "Core Server & Orchestration Hosting"),
        ("Vector DB Cost / DAU", f"${vector_cost_per_user:.2f} / DAU", "Pinecone / pgvector Usage Invoices", "Indexed Embedding Storage Rate"),
        ("AI Vendor Credits", f"${starting_ai_credits:,.0f}", "OpenAI / Anthropic Grant Agreements", "Unearned Promotional Credit Balance"),
        ("Enterprise Contract Term", f"{enterprise_contract_months} months", "Signed Enterprise Agreements", "New forecast deals expire after modeled term"),
        ("Enterprise Implementation Fee", f"${enterprise_implementation_fee:,.0f} / new deal", "Signed SOW / Implementation Schedule", "Recognized in modeled deal month; validate accounting policy"),
        ("Enterprise Prompts per Account", f"{enterprise_prompts_per_account:,.0f} / month", "Customer-level LLM telemetry", "Account-specific usage assumption"),
        ("Heavy User Share / Multiplier", f"{heavy_user_share*100:.0f}% / {heavy_user_multiplier:.1f}x", "LLM Gateway Distribution Logs", "Separate typical and heavy Pro cohorts"),
        ("AI Credit Expiry Month", f"{ai_credit_expiry_month or 'None in horizon'}", "Vendor Grant Terms", "Unused balance expires before the following forecast month"),
        ("Cloud Credit Expiry Month", f"{cloud_credit_expiry_month or 'None in horizon'}", "Cloud Grant Terms", "Unused balance expires before the following forecast month"),
        ("Invoice Payment Lag", f"{invoice_lag_months} months", "Vendor Payment Terms", "Cash paid shifted by modeled invoice lag")
    ]
    audit_df = pd.DataFrame(audit_data, columns=["Model Parameter", "Current Value", "Data Source / System of Record", "Provenance / Notes"])
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

with tab6:
    st.subheader("🎯 Value-Creation Anchor & Hybrid Pricing Strategy")
    st.markdown("Estimate customer value and compare pricing archetypes. Values in the archetype table are illustrative until each pricing plan is parameterized with its own price, allowance, overage, and usage assumptions.")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 1. Value-Creation Anchor Calculator")
        value_metric_type = st.selectbox("Primary AI Value Metric", ["Automation (Hours Saved)", "Augmentation (Error Reduction)", "Connection (Network Access)", "Revenue (Tasks / Leads Generated)"])
        
        if value_metric_type == "Automation (Hours Saved)":
            hours_saved = st.number_input("Hours Saved / User / Month", min_value=0.0, value=10.0, step=1.0)
            hourly_wage = st.number_input("Average User Hourly Wage ($)", min_value=0.0, value=50.0, step=5.0)
            economic_value = hours_saved * hourly_wage
            st.info(f"💡 Total Economic Value Created: **${economic_value:,.0f} / user / month**")
        elif value_metric_type == "Augmentation (Error Reduction)":
            errors_avoided = st.number_input("Critical Errors Avoided / Month", min_value=0.0, value=2.0, step=1.0)
            cost_per_error = st.number_input("Cost per Error ($)", min_value=0.0, value=250.0, step=25.0)
            economic_value = errors_avoided * cost_per_error
            st.info(f"💡 Total Economic Value Created: **${economic_value:,.0f} / user / month**")
        elif value_metric_type == "Connection (Network Access)":
            economic_value = st.number_input("Estimated Network/Data Value ($ / mo)", min_value=0.0, value=200.0, step=25.0)
            st.info(f"💡 Total Economic Value Created: **${economic_value:,.0f} / user / month**")
        else:
            tasks_generated = st.number_input("Qualified Leads / Tasks Completed / Month", min_value=0.0, value=5.0, step=1.0)
            value_per_task = st.number_input("Value per Task / Lead ($)", min_value=0.0, value=100.0, step=10.0)
            economic_value = tasks_generated * value_per_task
            st.info(f"💡 Total Economic Value Created: **${economic_value:,.0f} / user / month**")

        capture_ratio = (arpu_pro / economic_value * 100) if economic_value > 0 else 0
        roi_multiple = (economic_value / arpu_pro) if arpu_pro > 0 else 0
        st.markdown(f"""
        * **Pro Tier Price Charged:** `${arpu_pro:.2f} / mo`
        * **Value Capture Ratio:** `{capture_ratio:.1f}%` of total value created
        * **Customer ROI Multiple:** `{roi_multiple:.1f}x` (Customer gets ${roi_multiple:.1f} of value for every $1 paid)
        """)

    with col_v2:
        st.markdown("#### 2. Quantitative Pricing Archetype Comparison Engine")
        st.markdown("Illustrative archetype comparison only. Current archetype revenues use simple placeholders, not equivalent customer-level plan simulations; do not use this table alone to choose final prices.")

        # Compare pricing only against modeled paid-user and enterprise usage;
        # free-tier prompts remain a cost burden rather than being silently billed.
        total_forecast_prompts = float((df["Typical Pro Prompts"] + df["Heavy Pro Prompts"] + df["Enterprise Prompts"]).sum())
        paid_subscriber_months = float(df["Average Paid Subscribers"].sum())
        blended_cogs = float(df["Total Modeled COGS ($)"].sum())
        st.caption("All four structures use the same paid-user and enterprise prompt volume, paid subscriber-months, and modeled COGS. Free-tier prompts are not billed in these comparisons but their costs remain in company COGS. Revenue differs only by the editable billing rules below. This is a portfolio-level comparison, not customer-specific contract advice.")
        with st.expander("Edit comparable pricing rules", expanded=True):
            payg_rate = st.number_input("Pay-as-you-go price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f")
            sub_fee = st.number_input("Subscription + allowance fee / paid subscriber / month ($)", min_value=0.0, value=30.0, step=5.0)
            sub_allowance = st.number_input("Subscription included prompts / subscriber / month", min_value=0, value=1500, step=250)
            sub_overage = st.number_input("Subscription overage price per prompt ($)", min_value=0.0, value=0.006, step=0.001, format="%.4f")
            hybrid_fee = st.number_input("Hybrid base fee / paid subscriber / month ($)", min_value=0.0, value=20.0, step=5.0)
            hybrid_allowance = st.number_input("Hybrid included prompts / subscriber / month", min_value=0, value=1000, step=250)
            hybrid_overage = st.number_input("Hybrid overage price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f")
            prepaid_credit_price = st.number_input("Prepaid price per credit ($)", min_value=0.0, value=1.0, step=0.25)
            prepaid_prompts_per_credit = st.number_input("Prompts covered per credit", min_value=0.01, value=100.0, step=10.0)

        payg_rev = total_forecast_prompts * payg_rate
        sub_included = paid_subscriber_months * sub_allowance
        sub_rev = paid_subscriber_months * sub_fee + max(0.0, total_forecast_prompts - sub_included) * sub_overage
        hybrid_included = paid_subscriber_months * hybrid_allowance
        hybrid_rev = paid_subscriber_months * hybrid_fee + max(0.0, total_forecast_prompts - hybrid_included) * hybrid_overage
        prepaid_rev = (total_forecast_prompts / prepaid_prompts_per_credit) * prepaid_credit_price if prepaid_prompts_per_credit > 0 else 0.0

        archetype_rows = []
        for label, revenue, predictability, risk in [
            ("1. Pay-as-you-go", payg_rev, "Low (usage-driven)", "Low commitment; revenue volatility"),
            ("2. Subscription + allowance", sub_rev, "High (base fee)", "Allowance/overage boundary friction"),
            ("3. Hybrid base + overage", hybrid_rev, "High (base + usage)", "Requires clear meter and overage UX"),
            ("4. Prepaid credit burndown", prepaid_rev, "Upfront cash; recognition varies", "Breakage, expiry and replenishment risk"),
        ]:
            gp = revenue - blended_cogs
            gm = gp / revenue * 100 if revenue > 0 else np.nan
            archetype_rows.append({"Pricing Archetype": label, "Simulated Revenue ($)": revenue, "Same Modeled COGS ($)": blended_cogs,
                                   "Gross Profit ($)": gp, "Gross Margin (%)": gm, "Revenue Predictability": predictability, "Key Risk": risk})
        archetype_df = pd.DataFrame(archetype_rows)
        st.dataframe(archetype_df.style.format({"Simulated Revenue ($)": "${:,.0f}", "Same Modeled COGS ($)": "${:,.0f}",
                                                  "Gross Profit ($)": "${:,.0f}", "Gross Margin (%)": "{:.1f}%"}, na_rep="—"), use_container_width=True, hide_index=True)

        st.markdown("#### 3. Overage Margin Guardrail Check")
        # Price per prompt needed to cover the modeled blended API cost at the selected GM target.
        suggested_overage_rate = (blended_token_cost_m1 / (1 - margin_target / 100)) if margin_target < 100 else 0.0
        st.info(f"**Indicative API-only overage floor:** ${suggested_overage_rate:.5f} per prompt at a {margin_target}% gross-margin target. This excludes non-API costs and is not a per-1K-token rate; set the customer-facing unit to match your actual billing meter.")

st.markdown("---")
st.caption("© 2026. Institutional AI Deal Desk & Contract Pricing Simulator. All rights reserved.")

# ==========================================
# CHANGE LOG — 2026-10-11
# ==========================================
st.markdown("---")
with st.expander("📝 Change log — fixes and enhancements", expanded=False):
    st.markdown("""
    **Logic and labeling corrections**
    - Replaced the hardcoded 60% reverse-margin assumption with the selected gross-margin target.
    - Corrected the usage-ceiling calculation to solve for prompts from the remaining API-cost allowance after allocating modeled non-API costs per ending paid subscriber. The result is explicitly labeled an estimate.
    - Renamed “Full-Year Gross Margin” to “Forecast-Period Gross Margin” because the forecast horizon can be shorter or longer than 12 months.
    - Replaced the hardcoded 500-prompts recommendation with the model-derived estimated usage ceiling and a warning to validate before enforcement.
    - Corrected the overage guardrail label and formula: it is now an indicative API-only price per prompt at the selected margin target, not a per-1K-token rate.
    - Renamed the “Aggressive Caching & Batch” scenario to “API Price Reduction (-20%)” because that scenario changes API prices rather than caching/batch settings.
    - The prior revision flagged placeholder pricing revenues; this revision replaces those placeholders with formula-driven comparisons using shared usage and cost assumptions.

    **Founder-facing UX enhancements**
    - Added sanity warnings when starting paid subscribers exceed DAU or when paid subscribers exist alongside zero DAU.
    - Added an ending-month unit-economics snapshot for blended COGS, revenue, and gross profit per paid subscriber.
    - Added explanatory captions for promotional-credit cash payable and the limitations of blended subscriber averages.
    - Made Advanced Deal Desk Parameters off by default so first-time founders see the simpler inputs first; advanced controls remain available with one toggle.
    - Clarified that audit-register source names are suggested sources, not live verified data connections.
    - Clarified that the Pareto skew percentage is currently an assumption display and does not alter the cost engine.

    **Retained intentionally**
    - All six existing tabs, scenario rows, value-creation calculator, audit register, charts, inputs, and pricing archetype table remain in place. No existing feature or section was deleted.

    **New in this revision — 2026-10-11**
    - Added Pre-launch, Early traction, and Scaling onboarding presets. Presets only set starting values; inputs remain editable.
    - Added enterprise contract term, per-new-deal implementation fees, and enterprise prompts per account per month. Newly modeled deal cohorts expire after the selected term; starting enterprise contracts are treated as existing/renewed accounts.
    - Split Pro paid-user prompts into typical and heavy-user cohorts using editable heavy-user share and usage multiplier. Enterprise usage is modeled separately from Pro and free usage.
    - Added grant expiry month assumptions and monthly tracking of expired AI/cloud credits, plus a visible credit-exhaustion month indicator.
    - Added invoice payment lag and separated accrued payable from estimated cash paid; invoices unpaid at the forecast horizon are surfaced.
    - Rebuilt the four pricing archetypes to use the same modeled prompt volume, paid subscriber-months, and COGS, with editable price, allowance, and overage assumptions.
    - Expanded monthly detail and audit register to expose the new contract, cohort, credit, and cash timing drivers.

    **Known limitations to keep visible**
    - Enterprise contract cohorts use an aggregate account count and average monthly contract ARPU; they do not yet track named customers, individual contract start/end dates, renewals, ramp schedules, discounts, or per-account margin.
    - Implementation fees are modeled as recognized in the new-deal month for planning purposes; accounting recognition may differ by contract and performance obligation.
    - Prepaid-credit pricing simulation assumes all forecast prompts are billable and uses no breakage, expiry, refund, or deferred-revenue schedule; it is a comparable gross-revenue proxy, not cash/GAAP revenue accounting.
    - The pricing comparison is portfolio-level; free users are included in forecast prompts while pricing revenue is calculated using the chosen portfolio pricing rules. Edit assumptions carefully and validate against the actual billable meter.
    - Credit expiry is modeled at month granularity, and invoice lag is modeled in whole months; exact day-level expiry, invoice dates, and cash timing are not represented.
    - The usage ceiling remains a blended estimate and should not be used as a contractual customer-specific limit.
    """)
