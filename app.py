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

/* CFO Commentary Box */
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
</style>
""", unsafe_allow_html=True)

# Compact Header Banner
st.markdown("""
<div class="compact-header">
    <h1>⚡ AI Deal Desk & Unit Economics Simulator</h1>
    <p>Proactive contract pricing, split Pro/Enterprise growth, value metrics, hybrid models, and credit depletion.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. SIDEBAR ASSUMPTIONS & UX GOVERNANCE
# ==========================================
st.sidebar.header("1. Rolling Forecast Horizon")
current_now = datetime.datetime.now()
start_year = st.sidebar.number_input("Forecast Start Year", min_value=2024, max_value=2035, value=current_now.year)
start_month = st.sidebar.slider("Forecast Start Month", 1, 12, current_now.month)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 36, 12)

# UX Toggle: Simple vs. Advanced Mode
st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("🛠️ Enable Advanced Deal Desk Parameters", value=True, help="Toggle on to access full granular model assumptions. Toggle off for simplified executive overview.")

# --- CORE / DEFAULT INPUTS (Always Visible) ---
st.sidebar.header("2. Core Growth & Funnel")
starting_dau = st.sidebar.number_input("Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500)
starting_pro_subs = st.sidebar.number_input("Starting Pro Subscribers ($30/mo)", min_value=0, value=200, step=50)
starting_ent_subs = st.sidebar.number_input("Starting Enterprise Subscribers ($250/mo)", min_value=0, value=50, step=10)
starting_paid_subs = starting_pro_subs + starting_ent_subs

# Split Growth Inputs for Pro vs Enterprise
pro_monthly_growth = st.sidebar.slider("Pro Tier Monthly Growth (%)", 0.0, 25.0, 8.0, 0.5) / 100
ent_monthly_growth = st.sidebar.slider("Enterprise Tier Monthly Growth (%)", 0.0, 25.0, 5.0, 0.5) / 100
baseline_prompts = st.sidebar.slider("Baseline Prompts / Paid User / Day", 1, 20, 5)

# --- ADVANCED INPUTS (Inside Collapsible Expanders if Advanced Mode is True) ---
if advanced_mode:
    with st.sidebar.expander("📂 1. Subscriber Funnel & Conversion Levers"):
        paid_dau_factor = st.slider("Paid Subscribers Active Daily (%)", 0, 100, 70, 5) / 100
        new_user_conversion = st.slider("New User Conversion (%)", 0.0, 50.0, 15.0, 0.5) / 100
        enterprise_acquisition_share = st.slider("Enterprise Share of New Conversions (%)", 0.0, 50.0, 20.0, 1.0) / 100
        free_base_conversion = st.slider("Free-User Proxy Conversion (%)", 0.0, 10.0, 2.0, 0.1) / 100
        monthly_churn = st.slider("Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5) / 100
        arpu_pro = st.number_input("Pro Tier Monthly ARPU ($)", min_value=0.0, value=30.0, step=5.0)
        arpu_ent = st.number_input("Enterprise Tier Monthly ARPU ($)", min_value=0.0, value=250.0, step=25.0)
        expansion_arpu_pct = st.slider("Expansion / Overage Revenue (% of ARPU)", 0.0, 50.0, 10.0, 1.0) / 100
        payment_fee_pct = st.slider("Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1) / 100
    
    with st.sidebar.expander("📂 2. Multi-Peak Usage & Pareto Skew"):
        st.markdown("**Peak 1 (Spring / Launch)**")
        peak_prompts_1 = st.slider("Peak 1 Volume", 1, 100, 35)
        peak_month_1 = st.slider("Peak 1 Month", 1, forecast_horizon, min(3, forecast_horizon))
        surge_width_1 = st.slider("Peak 1 Spread ($\sigma$)", 0.5, 4.0, 1.5, 0.5)
        
        st.markdown("**Peak 2 (Fall / Holiday)**")
        peak_prompts_2 = st.slider("Peak 2 Volume", 1, 100, 45)
        peak_month_2 = st.slider("Peak 2 Month", 1, forecast_horizon, min(11, forecast_horizon))
        surge_width_2 = st.slider("Peak 2 Spread ($\sigma$)", 0.5, 4.0, 2.0, 0.5)
        
        pareto_skew_share = st.slider("Power-User Token Skew Share (%)", 50, 95, 80, 5) / 100
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
        base_fixed_infra = st.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=2500.0, step=500.0)
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
        starting_ai_credits = st.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=20000.0, step=2500.0)
        starting_cloud_credits = st.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=30000.0, step=5000.0)
        ai_credit_eligible_share = st.slider("AI Credit Eligible Share (%)", 0, 100, 100) / 100
        cloud_credit_eligible_share = st.slider("Cloud Credit Eligible Share (%)", 0, 100, 100) / 100
        margin_target = st.slider("Target Gross Margin (%)", 0, 90, 60)
        budget_monthly = st.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)
else:
    paid_dau_factor = 0.70
    new_user_conversion = 0.15
    enterprise_acquisition_share = 0.20
    free_base_conversion = 0.02
    monthly_churn = 0.03
    arpu_pro = 30.0
    arpu_ent = 250.0
    expansion_arpu_pct = 0.10
    payment_fee_pct = 0.029
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
    base_fixed_infra = 2500.0
    step_trigger_type = "DAU Threshold"
    raw_step_threshold = 20000.0
    step_cost_increment = 1500.0
    vector_cost_per_user = 0.15
    starting_ai_credits = 20000.0
    starting_cloud_credits = 30000.0
    ai_credit_eligible_share = 1.0
    cloud_credit_eligible_share = 1.0
    margin_target = 60
    budget_monthly = 0.0


# ==========================================
# 2. CORE ENGINE SIMULATION FUNCTION
# ==========================================
@st.cache_data(show_spinner=False)
def run_model_simulation(
    p_start_year, p_start_month, p_horizon, p_dau, p_pro_subs, p_ent_subs, p_pro_growth, p_ent_growth, p_paid_dau_factor,
    p_new_conv, p_ent_acq_share, p_base_conv, p_churn, p_arpu_pro, p_arpu_ent, p_expansion_pct,
    p_base_prompts, p_p1_vol, p_p1_m, p_s1_w, p_p2_vol, p_p2_m, p_s2_w, p_free_mult,
    p_in_tok, p_out_tok, p_in_grow, p_out_grow, p_front_mix,
    p_f_in_p, p_f_out_p, p_s_in_p, p_s_out_p, p_cache_rate, p_cache_ratio,
    p_batch_share, p_batch_price_ratio, p_base_infra, p_step_type, p_step_thresh_raw,
    p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred, p_ai_elig, p_cloud_elig,
    p_overhead, p_fee_pct
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

    current_dau = float(p_dau)
    prev_pro_subs = float(p_pro_subs)
    prev_ent_subs = float(p_ent_subs)

    sim_rows = []

    for i in range(n):
        beg_pro = prev_pro_subs
        beg_ent = prev_ent_subs
        beg_paid_subs = beg_pro + beg_ent

        if i == 0:
            dau_val = current_dau
            new_users_acquired = 0.0
            free_base = max(0.0, dau_val - beg_paid_subs)
            prev_dau_val = current_dau
        else:
            prev_dau_val = current_dau
            current_dau = prev_dau_val * (1 + ((p_pro_growth + p_ent_growth) / 2))
            dau_val = current_dau
            new_users_acquired = dau_val - prev_dau_val
            free_base = max(0.0, prev_dau_val - beg_paid_subs)

        total_new_conversions = (new_users_acquired * p_new_conv) + (free_base * p_base_conv)
        ent_share = p_ent_acq_share
        pro_share = 1.0 - ent_share

        new_pro_conv = total_new_conversions * pro_share
        new_ent_conv = total_new_conversions * ent_share

        churned_pro = beg_pro * p_churn
        churned_ent = beg_ent * (p_churn * 0.5)

        ending_pro = max(0.0, (beg_pro * (1 + p_pro_growth)) + new_pro_conv - churned_pro)
        ending_ent = max(0.0, (beg_ent * (1 + p_ent_growth)) + new_ent_conv - churned_ent)
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

        step_multiplier = max(0, math.ceil(trigger_metric / p_step_thresh_raw) - 1) if p_step_thresh_raw > 0 and trigger_metric > p_step_thresh_raw else 0
        infrastructure_cost = p_base_infra + (step_multiplier * p_step_cost)

        base_rev_pro = avg_pro * p_arpu_pro
        base_rev_ent = avg_ent * p_arpu_ent
        base_revenue = base_rev_pro + base_rev_ent
        expansion_revenue = base_revenue * p_expansion_pct
        recognized_revenue = base_revenue + expansion_revenue
        ending_mrr_runrate = (ending_pro * p_arpu_pro + ending_ent * p_arpu_ent) * (1 + p_expansion_pct)
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

        sim_rows.append({
            "Month": month_names[i],
            "DAU": round(dau_val),
            "Ending Pro Subs": round(ending_pro),
            "Ending Ent Subs": round(ending_ent),
            "Ending Paid Subscribers": round(ending_paid_subs),
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
            "Estimated Cash Payable ($)": estimated_cash_payable,
        })

    return pd.DataFrame(sim_rows)

try:
    df = run_model_simulation(
        start_year, start_month, forecast_horizon, starting_dau, starting_pro_subs, starting_ent_subs, 
        pro_monthly_growth, ent_monthly_growth, paid_dau_factor,
        new_user_conversion, enterprise_acquisition_share, free_base_conversion, monthly_churn, arpu_pro, arpu_ent, expansion_arpu_pct,
        baseline_prompts, peak_prompts_1, peak_month_1, surge_width_1, peak_prompts_2, peak_month_2, surge_width_2, free_user_usage_mult,
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

blended_token_cost_m1 = (df.loc[0, "API Cost ($)"] / df.loc[0, "Monthly Prompts"]) if df.loc[0, "Monthly Prompts"] > 0 else 0.00001
pro_arpu_total = arpu_pro * (1 + expansion_arpu_pct)
allowable_cogs_60pct = pro_arpu_total * (1 - 0.60)
max_prompts_60 = (allowable_cogs_60pct / (blended_token_cost_m1 * 1000)) * 1000 if blended_token_cost_m1 > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{forecast_horizon})", f"${last_arr:,.0f} ARR", f"Recognized Rev: ${annual_rev:,.0f}")
c2.metric("Full-Year Gross Margin", f"{annual_gm:.1f}%", f"Target: {margin_target}%")
c3.metric("Reverse-Margin Limit (60%)", f"{max_prompts_60:,.0f} Prompts/Mo", "Max Pro Tier Usage Ceiling")
c4.metric("Pareto Skew Concentration", f"{pareto_skew_share*100:.0f}% Top 20%", "Heavy User Load Factor")

policy_recs = []
if last_margin < margin_target:
    policy_recs.append(f"<b>Deal Desk Advisory:</b> Pro tier margin ({last_margin:.1f}%) breaches target ({margin_target}%). <b>Action:</b> Implement automated prompt rerouting to Standard models for power users exceeding 500 prompts/month.")
else:
    policy_recs.append(f"<b>Deal Desk Status:</b> Gross margin remains robust at {last_margin:.1f}%, comfortably meeting the {margin_target}% hurdle rate.")

if m_last["Remaining AI Credits ($)"] <= 0:
    policy_recs.append("<b>Credit Depletion Warning:</b> AI vendor credits are fully exhausted before horizon end. Out-of-pocket cash payable for API COGS escalates immediately.")
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

    st.markdown("#### COGS Component Disaggregation")
    cogs_fig = go.Figure()
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["API Cost ($)"], name="API Inference Cost", marker_color="#4299e1"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Vector DB Cost ($)"], name="Vector DB Cost", marker_color="#4fd1c5"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Fixed Infrastructure ($)"], name="Infrastructure Step-Cost", marker_color="#ed8936"))
    cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Payment Processing ($)"], name="Payment Processing Fees", marker_color="#9f7aea"))
    cogs_fig.update_layout(template="plotly_white", barmode="stack", height=320, margin=dict(l=30, r=30, t=20, b=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(cogs_fig, use_container_width=True)

with tab2:
    st.subheader("Credit Drawdown Velocity & Cash Payable")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Out-of-Pocket Cash Payable ($)", marker_color="#dd6b20"))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color="#38a169", width=2)))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Cloud Credits ($)"], name="Remaining Cloud Credits ($)", mode="lines", line=dict(color="#3182ce", width=2, dash="dot")))
    fig2.update_layout(template="plotly_white", height=340, barmode="stack", margin=dict(l=30, r=30, t=30, b=20), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Monthly Roll-Forward Detail Table")
    st.dataframe(df.style.format({
        "DAU": "{:,.0f}", "Ending Pro Subs": "{:,.0f}", "Ending Ent Subs": "{:,.0f}",
        "Ending Paid Subscribers": "{:,.0f}", "Monthly Prompts": "{:,.0f}",
        "Ending MRR Run-Rate ($)": "${:,.0f}", "Recognized Revenue ($)": "${:,.0f}",
        "API Cost ($)": "${:,.0f}", "Vector DB Cost ($)": "${:,.0f}",
        "Fixed Infrastructure ($)": "${:,.0f}", "Payment Processing ($)": "${:,.0f}",
        "Total Modeled COGS ($)": "${:,.0f}", "Modeled Gross Profit ($)": "${:,.0f}",
        "Modeled Gross Margin (%)": "{:.1f}%", "Remaining AI Credits ($)": "${:,.0f}",
        "Remaining Cloud Credits ($)": "${:,.0f}", "Estimated Cash Payable ($)": "${:,.0f}"
    }, na_rep="—"), use_container_width=True)

with tab4:
    st.subheader("📈 Multi-Scenario Comparative Analysis Matrix")
    st.markdown("Evaluating baseline assumptions against bull, bear, high-usage, and cost-shock scenarios.")

    def run_scenario(name, g_mult, prompt_mult, price_mult):
        scen_df = run_model_simulation(
            start_year, start_month, forecast_horizon, starting_dau, starting_pro_subs, starting_ent_subs,
            pro_monthly_growth * g_mult, ent_monthly_growth * g_mult, paid_dau_factor, 
            new_user_conversion, enterprise_acquisition_share, free_base_conversion, monthly_churn,
            arpu_pro, arpu_ent, expansion_arpu_pct,
            baseline_prompts * prompt_mult, peak_prompts_1 * prompt_mult, peak_month_1, surge_width_1,
            peak_prompts_2 * prompt_mult, peak_month_2, surge_width_2, free_user_usage_mult,
            input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
            frontier_input_price * price_mult, frontier_output_price * price_mult,
            standard_input_price * price_mult, standard_output_price * price_mult,
            cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
            step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
            starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share,
            overhead_pct, payment_fee_pct
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
        run_scenario("Aggressive Caching & Batch", 1.0, 1.0, 0.8),
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
    st.markdown("Complete audit trail documenting the provenance and governance of all model inputs.")

    audit_data = [
        ("Starting Daily Active Users (DAU)", f"{starting_dau:,.0f}", "Mixpanel / Segment Event Pipeline", "Management Baseline Estimate"),
        ("Starting Pro Subscribers ($30/mo)", f"{starting_pro_subs:,.0f}", "Stripe Billing Subscriptions Table", "Actual Active Pro Subscriptions"),
        ("Starting Enterprise Subscribers ($250/mo)", f"{starting_ent_subs:,.0f}", "Stripe Billing Subscriptions Table", "Actual Active Enterprise Contracts"),
        ("Pro Tier Monthly Growth Rate", f"{pro_monthly_growth*100:.1f}% MoM", "Self-Serve PLG Growth Model", "Management Pro Growth Target"),
        ("Enterprise Tier Monthly Growth Rate", f"{ent_monthly_growth*100:.1f}% MoM", "Enterprise Sales Pipeline Forecast", "Management Enterprise Growth Target"),
        ("Paid DAU Activity Factor", f"{paid_dau_factor*100:.1f}%", "Mixpanel Daily Active Engagement Ratio", "Product Telemetry Benchmark"),
        ("New User Conversion Rate", f"{new_user_conversion*100:.1f}%", "Stripe Checkout & Funnel Analytics", "Illustrative Baseline — Source Required (applies M2+)"),
        ("Enterprise Share of New Conversions", f"{enterprise_acquisition_share*100:.1f}%", "Deal Desk Conversion Mix", "Sales vs Self-Serve Ratio"),
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
        ("AI Vendor Credits", f"${starting_ai_credits:,.0f}", "OpenAI / Anthropic Grant Agreements", "Unearned Promotional Credit Balance")
    ]
    audit_df = pd.DataFrame(audit_data, columns=["Model Parameter", "Current Value", "Data Source / System of Record", "Provenance / Notes"])
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

with tab6:
    st.subheader("🎯 Value-Creation Anchor & Hybrid Pricing Strategy")
    st.markdown("Align your pricing against economic value created (Automation, Augmentation, Connection, Revenue) and evaluate hybrid billing models.")

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
        st.markdown("Simulating annual revenue, gross margin, and churn risk across standard Metronome/Stripe billing architectures:")

        # Dynamic calculation based on total forecast prompts & costs
        total_forecast_prompts = df["Monthly Prompts"].sum()
        blended_cogs = annual_cogs

        # 1. Pay-as-you-go: Charge per 1k prompts ($0.008/prompt proxy)
        payg_rev = total_forecast_prompts * 0.008
        payg_gm = ((payg_rev - blended_cogs) / payg_rev * 100) if payg_rev > 0 else 0

        # 2. Sub + Allowance (Flat ARPU, capped)
        sub_allow_rev = annual_rev * 0.90
        sub_allow_gm = ((sub_allow_rev - blended_cogs) / sub_allow_rev * 100) if sub_allow_rev > 0 else 0

        # 3. Hybrid (Sub + Overage)
        hybrid_rev = annual_rev
        hybrid_gm = annual_gm

        # 4. Credit Burndown / Replenishing (Prepaid packages)
        credit_rev = annual_rev * 1.05
        credit_gm = ((credit_rev - blended_cogs) / credit_rev * 100) if credit_rev > 0 else 0

        archetype_data = [
            {"Pricing Archetype": "1. Pay-as-you-go", "Est. Full-Year Revenue": f"${payg_rev:,.0f}", "Gross Margin (%)": f"{payg_gm:.1f}%", "Revenue Predictability": "Low (Usage Dependent)", "Churn / Barrier Risk": "Very Low (No Commitment)"},
            {"Pricing Archetype": "2. Sub + Usage Allowance", "Est. Full-Year Revenue": f"${sub_allow_rev:,.0f}", "Gross Margin (%)": f"{sub_allow_gm:.1f}%", "Revenue Predictability": "High (Fixed MRR)", "Churn / Barrier Risk": "High (Hard Cap Friction)"},
            {"Pricing Archetype": "3. Sub + Overage (Hybrid)", "Est. Full-Year Revenue": f"${hybrid_rev:,.0f}", "Gross Margin (%)": f"{hybrid_gm:.1f}%", "Revenue Predictability": "Very High (Base + Expansion)", "Churn / Barrier Risk": "Low (Seamless Scaling)"},
            {"Pricing Archetype": "4. Credit Burndown / Replenish", "Est. Full-Year Revenue": f"${credit_rev:,.0f}", "Gross Margin (%)": f"{credit_gm:.1f}%", "Revenue Predictability": "High (Prepaid Cash Upfront)", "Churn / Barrier Risk": "Moderate (Prepaid Friction)"}
        ]
        
        archetype_df = pd.DataFrame(archetype_data)
        st.dataframe(archetype_df, use_container_width=True, hide_index=True)

        st.markdown("#### 3. Overage Margin Guardrail Check")
        suggested_overage_rate = blended_token_cost_m1 * 1.50
        st.success(f"🛡️ **Guardrail Recommendation:** To maintain your target {margin_target}% gross margin during peak usage surges, suggested minimum overage rate is **${suggested_overage_rate:.5f} per 1K tokens**.")

st.markdown("---")
st.caption("© 2026. Institutional AI Deal Desk & Contract Pricing Simulator. All rights reserved.")
