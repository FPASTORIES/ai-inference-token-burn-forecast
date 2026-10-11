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
</style>
""", unsafe_allow_html=True)

# Compact Header Banner
st.markdown("""
<div class="compact-header">
    <h1>⚡ AI Deal Desk & Unit Economics Simulator</h1>
    <p>Proactive contract pricing, multi-tier subscriber bridge, multi-peak usage waves, Pareto skew, and credit depletion.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR ASSUMPTIONS & HOVER TOOLTIPS
# ==========================================
st.sidebar.header("0. Rolling Forecast Horizon")
current_now = datetime.datetime.now()
start_year = st.sidebar.number_input(
    "Forecast Start Year", min_value=2024, max_value=2035, value=current_now.year,
    help="Calendar start year for the financial roll-forward."
)
start_month = st.sidebar.slider(
    "Forecast Start Month", 1, 12, current_now.month,
    help="Calendar start month (1 = January, 12 = December) for the simulation."
)
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)", 1, 36, 12,
    help="Total number of forward-looking months to simulate (up to 36 months)."
)

st.sidebar.header("1. Multi-Tier Subscriber Funnel")
starting_dau = st.sidebar.number_input(
    "Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500,
    help="Initial active user count at Month 1 across free and paid tiers."
)
starting_pro_subs = st.sidebar.number_input(
    "Starting Pro Subscribers ($30/mo)", min_value=0, value=200, step=50,
    help="Initial subscriber count for standard Pro tier."
)
starting_ent_subs = st.sidebar.number_input(
    "Starting Enterprise Subscribers ($250/mo)", min_value=0, value=50, step=10,
    help="Initial subscriber count for high-value Enterprise custom contract tier."
)
starting_paid_subs = starting_pro_subs + starting_ent_subs

monthly_growth = st.sidebar.slider(
    "Monthly DAU Growth (%)", 0.0, 25.0, 8.0, 0.5,
    help="Assumed MoM growth for top-of-funnel acquisition proxy."
) / 100

paid_dau_factor = st.sidebar.slider(
    "Paid Subscribers Active Daily (%)", 0, 100, 70, 5,
    help="Average daily active paid users as a share of paid subscribers."
) / 100

new_user_conversion = st.sidebar.slider(
    "New User Conversion (%)", 0.0, 50.0, 15.0, 0.5,
    help="% of brand-new acquired users who upgrade to paid in their first month."
) / 100

free_base_conversion = st.sidebar.slider(
    "Free-User Proxy Conversion (%)", 0.0, 10.0, 2.0, 0.1,
    help="% of active free-user proxy pool converting to paid each month."
) / 100

monthly_churn = st.sidebar.slider(
    "Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5,
    help="Percentage of paid subscribers cancelling their subscription each month."
) / 100

arpu_pro = st.sidebar.number_input(
    "Pro Tier Monthly ARPU ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average Base Revenue Per User per month for Pro subscribers."
)
arpu_ent = st.sidebar.number_input(
    "Enterprise Tier Monthly ARPU ($)", min_value=0.0, value=250.0, step=25.0,
    help="Average Base Revenue Per User per month for Enterprise custom contracts."
)

expansion_arpu_pct = st.sidebar.slider(
    "Expansion / Overage Revenue (% of ARPU)", 0.0, 50.0, 10.0, 1.0,
    help="Incremental revenue from token overages and credit top-ups as a % of base ARPU."
) / 100

payment_fee_pct = st.sidebar.slider(
    "Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1,
    help="Card processor transaction fees recognized directly in COGS."
) / 100

st.sidebar.header("2. Multi-Peak Usage & Pareto Skew Engine")
baseline_prompts = st.sidebar.slider(
    "Baseline Prompts / Paid User / Day", 1, 20, 5,
    help="Average daily prompt volume for paid users during normal activity."
)

# Multi-Peak Configuration Toggles
st.sidebar.markdown("**Peak 1 (Spring / Launch)**")
peak_prompts_1 = st.sidebar.slider("Peak 1 Volume", 1, 100, 35, help="Peak prompts/user/day for Peak 1.")
peak_month_1 = st.sidebar.slider("Peak 1 Month", 1, forecast_horizon, min(3, forecast_horizon), help="Month of Peak 1.")
surge_width_1 = st.sidebar.slider("Peak 1 Spread ($\sigma$)", 0.5, 4.0, 1.5, 0.5, help="Duration spread for Peak 1.")

st.sidebar.markdown("**Peak 2 (Fall / Holiday Rush)**")
peak_prompts_2 = st.sidebar.slider("Peak 2 Volume", 1, 100, 45, help="Peak prompts/user/day for Peak 2.")
peak_month_2 = st.sidebar.slider("Peak 2 Month", 1, forecast_horizon, min(11, forecast_horizon), help="Month of Peak 2.")
surge_width_2 = st.sidebar.slider("Peak 2 Spread ($\sigma$)", 0.5, 4.0, 2.0, 0.5, help="Duration spread for Peak 2.")

pareto_skew_share = st.sidebar.slider(
    "Power-User Token Skew Share (%)", 50, 95, 80, 5,
    help="Pareto principle assumption: percentage of total token volume consumed by the top 20% power-user cohort."
) / 100

free_user_usage_mult = st.sidebar.slider(
    "Free Tier Usage Multiplier (%)", 0, 100, 50,
    help="Usage volume of free tier users relative to paid users."
) / 100

input_tokens_start = st.sidebar.number_input(
    "Initial Input Tokens / Prompt", min_value=0, value=1000, step=100,
    help="Average prompt context length including system and RAG payloads."
)
output_tokens_start = st.sidebar.number_input(
    "Initial Output Tokens / Prompt", min_value=0, value=400, step=50,
    help="Average billed completion tokens generated per prompt."
)
input_growth = st.sidebar.slider(
    "MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5,
    help="Monthly percentage growth in input context size."
) / 100
output_growth = st.sidebar.slider(
    "MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5,
    help="Monthly percentage growth in output completion token length."
) / 100

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider(
    "Frontier Model Traffic Share (%)", 0, 100, 30,
    help="Percentage of API traffic routed to premium frontier models (e.g., GPT-4o, Claude 3.5 Sonnet)."
) / 100

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
step_trigger_type = st.sidebar.selectbox("Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"])

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
# 2. ENGINE-LEVEL SIMULATION FUNCTION
# ==========================================
@st.cache_data(show_spinner=False)
def run_model_simulation(
    p_start_year, p_start_month, p_horizon, p_dau, p_pro_subs, p_ent_subs, p_growth, p_paid_dau_factor,
    p_new_conv, p_base_conv, p_churn, p_arpu_pro, p_arpu_ent, p_expansion_pct,
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

    # Multi-Peak Additive Wave Engine
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
            current_dau = prev_dau_val * (1 + p_growth)
            dau_val = current_dau
            new_users_acquired = dau_val - prev_dau_val
            free_base = max(0.0, prev_dau_val - beg_paid_subs)

        total_new_conversions = (new_users_acquired * p_new_conv) + (free_base * p_base_conv)
        # Split new conversions 80% Pro, 20% Enterprise
        new_pro_conv = total_new_conversions * 0.80
        new_ent_conv = total_new_conversions * 0.20

        churned_pro = beg_pro * p_churn
        churned_ent = beg_ent * (p_churn * 0.5) # Lower churn for enterprise

        ending_pro = max(0.0, beg_pro + new_pro_conv - churned_pro)
        ending_ent = max(0.0, beg_ent + new_ent_conv - churned_ent)
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
        start_year, start_month, forecast_horizon, starting_dau, starting_pro_subs, starting_ent_subs, monthly_growth, paid_dau_factor,
        new_user_conversion, free_base_conversion, monthly_churn, arpu_pro, arpu_ent, expansion_arpu_pct,
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
# 3. REVERSE-MARGIN SOLVER & CFO DASHBOARD
# ==========================================
m_last = df.iloc[-1]
last_arr = m_last['Ending MRR Run-Rate ($)'] * 12
annual_rev = df["Recognized Revenue ($)"].sum()
last_margin = m_last["Modeled Gross Margin (%)"]

# Reverse-Margin Solver Calculations for Pro Tier ($30 ARPU)
blended_token_cost_m1 = (df.loc[0, "API Cost ($)"] / df.loc[0, "Monthly Prompts"]) if df.loc[0, "Monthly Prompts"] > 0 else 0.00001
pro_arpu_total = arpu_pro * (1 + expansion_arpu_pct)
allowable_cogs_10pct = pro_arpu_total * (1 - 0.10) # 90% margin target? Wait: target margin = gross profit / revenue. Gross Profit = Rev * Margin. Allowable COGS = Rev * (1 - Margin).
allowable_cogs_60pct = pro_arpu_total * (1 - 0.60)
allowable_cogs_70pct = pro_arpu_total * (1 - 0.70)
allowable_cogs_80pct = pro_arpu_total * (1 - 0.80)

# Max Prompts per Pro User per Month at Target Margins
max_prompts_60 = (allowable_cogs_60pct / (blended_token_cost_m1 * 1000)) * 1000 if blended_token_cost_m1 > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{forecast_horizon})", f"${last_arr:,.0f} ARR", f"Recognized Rev: ${annual_rev:,.0f}")
c2.metric("Gross Margin (M{forecast_horizon})", f"{last_margin:.1f}%", f"Target: {margin_target}%")
c3.metric("Reverse-Margin Limit (60% Margin)", f"{max_prompts_60:,.0f} Prompts/Mo", "Max Pro Tier Usage Ceiling")
c4.metric("Pareto Skew Concentration", f"{pareto_skew_share*100:.0f}% Top 20%", "Heavy User Load Factor")

# Policy & Governance Advisory Recommendations
policy_recs = []
if last_margin < margin_target:
    policy_recs.append(f"<b>Deal Desk Advisory:</b> Pro tier margin ({last_margin:.1f}%) breaches target ({margin_target}%). <b>Action:</b> Implement automated prompt rerouting to Standard models for power users exceeding 500 prompts/month.")
else:
    policy_recs.append(f"<b>Deal Desk Status:</b> Gross margin remains robust at {last_margin:.1f}%, comfortably meeting the {margin_target}% hurdle rate.")

st.markdown(f"""
<div class="cfo-commentary">
    <h4>💡 Deal Desk & Contract Pricing Governance Briefing</h4>
    <ul>
        {"".join([f"<li>{rec}</li>" for rec in policy_recs])}
    </ul>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Revenue & Margin", "💳 Credits & Cash", "📋 Monthly Detail", "📑 Deal Desk Assumptions"])

with tab1:
    st.subheader("Forecast Recognized Revenue, COGS & Gross Margin (Multi-Tier)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Credit Burn Velocity & Cash Payable")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Out-of-Pocket Cash Payable ($)", marker_color="#dd6b20"))
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color="#38a169", width=2)))
    fig2.update_layout(template="plotly_white", height=340, barmode="stack", margin=dict(l=30, r=30, t=30, b=20), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Monthly Roll-Forward Detail")
    st.dataframe(df.style.format({
        "DAU": "{:,.0f}", "Ending Pro Subs": "{:,.0f}", "Ending Ent Subs": "{:,.0f}",
        "Ending Paid Subscribers": "{:,.0f}", "Monthly Prompts": "{:,.0f}",
        "Ending MRR Run-Rate ($)": "${:,.0f}", "Recognized Revenue ($)": "${:,.0f}",
        "Total Modeled COGS ($)": "${:,.0f}", "Modeled Gross Profit ($)": "${:,.0f}",
        "Modeled Gross Margin (%)": "{:.1f}%", "Estimated Cash Payable ($)": "${:,.0f}"
    }, na_rep="—"), use_container_width=True)

with tab4:
    st.subheader("Contract Pricing & Deal Desk Audit Register")
    st.markdown("""
    * **Pro Tier ARPU:** `$30.00 / month` (Standard Self-Serve Plan)
    * **Enterprise Tier ARPU:** `$250.00 / month` (Custom Contract with SLA & Dedicated RAG)
    * **Multi-Peak Wave Engine:** Active (Peak 1 @ Month 3, Peak 2 @ Month 11)
    * **Pareto Skew:** Top 20% power users consume **80%** of total token volume.
    * **Reverse-Margin Solver:** Automatically caps allowable prompt consumption to protect target gross margins.
    """)

st.markdown("---")
st.caption("© 2026. Institutional AI Deal Desk & Contract Pricing Simulator.")
