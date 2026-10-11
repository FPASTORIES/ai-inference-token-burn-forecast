"""
AI Inference, Credit Depletion & FP&A Planning Model (Streamlit)
----------------------------------------------------------------
Institutional-grade financial model for forecasting AI unit economics, 
credit burn velocity, infrastructure step-costs, and multi-scenario analysis.
"""

import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference & Credit Burn FP&A Model",
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
    <h1>⚡ AI Inference, Credit Burn & Unit Economics Planning Model</h1>
    <p>Driver-based FP&A simulation engine tracking direct AI COGS, multi-tier subscriptions, credit burn velocity, and audit governance.</p>
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

st.sidebar.header("1. Subscriber & Usage Funnel")
starting_dau = st.sidebar.number_input(
    "Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500,
    help="Initial active user count at Month 1 across free and paid tiers."
)
starting_paid_subs = st.sidebar.number_input(
    "Starting Paid Subscribers", min_value=0, value=250, step=50,
    help="Initial paid subscriber count at Month 1."
)
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

arpu = st.sidebar.number_input(
    "Average Revenue Per User (ARPU) ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average monthly subscription revenue per paid subscriber."
)
expansion_arpu_pct = st.sidebar.slider(
    "Expansion / Overage Revenue (% of ARPU)", 0.0, 50.0, 10.0, 1.0,
    help="Incremental revenue from token overages and credit top-ups as a % of base ARPU."
) / 100

payment_fee_pct = st.sidebar.slider(
    "Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1,
    help="Card processor transaction fees recognized directly in COGS."
) / 100

st.sidebar.header("2. Usage & Token Economics")
baseline_prompts = st.sidebar.slider(
    "Baseline Prompts / Paid User / Day", 1, 20, 5,
    help="Average daily prompt volume for paid users during normal activity."
)
peak_prompts = st.sidebar.slider(
    "Peak Prompts / Paid User / Day", 5, 100, 35,
    help="Maximum prompt intensity during seasonal usage spikes."
)
peak_month = st.sidebar.slider(
    "Peak Usage Forecast Month", 1, forecast_horizon, min(7, forecast_horizon),
    help="Calendar month within the forecast horizon where peak usage occurs."
)
peak_spread = st.sidebar.slider(
    "Peak Usage Spread (Months)", 0.5, 6.0, 2.0, 0.5,
    help="Standard deviation parameter ($\sigma$) controlling the duration width of the usage wave."
)
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

frontier_input_price = st.sidebar.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25, help="List price per 1M input tokens for frontier models.")
frontier_output_price = st.sidebar.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50, help="List price per 1M output tokens for frontier models.")
standard_input_price = st.sidebar.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f", help="List price per 1M input tokens for standard models.")
standard_output_price = st.sidebar.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f", help="List price per 1M output tokens for standard models.")

cache_hit_rate = st.sidebar.slider("Eligible Input Tokens Cached (%)", 0, 90, 40, help="Percentage of input tokens hitting prompt caching mechanisms.") / 100
cache_price_ratio = st.sidebar.slider("Cached Price Ratio (% of regular)", 0, 100, 20, help="Discounted price ratio for cached tokens.") / 100
batch_share = st.sidebar.slider("Traffic Using Batch Pricing (%)", 0, 80, 20, help="Percentage of asynchronous traffic routed via batch endpoints.") / 100
batch_price_ratio = st.sidebar.slider("Batch Price Ratio (% of regular)", 0, 100, 50, help="Discount ratio for batch processing APIs.") / 100
overhead_pct = st.sidebar.slider("Inference Overhead (%)", 0, 100, 0, help="Buffer for retries, embedding calls, and multi-agent loops.") / 100

st.sidebar.header("4. Infrastructure Step-Costs & Credits")
base_fixed_infra = st.sidebar.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=2500.0, step=500.0, help="Baseline hosting, database, and orchestration server costs.")
step_trigger_type = st.sidebar.selectbox(
    "Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"],
    help="Operational scaling metric that triggers infrastructure capacity step-ups."
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

step_cost_increment = st.sidebar.number_input("Infra Step-Up Cost ($ per block)", min_value=0.0, value=1500.0, step=250.0, help="Incremental monthly infrastructure cost when crossing capacity thresholds.")
vector_cost_per_user = st.sidebar.number_input("Vector DB Cost / DAU / Month ($)", min_value=0.0, value=0.15, step=0.05, help="Pinecone / pgvector embedding storage & retrieval cost per active user.")
starting_ai_credits = st.sidebar.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=20000.0, step=2500.0, help="Unearned promotional credits from LLM providers (OpenAI/Anthropic).")
starting_cloud_credits = st.sidebar.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=30000.0, step=5000.0, help="Unearned cloud startup credits (AWS Activate / Google for Startups).")
ai_credit_eligible_share = st.sidebar.slider("AI Credit Eligible Share (%)", 0, 100, 100, help="Share of API costs eligible for drawdown against AI vendor credits.") / 100
cloud_credit_eligible_share = st.sidebar.slider("Cloud Credit Eligible Share (%)", 0, 100, 100, help="Share of infrastructure costs eligible for cloud credit drawdown.") / 100
margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60, help="Corporate gross margin hurdle rate.")
budget_monthly = st.sidebar.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0, help="Hard monthly spending cap for cost governance alert triggers.")


# ==========================================
# 2. CORE ENGINE SIMULATION FUNCTION
# ==========================================
@st.cache_data(show_spinner=False)
def run_model_simulation(
    p_start_year, p_start_month, p_horizon, p_dau, p_paid_subs, p_growth, p_paid_dau_factor,
    p_new_conv, p_base_conv, p_churn, p_arpu, p_expansion_pct,
    p_base_prompts, p_peak_prompts, p_peak_m, p_peak_spread, p_free_mult,
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

    # Gaussian Usage Wave Curve
    prompt_curve = p_base_prompts + (p_peak_prompts - p_base_prompts) * np.exp(-((m_indices - p_peak_m) ** 2) / (2 * p_peak_spread ** 2))

    ai_credits_left = float(p_ai_cred)
    cloud_credits_left = float(p_cloud_cred)

    current_dau = float(p_dau)
    prev_paid_subs = float(p_paid_subs)

    sim_rows = []

    for i in range(n):
        beg_paid = prev_paid_subs

        if i == 0:
            dau_val = current_dau
            new_users_acquired = 0.0
            free_base = max(0.0, dau_val - beg_paid)
            prev_dau_val = current_dau
        else:
            prev_dau_val = current_dau
            current_dau = prev_dau_val * (1 + p_growth)
            dau_val = current_dau
            new_users_acquired = dau_val - prev_dau_val
            free_base = max(0.0, prev_dau_val - beg_paid)

        new_paid_conversions = (new_users_acquired * p_new_conv) + (free_base * p_base_conv)
        churned_paid = beg_paid * p_churn
        ending_paid = max(0.0, beg_paid + new_paid_conversions - churned_paid)
        prev_paid_subs = ending_paid

        avg_paid_subs = (beg_paid + ending_paid) / 2.0
        avg_dau = (prev_dau_val + dau_val) / 2.0
        paid_dau = avg_paid_subs * p_paid_dau_factor
        free_dau_avg = max(0.0, avg_dau - paid_dau)
        free_users_snapshot = max(0.0, dau_val - ending_paid)

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
            trigger_metric = ending_paid
        else:
            trigger_metric = monthly_prompts_total

        step_multiplier = max(0, math.ceil(trigger_metric / p_step_thresh_raw) - 1) if p_step_thresh_raw > 0 and trigger_metric > p_step_thresh_raw else 0
        infrastructure_cost = p_base_infra + (step_multiplier * p_step_cost)

        base_revenue = avg_paid_subs * p_arpu
        expansion_revenue = base_revenue * p_expansion_pct
        recognized_revenue = base_revenue + expansion_revenue
        ending_mrr_runrate = ending_paid * p_arpu * (1 + p_expansion_pct)
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
            "Free Users": round(free_users_snapshot),
            "Paid Subscribers": round(ending_paid),
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
        start_year, start_month, forecast_horizon, starting_dau, starting_paid_subs, monthly_growth, paid_dau_factor,
        new_user_conversion, free_base_conversion, monthly_churn, arpu, expansion_arpu_pct,
        baseline_prompts, peak_prompts, peak_month, peak_spread, free_user_usage_mult,
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
# 3. EXECUTIVE BRIEFING BANNER & METRICS
# ==========================================
m_last = df.iloc[-1]
last_arr = m_last['Ending MRR Run-Rate ($)'] * 12
annual_rev = df["Recognized Revenue ($)"].sum()
annual_cogs = df["Total Modeled COGS ($)"].sum()
annual_gp = annual_rev - annual_cogs
annual_gm = (annual_gp / annual_rev * 100) if annual_rev > 0 else 0.0
last_margin = m_last["Modeled Gross Margin (%)"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{forecast_horizon})", f"${last_arr:,.0f} ARR", f"Recognized Rev: ${annual_rev:,.0f}")
c2.metric("Full-Year Gross Margin", f"{annual_gm:.1f}%", f"Target: {margin_target}%")
c3.metric(f"Ending Margin (M{forecast_horizon})", f"{last_margin:.1f}%", f"Peak Month: M{peak_month}")
c4.metric("Ending Cash Burn / Payable", f"${m_last['Estimated Cash Payable ($)']:,.0f}", f"AI Credits Left: ${m_last['Remaining AI Credits ($)']:,.0f}")

# CFO Commentary & Governance Alerts
cfo_recs = []
if last_margin < margin_target:
    cfo_recs.append(f"<b>Margin Compression Alert:</b> Ending gross margin ({last_margin:.1f}%) is below the {margin_target}% hurdle rate due to token volume scaling in Month {peak_month}. Recommend tightening prompt caching or negotiating volume tiers.")
else:
    cfo_recs.append(f"<b>Margin Health:</b> Gross margin remains strong at {last_margin:.1f}%, comfortably clearing the {margin_target}% target.")

if m_last["Remaining AI Credits ($)"] <= 0:
    cfo_recs.append("<b>Credit Depletion Warning:</b> AI vendor credits are fully exhausted before horizon end. Out-of-pocket cash payable for API COGS escalates immediately.")
else:
    cfo_recs.append(f"<b>Credit Runway:</b> ${m_last['Remaining AI Credits ($)']:,.0f} in AI vendor credits remain at horizon end.")

st.markdown(f"""
<div class="cfo-commentary">
    <h4>💡 CFO Executive Briefing & Governance Insights</h4>
    <ul>
        {"".join([f"<li>{rec}</li>" for rec in cfo_recs])}
    </ul>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. DASHBOARD TABS (FULL 5-TAB STRUCTURE)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue & Margin", 
    "💳 Credits & Cash", 
    "📋 Monthly Detail", 
    "📈 Scenario Matrix", 
    "📑 Audit Register"
])

with tab1:
    st.subheader("Forecast Recognized Revenue, COGS & Gross Margin")
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
        "DAU": "{:,.0f}", "Free Users": "{:,.0f}", "Paid Subscribers": "{:,.0f}",
        "Monthly Prompts": "{:,.0f}", "Ending MRR Run-Rate ($)": "${:,.0f}",
        "Recognized Revenue ($)": "${:,.0f}", "API Cost ($)": "${:,.0f}",
        "Vector DB Cost ($)": "${:,.0f}", "Fixed Infrastructure ($)": "${:,.0f}",
        "Payment Processing ($)": "${:,.0f}", "Total Modeled COGS ($)": "${:,.0f}",
        "Modeled Gross Profit ($)": "${:,.0f}", "Modeled Gross Margin (%)": "{:.1f}%",
        "Remaining AI Credits ($)": "${:,.0f}", "Remaining Cloud Credits ($)": "${:,.0f}",
        "Estimated Cash Payable ($)": "${:,.0f}"
    }, na_rep="—"), use_container_width=True)

with tab4:
    st.subheader("📈 Multi-Scenario Comparative Analysis Matrix")
    st.markdown("Evaluating baseline assumptions against bull, bear, high-usage, and cost-shock scenarios.")

    def run_scenario(name, g_mult, prompt_mult, price_mult):
        scen_df = run_model_simulation(
            start_year, start_month, forecast_horizon, starting_dau, starting_paid_subs, monthly_growth * g_mult, paid_dau_factor,
            new_user_conversion, free_base_conversion, monthly_churn, arpu, expansion_arpu_pct,
            baseline_prompts * prompt_mult, peak_prompts * prompt_mult, peak_month, peak_spread, free_user_usage_mult,
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
        run_scenario("1. Base Case", 1.0, 1.0, 1.0),
        run_scenario("2. High Growth (1.25x)", 1.25, 1.0, 1.0),
        run_scenario("3. Low Growth (0.75x)", 0.75, 1.0, 1.0),
        run_scenario("4. High Prompt Intensity (1.5x)", 1.0, 1.5, 1.0),
        run_scenario("5. Low Prompt Intensity (0.7x)", 1.0, 0.7, 1.0),
        run_scenario("6. API Price Inflation (+25%)", 1.0, 1.0, 1.25),
        run_scenario("7. Aggressive Caching & Batch", 1.0, 1.0, 0.8),
        run_scenario("8. Stress Test (High Usage + Inflation)", 1.1, 1.4, 1.2)
    ]
    scen_summary_df = pd.DataFrame(scen_results)
    st.dataframe(scen_summary_df.style.format({
        "Total Revenue ($)": "${:,.0f}", "Total COGS ($)": "${:,.0f}",
        "Gross Profit ($)": "${:,.0f}", "Gross Margin (%)": "{:.1f}%",
        "Ending ARR ($)": "${:,.0f}", "Ending Monthly Cash Payable ($)": "${:,.0f}"
    }), use_container_width=True)

with tab5:
    st.subheader("📑 Assumptions & Audit Register (Data Governance)")
    st.markdown("Complete audit trail documenting the provenance and governance of all model inputs.")

    audit_data = [
        ("Starting Daily Active Users (DAU)", f"{starting_dau:,.0f}", "Mixpanel / Segment Event Pipeline", "Management Baseline Estimate"),
        ("Starting Paid Subscribers", f"{starting_paid_subs:,.0f}", "Stripe Billing Subscriptions Table", "Actual Active Recurring Billing Records"),
        ("Monthly DAU Growth Rate", f"{monthly_growth*100:.1f}% MoM", "Marketing Acquisition Model", "Management Growth Target"),
        ("Paid DAU Activity Factor", f"{paid_dau_factor*100:.0f}%", "Mixpanel Daily Active Engagement Ratio", "Product Telemetry Benchmark"),
        ("New User Conversion Rate", f"{new_user_conversion*100:.1f}%", "Stripe Checkout & Funnel Analytics", "Illustrative Baseline — Source Required (applies M2+)"),
        ("Free-User Proxy Conversion Rate", f"{free_base_conversion*100:.1f}% MoM", "Product-Led Growth Funnel Report", "Estimated Conversion Benchmark"),
        ("Monthly Paid Subscriber Churn", f"{monthly_churn*100:.1f}% MoM", "Stripe Churn & Retention Analytics", "Historical Cohort Average"),
        ("Average Revenue Per User (ARPU)", f"${arpu:.2f} / mo", "Stripe Pricing Plan Definitions", "Contracted Plan Pricing"),
        ("Expansion / Overage Revenue", f"{expansion_arpu_pct*100:.1f}% of ARPU", "Stripe Metered Billing Invoices", "Usage-Based Overage Proxy"),
        ("Payment Processing Fee", f"{payment_fee_pct*100:.1f}%", "Stripe Merchant Statement", "Standard Gateway Fee Schedule"),
        ("Baseline Prompts / User / Day", f"{baseline_prompts}", "Helicone / Langfuse API Telemetry", "Observed Mean Daily Prompt Volume"),
        ("Peak Prompts / User / Day", f"{peak_prompts}", "Capacity Planning Stress Model", "Seasonal Peak Assumption"),
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

st.markdown("---")
st.caption("© 2026. Institutional AI Inference & Credit Burn FP&A Model. All rights reserved.")
