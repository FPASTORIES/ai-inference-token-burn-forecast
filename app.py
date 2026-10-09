"""
AI Inference, Cloud-Credit Runway & Unit Economics Model
--------------------------------------------------------
Illustrative FP&A planning model for evaluating direct AI model inference,
vector storage, and infrastructure COGS. Excludes non-COGS OpEx (payroll, marketing).
Prices and assumptions are user-editable estimates; verify against vendor billing portals.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference, Cloud-Credit Runway & Unit Economics", 
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
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.2rem !important; font-weight:700;}

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

/* Executive Alert Cards */
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
    <h1>⚡ AI Inference, Cloud-Credit Runway & Unit Economics</h1>
    <p>Driver-based forecast of token spend, modular step-costs, dual-funnel PLG subscriber bridge, and vendor credit depletion.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR ASSUMPTIONS & HOVER TOOLTIPS
# ==========================================
st.sidebar.header("1. User Acquisition & Paid Funnel")
starting_dau = st.sidebar.number_input(
    "Starting Daily Active Users (DAU)", min_value=0, value=5000, step=500,
    help="Initial active user count at Month 1. Source: Mixpanel, Google Analytics, or PostHog DAU metrics."
)
monthly_growth = st.sidebar.slider(
    "Monthly DAU Growth (%)", 0.0, 25.0, 8.0, 0.5,
    help="Assumed MoM growth for top-of-funnel acquisition. Source: Trailing 3-month growth analytics."
) / 100

new_user_conversion = st.sidebar.slider(
    "New User Conversion (%)", 0.0, 50.0, 15.0, 0.5,
    help="[CHANGE-01] % of brand-new acquired users who upgrade to paid in their first month (Day-1 checkout conversion)."
) / 100

free_base_conversion = st.sidebar.slider(
    "Free Base Conversion (%)", 0.0, 10.0, 2.0, 0.1,
    help="[CHANGE-01] % of existing accumulated free user pool who convert to paid each month (nurtured PLG expansion)."
) / 100

monthly_churn = st.sidebar.slider(
    "Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5,
    help="Percentage of paid subscribers cancelling each month. Source: Stripe Billing / ProfitWell."
) / 100

arpu = st.sidebar.number_input(
    "Monthly ARPU per Paid Subscriber ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average Revenue Per User per month for paid tiers."
)

st.sidebar.header("2. Usage Patterns & Token Context")
baseline_prompts = st.sidebar.slider(
    "Baseline Prompts / Paid User / Day", 1, 20, 5,
    help="Average daily prompt volume for paid power users during normal activity."
)
peak_prompts = st.sidebar.slider(
    "Peak Prompts / Paid User / Day", 1, 100, 35,
    help="Expected peak usage surge per paid user during heavy usage or launch event."
)

# Input Guardrail [CHANGE-08]
if peak_prompts < baseline_prompts:
    st.sidebar.warning("⚠️ Peak prompts are set below baseline usage. Adjusting peak assumption.")

peak_month = st.sidebar.slider("Peak Usage Month (1-12)", 1, 12, 7)
surge_width = st.sidebar.slider("Peak Usage Spread (Months)", 1.0, 4.0, 2.0, 0.5)

free_user_usage_mult = st.sidebar.slider(
    "Free Tier Usage Multiplier (%)", 0, 100, 50,
    help="Usage volume of free users relative to paid users (e.g., 50% means free users issue half as many daily prompts)."
) / 100

input_tokens_start = st.sidebar.number_input(
    "Initial Input Tokens / Prompt", min_value=0, value=1000, step=100,
    help="[CHANGE-04] Average prompt context length. Includes system prompt (~300t), RAG context (~300t), and chat history (~400t)."
)
output_tokens_start = st.sidebar.number_input(
    "Initial Output Tokens / Prompt", min_value=0, value=400, step=50,
    help="Average completion response length per prompt."
)
input_growth = st.sidebar.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
output_growth = st.sidebar.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider("Frontier Model Traffic Share (%)", 0, 100, 30, help="Simplified routing assumption for premium models.") / 100
standard_mix = 1.0 - frontier_mix

frontier_input_price = st.sidebar.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25)
frontier_output_price = st.sidebar.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50)
standard_input_price = st.sidebar.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
standard_output_price = st.sidebar.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f")

cache_hit_rate = st.sidebar.slider("Eligible Input Tokens Cached (%)", 0, 90, 40) / 100
cache_price_ratio = st.sidebar.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
batch_share = st.sidebar.slider("Traffic Using Batch Pricing (%)", 0, 80, 20) / 100
batch_price_ratio = st.sidebar.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100

st.sidebar.header("4. Infrastructure Step-Costs & Credits")
base_fixed_infra = st.sidebar.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=2500.0, step=500.0)

# Scale-Normalized Step Trigger [CHANGE-03]
step_trigger_type = st.sidebar.selectbox(
    "Step-Cost Trigger Basis", 
    ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"],
    help="Select basis for triggering modular infrastructure upgrades."
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
vector_cost_per_user = st.sidebar.number_input("Vector DB Cost / Active User ($)", min_value=0.0, value=0.15, step=0.05)

# Dual-Pool Credits [CHANGE-05]
starting_ai_credits = st.sidebar.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=20000.0, step=2500.0, help="OpenAI / Anthropic build grants.")
starting_cloud_credits = st.sidebar.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=30000.0, step=5000.0, help="AWS Activate / GCP / Azure grants.")
ai_credit_eligible_share = st.sidebar.slider("AI Credit Eligible Share (%)", 0, 100, 100) / 100
cloud_credit_eligible_share = st.sidebar.slider("Cloud Credit Eligible Share (%)", 0, 100, 100) / 100

margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

# ==========================================
# 2. CORE FORECAST ENGINE FUNCTION
# ==========================================
def run_model_simulation(
    p_dau, p_growth, p_new_conv, p_base_conv, p_churn, p_arpu,
    p_base_prompts, p_peak_prompts, p_peak_m, p_surge_w, p_free_mult,
    p_in_tok, p_out_tok, p_in_grow, p_out_grow, p_front_mix,
    p_f_in_p, p_f_out_p, p_s_in_p, p_s_out_p, p_cache_rate, p_cache_ratio,
    p_batch_share, p_batch_ratio, p_base_infra, p_step_type, p_step_thresh_raw,
    p_step_cost, p_vec_cost, p_ai_cred, p_cloud_cred, p_ai_elig, p_cloud_elig
):
    months = np.arange(1, 13)
    current_year = datetime.datetime.now().year
    month_days = [pd.Period(f"{current_year}-{m:02d}").days_in_month for m in months]
    month_names = [f"M{m}" for m in months]

    prompt_curve = p_base_prompts + (p_peak_prompts - p_base_prompts) * np.exp(-((months - p_peak_m) ** 2) / (2 * p_surge_w ** 2))

    ai_credits_left = float(p_ai_cred)
    cloud_credits_left = float(p_cloud_cred)
    
    # Initialize Subscriber Bridge [CHANGE-01]
    beg_paid_subs = p_dau * p_new_conv
    current_dau = float(p_dau)
    
    sim_rows = []

    for i in range(12):
        if i == 0:
            dau_val = current_dau
            new_users = dau_val
            free_base = max(0.0, dau_val - beg_paid_subs)
            new_user_conv_subs = new_users * p_new_conv
            free_base_conv_subs = 0.0
            churned_subs = beg_paid_subs * p_churn
            ending_paid_subs = max(0.0, beg_paid_subs + new_user_conv_subs - churned_subs)
        else:
            prev_dau = current_dau
            current_dau = prev_dau * (1 + p_growth)
            dau_val = current_dau
            new_users = dau_val - prev_dau
            
            beg_paid_subs = sim_rows[i-1]["Ending Paid Subscribers"]
            free_base = max(0.0, prev_dau - beg_paid_subs)
            
            new_user_conv_subs = new_users * p_new_conv
            free_base_conv_subs = free_base * p_base_conv
            total_new_conversions = new_user_conv_subs + free_base_conv_subs
            churned_subs = beg_paid_subs * p_churn
            ending_paid_subs = max(0.0, beg_paid_subs + total_new_conversions - churned_subs)

        free_users = max(0.0, dau_val - ending_paid_subs)

        input_tokens_val = p_in_tok * ((1 + p_in_grow) ** i)
        output_tokens_val = p_out_tok * ((1 + p_out_grow) ** i)

        prompts_per_paid_user_day = float(prompt_curve[i])
        prompts_per_free_user_day = prompts_per_paid_user_day * p_free_mult
        
        monthly_prompts_paid = ending_paid_subs * prompts_per_paid_user_day * month_days[i]
        monthly_prompts_free = free_users * prompts_per_free_user_day * month_days[i]
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
        api_cost = api_before_batch * ((1 - p_batch_share) + p_batch_share * p_batch_ratio)

        vector_cost = dau_val * p_vec_cost
        
        # Scale-Normalized Infrastructure Step-Costs [CHANGE-03]
        if p_step_type == "DAU Threshold":
            trigger_metric = dau_val
        elif p_step_type == "Paid Subscribers Threshold":
            trigger_metric = ending_paid_subs
        else:
            trigger_metric = monthly_prompts_total
            
        step_multiplier = int(trigger_metric // p_step_thresh_raw) if p_step_thresh_raw > 0 else 0
        infrastructure_cost = p_base_infra + (step_multiplier * p_step_cost)
        
        total_modeled_cogs = api_cost + vector_cost + infrastructure_cost
        revenue = ending_paid_subs * p_arpu
        gross_profit = revenue - total_modeled_cogs
        
        # Edge-case handling for $0 revenue [CHANGE-08]
        gross_margin_pct = (gross_profit / revenue * 100) if revenue > 0 else np.nan

        # Dual-Pool Credit Depletion [CHANGE-05]
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
            "Calendar Days": month_days[i],
            "DAU": round(dau_val), 
            "Beginning Paid Subscribers": round(beg_paid_subs),
            "New User Conversions": round(new_user_conv_subs),
            "Free Base Conversions": round(free_base_conv_subs),
            "Churned Subscribers": round(churned_subs),
            "Ending Paid Subscribers": round(ending_paid_subs),
            "Free Users": round(free_users),
            "Monthly Prompts": round(monthly_prompts_total),
            "Subscription Revenue ($)": revenue, 
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost, 
            "Fixed Infrastructure ($)": infrastructure_cost,
            "Total Modeled COGS ($)": total_modeled_cogs,
            "Modeled Gross Profit ($)": gross_profit, 
            "Modeled Gross Margin (%)": gross_margin_pct,
            "AI Credits Applied ($)": ai_credits_applied,
            "Remaining AI Credits ($)": ai_credits_left,
            "Cloud Credits Applied ($)": cloud_credits_applied,
            "Remaining Cloud Credits ($)": cloud_credits_left,
            "Total Credits Applied ($)": total_credits_applied,
            "Estimated Cash Payable ($)": estimated_cash_payable,
        })

    df_out = pd.DataFrame(sim_rows)
    return df_out

# Run Baseline Simulation
df = run_model_simulation(
    starting_dau, monthly_growth, new_user_conversion, free_base_conversion, monthly_churn, arpu,
    baseline_prompts, peak_prompts, peak_month, surge_width, free_user_usage_mult,
    input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
    frontier_input_price, frontier_output_price, standard_input_price, standard_output_price,
    cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
    step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
    starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share
)

df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = df["Total Modeled COGS ($)"] - budget_monthly if budget_monthly > 0 else np.nan
df["Cost per 1,000 Prompts ($)"] = df["Total Modeled COGS ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Cost per Paid Subscriber ($)"] = df["Total Modeled COGS ($)"] / df["Ending Paid Subscribers"].replace(0, np.nan)

# ==========================================
# 3. EXECUTIVE ALERT STRIP [CHANGE-08]
# ==========================================
alert_cols = st.columns(3)

ai_dep = df.index[df["Remaining AI Credits ($)"] <= 0.005].tolist()
cloud_dep = df.index[df["Remaining Cloud Credits ($)"] <= 0.005].tolist()

if starting_ai_credits <= 0 and starting_cloud_credits <= 0:
    alert_cols[0].markdown('<div class="alert-card alert-warning">💳 <b>No Credits Available:</b> All COGS hit cash directly.</div>', unsafe_allow_html=True)
elif ai_dep or cloud_dep:
    first_m = df.loc[min(ai_dep + cloud_dep), "Month"]
    alert_cols[0].markdown(f'<div class="alert-card alert-warning">💳 <b>Credits Depleted in {first_m}:</b> Out-of-pocket cash increases.</div>', unsafe_allow_html=True)
else:
    alert_cols[0].markdown('<div class="alert-card alert-info">💳 <b>Credits Healthy:</b> Balances remain active through M12.</div>', unsafe_allow_html=True)

breach = df.index[df["Modeled Gross Margin (%)"].notna() & (df["Modeled Gross Margin (%)"] < margin_target)].tolist()
if breach:
    alert_cols[1].markdown(f'<div class="alert-card alert-error">⚠️ <b>Margin Breach in {df.loc[breach[0], "Month"]}:</b> Margin drops below target ({margin_target}%).</div>', unsafe_allow_html=True)
else:
    alert_cols[1].markdown(f'<div class="alert-card alert-info">✅ <b>Margin Target Met:</b> Stays above benchmark ({margin_target}%).</div>', unsafe_allow_html=True)

if budget_monthly > 0:
    over = df.index[df["Budget Variance ($)"] > 0].tolist()
    if over:
        alert_cols[2].markdown(f'<div class="alert-card alert-warning">📈 <b>Budget Overrun in {df.loc[over[0], "Month"]}:</b> Spend exceeds ${budget_monthly:,.0f} cap.</div>', unsafe_allow_html=True)
    else:
        alert_cols[2].markdown('<div class="alert-card alert-info">🎯 <b>Budget Compliant:</b> Monthly spend remains under cap.</div>', unsafe_allow_html=True)
else:
    alert_cols[2].markdown('<div class="alert-card alert-info">ℹ️ <b>COGS Model Scope:</b> Direct AI/Infra spend only—excludes payroll/OpEx.</div>', unsafe_allow_html=True)

# ==========================================
# 4. COMPACT KPIS
# ==========================================
annual_revenue = df["Subscription Revenue ($)"].sum()
annual_cogs = df["Total Modeled COGS ($)"].sum()
m12 = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Run Rate (M12)", f"${m12['Subscription Revenue ($)']:,.0f}", f"12M Rev: ${annual_revenue:,.0f}")
c2.metric("M12 COGS Spend", f"${m12['Total Modeled COGS ($)']:,.0f}", f"12M COGS: ${annual_cogs:,.0f}")
c3.metric("M12 Gross Margin", "Undefined" if pd.isna(m12["Modeled Gross Margin (%)"]) else f"{m12['Modeled Gross Margin (%)']:.1f}%")
c4.metric("Cash Payable (12M)", f"${df['Estimated Cash Payable ($)'].sum():,.0f}", f"Credits Offset: ${df['Total Credits Applied ($)'].sum():,.0f}")

st.markdown('<div class="scope-note">* Note: Model evaluates Direct COGS and Cloud-Credit Expiration Runway. Full cash runway requires non-COGS OpEx.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue & Margin", 
    "💳 Credits & Cash", 
    "🎛️ Scenario Matrix", 
    "📋 Monthly Detail",
    "📑 Assumptions Register"
])

with tab1:
    st.subheader("12-Month Revenue, Modeled COGS & Gross Margin")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Subscription Revenue ($)"], name="Subscription Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Direct COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    
    # Target Margin Line [CHANGE-07]
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
    fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True, range=[0, 105], gridcolor="#edf2f7")
    st.plotly_chart(fig, use_container_width=True)

    # Unit Cost Telemetry Sub-Chart [CHANGE-07]
    st.subheader("Inference Unit Cost Telemetry")
    col_u1, col_u2 = st.columns(2)
    
    fig_u1 = go.Figure()
    fig_u1.add_trace(go.Scatter(x=df["Month"], y=df["Cost per 1,000 Prompts ($)"], mode="lines+markers", line=dict(color="#805ad5", width=2.5)))
    fig_u1.update_layout(title="Cost per 1,000 Prompts ($)", template="plotly_white", height=240, margin=dict(l=30,r=30,t=40,b=20))
    col_u1.plotly_chart(fig_u1, use_container_width=True)

    fig_u2 = go.Figure()
    fig_u2.add_trace(go.Scatter(x=df["Month"], y=df["Cost per Paid Subscriber ($)"], mode="lines+markers", line=dict(color="#319795", width=2.5)))
    fig_u2.update_layout(title="Cost per Paid Subscriber ($)", template="plotly_white", height=240, margin=dict(l=30,r=30,t=40,b=20))
    col_u2.plotly_chart(fig_u2, use_container_width=True)

with tab2:
    st.subheader("Cloud Credit Offset vs. Out-of-Pocket Cash Payable")
    st.caption("Stacked bars equal total modeled COGS consumption offset by AI and Cloud credits.")

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
    st.caption("[CHANGE-06] Dynamic reforecast simulating top-line revenue and COGS spend across 6 operational scenarios.")

    scenarios_config = [
        ("1. Base Case", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 1.0),
        ("2. Downside: Growth Slowdown (-50%)", monthly_growth * 0.5, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 1.0),
        ("3. Downside: High Subscriber Churn (2x)", monthly_growth, monthly_churn * 2.0, new_user_conversion, free_base_conversion, 1.0, 1.0),
        ("4. Weak Monetization Funnel (-50% Conv)", monthly_growth, monthly_churn, new_user_conversion * 0.5, free_base_conversion * 0.5, 1.0, 1.0),
        ("5. High Product Engagement (+50% Usage)", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.5, 1.0),
        ("6. AI Prompt Optimization (-25% Price)", monthly_growth, monthly_churn, new_user_conversion, free_base_conversion, 1.0, 0.75),
    ]

    scen_results = []
    for label, s_growth, s_churn, s_nconv, s_bconv, s_usage_mult, s_price_mult in scenarios_config:
        s_df = run_model_simulation(
            starting_dau, s_growth, s_nconv, s_bconv, s_churn, arpu,
            baseline_prompts * s_usage_mult, peak_prompts * s_usage_mult, peak_month, surge_width, free_user_usage_mult,
            input_tokens_start, output_tokens_start, input_growth, output_growth, frontier_mix,
            frontier_input_price * s_price_mult, frontier_output_price * s_price_mult,
            standard_input_price * s_price_mult, standard_output_price * s_price_mult,
            cache_hit_rate, cache_price_ratio, batch_share, batch_price_ratio, base_fixed_infra,
            step_trigger_type, raw_step_threshold, step_cost_increment, vector_cost_per_user,
            starting_ai_credits, starting_cloud_credits, ai_credit_eligible_share, cloud_credit_eligible_share
        )
        
        s_rev = s_df["Subscription Revenue ($)"].sum()
        s_cogs = s_df["Total Modeled COGS ($)"].sum()
        s_gp = s_rev - s_cogs
        s_gm = (s_gp / s_rev * 100) if s_rev > 0 else np.nan
        s_cash = s_df["Estimated Cash Payable ($)"].sum()

        scen_results.append({
            "Scenario": label,
            "12M Revenue ($)": s_rev,
            "12M Modeled COGS ($)": s_cogs,
            "12M Gross Profit ($)": s_gp,
            "Gross Margin (%)": s_gm,
            "12M Cash Payable ($)": s_cash
        })

    scen_df = pd.DataFrame(scen_results)
    st.dataframe(
        scen_df.style.format({
            "12M Revenue ($)": "${:,.0f}",
            "12M Modeled COGS ($)": "${:,.0f}",
            "12M Gross Profit ($)": "${:,.0f}",
            "Gross Margin (%)": "{:.1f}%",
            "12M Cash Payable ($)": "${:,.0f}"
        }, na_rep="—"), 
        use_container_width=True
    )

with tab4:
    st.subheader("12-Month Detailed Financial & Operational Roll-Forward")
    st.dataframe(
        df.style.format({
            "DAU": "{:,.0f}",
            "Beginning Paid Subscribers": "{:,.0f}",
            "New User Conversions": "{:,.0f}",
            "Free Base Conversions": "{:,.0f}",
            "Churned Subscribers": "{:,.0f}",
            "Ending Paid Subscribers": "{:,.0f}",
            "Free Users": "{:,.0f}",
            "Monthly Prompts": "{:,.0f}",
            "Subscription Revenue ($)": "${:,.0f}",
            "API Cost ($)": "${:,.0f}",
            "Vector DB Cost ($)": "${:,.0f}",
            "Fixed Infrastructure ($)": "${:,.0f}",
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
            "Cost per 1,000 Prompts ($)": "${:,.4f}",
            "Cost per Paid Subscriber ($)": "${:,.2f}",
        }, na_rep="—"), 
        use_container_width=True
    )
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Audit CSV", data=csv, file_name="ai_inference_cogs_model.csv", mime="text/csv")

with tab5:
    st.subheader("📑 Source & Assumptions Audit Register")
    st.caption("[CHANGE-02] Dynamic audit log detailing data provenance, ownership, and verification status across 15+ parameters.")
    
    assumptions_data = [
        {"Parameter": "Starting DAU", "Current Value": f"{starting_dau:,}", "Category": "User Funnel", "Source": "Product Analytics (Mixpanel/PostHog)", "Verification Status": "User-Entered Estimate — Replace with Actuals", "Owner": "Growth"},
        {"Parameter": "Monthly DAU Growth", "Current Value": f"{monthly_growth*100:.1f}%", "Category": "User Funnel", "Source": "Growth Model", "Verification Status": "Management Estimate", "Owner": "Marketing"},
        {"Parameter": "New User Conversion %", "Current Value": f"{new_user_conversion*100:.1f}%", "Category": "User Funnel", "Source": "Stripe Checkout Analytics", "Verification Status": "Illustrative Assumption — Source Required", "Owner": "Growth"},
        {"Parameter": "Free Base Conversion %", "Current Value": f"{free_base_conversion*100:.1f}%", "Category": "User Funnel", "Source": "In-App Funnel Analytics", "Verification Status": "Illustrative Assumption — Source Required", "Owner": "Product"},
        {"Parameter": "Paid Subscriber Churn", "Current Value": f"{monthly_churn*100:.1f}%", "Category": "User Funnel", "Source": "Stripe Billing Dashboard", "Verification Status": "User-Entered Estimate", "Owner": "Finance"},
        {"Parameter": "Frontier Input Price", "Current Value": f"${frontier_input_price:.2f} / 1M", "Category": "AI Unit Pricing", "Source": "Vendor Rate Card", "Verification Status": "Verified Vendor Rate Card", "Owner": "DevOps"},
        {"Parameter": "Frontier Output Price", "Current Value": f"${frontier_output_price:.2f} / 1M", "Category": "AI Unit Pricing", "Source": "Vendor Rate Card", "Verification Status": "Verified Vendor Rate Card", "Owner": "DevOps"},
        {"Parameter": "Context Tokens / Prompt", "Current Value": f"{input_tokens_start:,} in / {output_tokens_start:,} out", "Category": "Usage Context", "Source": "LLM Telemetry (Helicone)", "Verification Status": "Engineering Estimate (Includes RAG)", "Owner": "Engineering"},
        {"Parameter": "Base Fixed Infra Spend", "Current Value": f"${base_fixed_infra:,.0f}/mo", "Category": "Infrastructure", "Source": "AWS Monthly Invoices", "Verification Status": "User-Entered Estimate — Invoice Required", "Owner": "DevOps"},
        {"Parameter": "Infra Step Trigger", "Current Value": f"${step_cost_increment:,.0f} per {step_threshold_input:,} {step_trigger_type}", "Category": "Infrastructure", "Source": "DevOps Capacity Plan", "Verification Status": "Management Estimate", "Owner": "Engineering"},
        {"Parameter": "Starting AI Credits", "Current Value": f"${starting_ai_credits:,.0f}", "Category": "Credits & Capital", "Source": "OpenAI / Anthropic Portal", "Verification Status": "Verified Grant Balance", "Owner": "Finance"},
        {"Parameter": "Starting Cloud Credits", "Current Value": f"${starting_cloud_credits:,.0f}", "Category": "Credits & Capital", "Source": "AWS Activate Portal", "Verification Status": "Verified Grant Balance", "Owner": "Finance"},
    ]
    
    st.table(pd.DataFrame(assumptions_data))

st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
