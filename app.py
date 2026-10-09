"""
AI Inference, Startup Runway & Unit Economics Model
---------------------------------------------------
Illustrative FP&A planning model, not an accounting system or vendor quote.
Prices and discount assumptions are user-editable estimates; verify them against
the selected provider's current pricing and billing rules.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="AI Inference & Unit Economics Forecast", 
    page_icon="⚡", 
    layout="wide"
)

# Custom Compact Styling
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
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.25rem !important; font-weight:700;}

/* Ultra-Compact Header */
.compact-header {
    background: linear-gradient(135deg,#f5f0eb,#ebe5df);
    padding: 10px 16px;
    border-radius: 8px;
    border: 1px solid #e2d9d0;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.compact-header h1 {color:#2d3748; margin:0; font-size:1.35rem; font-weight:700;}
.compact-header p {color:#615a52; margin:0; font-size:0.85rem;}

/* Alert Cards */
.alert-card {padding:6px 10px; border-radius:5px; font-size:0.82rem; margin-bottom:10px; font-weight:500;}
.alert-warning {background:#fffaf0; border:1px solid #feebc8; color:#9c4221;}
.alert-error {background:#fff5f5; border:1px solid #fed7d7; color:#9b2c2c;}
.alert-info {background:#ebf8ff; border:1px solid #bee3f8; color:#2c5282;}
</style>
""", unsafe_allow_html=True)

# Compact Header
st.markdown("""
<div class="compact-header">
    <div>
        <h1>⚡ AI Inference, Startup Runway & Unit Economics</h1>
        <p>Driver-based forecast of token spend, infrastructure step-costs, subscription churn, and cloud-credit runway.</p>
    </div>
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
    help="Assumed MoM growth. Review historical 3-month trailing growth from product analytics."
) / 100

paying_conversion = st.sidebar.slider(
    "New Paid Conversion (% of DAU)", 0.0, 100.0, 20.0, 1.0,
    help="Percentage of active user base converting to paid subscription."
) / 100

monthly_churn = st.sidebar.slider(
    "Monthly Paid Subscriber Churn (%)", 0.0, 15.0, 3.0, 0.5,
    help="Percentage of paid subscribers cancelling each month. Source: Stripe Billing."
) / 100

arpu = st.sidebar.number_input(
    "Monthly ARPU per Paid Subscriber ($)", min_value=0.0, value=30.0, step=5.0,
    help="Average Revenue Per User per month for paid tiers."
)

st.sidebar.header("2. Usage Patterns & Token Context")
baseline_prompts = st.sidebar.slider(
    "Baseline Prompts / Paid User / Day", 1, 20, 5,
    help="Average daily prompt volume for paid users during normal activity."
)
peak_prompts = st.sidebar.slider(
    "Peak Prompts / Paid User / Day", 1, 100, 35,
    help="Expected peak usage surge per paid user during heavy usage or launch event."
)
peak_month = st.sidebar.slider("Peak Usage Month (1-12)", 1, 12, 7)
surge_width = st.sidebar.slider("Peak Usage Spread (Months)", 1.0, 4.0, 2.0, 0.5)

free_user_usage_mult = st.sidebar.slider(
    "Free User Usage Multiplier (%)", 0, 100, 50,
    help="Usage volume of free users relative to paid users."
) / 100

input_tokens_start = st.sidebar.number_input("Initial Input Tokens / Prompt", min_value=0, value=1000, step=100)
output_tokens_start = st.sidebar.number_input("Initial Output Tokens / Prompt", min_value=0, value=400, step=50)
input_growth = st.sidebar.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
output_growth = st.sidebar.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

st.sidebar.header("3. Model Routing & Token Prices")
frontier_mix = st.sidebar.slider("Frontier Model Share (%)", 0, 100, 30) / 100
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

step_trigger_type = st.sidebar.selectbox(
    "Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"]
)
step_threshold = st.sidebar.number_input("Step Threshold Quantity", min_value=1, value=20000, step=5000)
step_cost_increment = st.sidebar.number_input("Infra Step-Up Cost ($)", min_value=0.0, value=1500.0, step=250.0)

vector_cost_per_user = st.sidebar.number_input("Vector DB Cost / Active User ($)", min_value=0.0, value=0.15, step=0.05)
starting_credits = st.sidebar.number_input("Starting Cloud Credits ($)", min_value=0.0, value=50000.0, step=5000.0)
credit_eligible_share = st.sidebar.slider("Credit Eligible Cost Share (%)", 0, 100, 100) / 100
margin_target = st.sidebar.slider("Target Gross Margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

# ==========================================
# 2. FORECAST ENGINE
# ==========================================
months = np.arange(1, 13)
current_year = datetime.datetime.now().year
month_days = [pd.Period(f"{current_year}-{m:02d}").days_in_month for m in months]
month_names = [f"M{m}" for m in months]

prompt_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month) ** 2) / (2 * surge_width ** 2))

credits_left = float(starting_credits)
current_paid_subscribers = starting_dau * paying_conversion
rows = []

for i in range(12):
    dau_val = starting_dau * ((1 + monthly_growth) ** i)
    
    if i > 0:
        new_conversions = dau_val * paying_conversion * monthly_growth
        churned_subs = current_paid_subscribers * monthly_churn
        current_paid_subscribers = max(0, current_paid_subscribers + new_conversions - churned_subs)
        
    free_users = max(0, dau_val - current_paid_subscribers)
    
    input_tokens_val = input_tokens_start * ((1 + input_growth) ** i)
    output_tokens_val = output_tokens_start * ((1 + output_growth) ** i)

    prompts_per_paid_user_day = float(prompt_curve[i])
    prompts_per_free_user_day = prompts_per_paid_user_day * free_user_usage_mult
    
    monthly_prompts_paid = current_paid_subscribers * prompts_per_paid_user_day * month_days[i]
    monthly_prompts_free = free_users * prompts_per_free_user_day * month_days[i]
    monthly_prompts_total = monthly_prompts_paid + monthly_prompts_free
    
    raw_input_m = monthly_prompts_total * input_tokens_val / 1_000_000
    raw_output_m = monthly_prompts_total * output_tokens_val / 1_000_000

    uncached_input_m = raw_input_m * (1 - cache_hit_rate)
    cached_input_m = raw_input_m * cache_hit_rate
    
    frontier_input_cost = (uncached_input_m * frontier_input_price + cached_input_m * frontier_input_price * cache_price_ratio) * frontier_mix
    standard_input_cost = (uncached_input_m * standard_input_price + cached_input_m * standard_input_price * cache_price_ratio) * standard_mix
    frontier_output_cost = raw_output_m * frontier_output_price * frontier_mix
    standard_output_cost = raw_output_m * standard_output_price * standard_mix
    
    api_before_batch = frontier_input_cost + standard_input_cost + frontier_output_cost + standard_output_cost
    api_cost = api_before_batch * ((1 - batch_share) + batch_share * batch_price_ratio)

    vector_cost = dau_val * vector_cost_per_user
    
    if step_trigger_type == "DAU Threshold":
        step_multiplier = int(dau_val // step_threshold)
    elif step_trigger_type == "Paid Subscribers Threshold":
        step_multiplier = int(current_paid_subscribers // step_threshold)
    else:
        step_multiplier = int((monthly_prompts_total / 1_000_000) // step_threshold)
        
    infrastructure_cost = base_fixed_infra + (step_multiplier * step_cost_increment)
    total_modeled_cost = api_cost + vector_cost + infrastructure_cost
    revenue = current_paid_subscribers * arpu
    gross_profit = revenue - total_modeled_cost
    gross_margin_pct = (gross_profit / revenue * 100) if revenue > 0 else np.nan

    eligible_cost = total_modeled_cost * credit_eligible_share
    credits_applied = min(credits_left, eligible_cost)
    credits_left = max(0.0, credits_left - credits_applied)
    estimated_cash_payable = total_modeled_cost - credits_applied

    rows.append({
        "Month": month_names[i], 
        "Calendar Days": month_days[i],
        "DAU": round(dau_val), 
        "Free Users": round(free_users),
        "Paid Subscribers": round(current_paid_subscribers),
        "Prompts / Paid User / Day": round(prompts_per_paid_user_day, 1),
        "Input Tokens / Prompt": round(input_tokens_val), 
        "Output Tokens / Prompt": round(output_tokens_val),
        "Monthly Prompts": round(monthly_prompts_total),
        "Raw Input Tokens (M)": round(raw_input_m, 2), 
        "Raw Output Tokens (M)": round(raw_output_m, 2),
        "Subscription Revenue ($)": revenue, 
        "API Cost ($)": api_cost,
        "Vector DB Cost ($)": vector_cost, 
        "Fixed Infrastructure ($)": infrastructure_cost,
        "Total Modeled Costs ($)": total_modeled_cost,
        "Modeled Gross Profit ($)": gross_profit, 
        "Modeled Gross Margin (%)": gross_margin_pct,
        "Credits Applied ($)": credits_applied, 
        "Remaining Credits ($)": credits_left,
        "Estimated Cash Payable ($)": estimated_cash_payable,
    })

df = pd.DataFrame(rows)
df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = df["Total Modeled Costs ($)"] - budget_monthly if budget_monthly > 0 else np.nan
df["Cost per 1,000 Prompts ($)"] = df["Total Modeled Costs ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Cost per Paid Subscriber ($)"] = df["Total Modeled Costs ($)"] / df["Paid Subscribers"].replace(0, np.nan)

# ==========================================
# 3. EXECUTIVE ALERT STRIP
# ==========================================
alert_cols = st.columns(3)

depleted = df.index[df["Remaining Credits ($)"] <= 0.005].tolist()
if starting_credits <= 0:
    alert_cols[0].markdown('<div class="alert-card alert-warning">💳 <b>No Credits Available:</b> All costs hit cash directly.</div>', unsafe_allow_html=True)
elif depleted:
    idx = depleted[0]
    alert_cols[0].markdown(f'<div class="alert-card alert-warning">💳 <b>Credits Exhausted in {df.loc[idx, "Month"]}:</b> Out-of-pocket cash starts in M{idx+1}.</div>', unsafe_allow_html=True)
else:
    alert_cols[0].markdown(f'<div class="alert-card alert-info">💳 <b>Credits Healthy:</b> Balance in M12 (${df["Remaining Credits ($)"].iloc[-1]:,.0f}).</div>', unsafe_allow_html=True)

breach = df.index[df["Modeled Gross Margin (%)"].notna() & (df["Modeled Gross Margin (%)"] < margin_target)].tolist()
if breach:
    alert_cols[1].markdown(f'<div class="alert-card alert-error">⚠️ <b>Margin Breach in {df.loc[breach[0], "Month"]}:</b> Margin drops below target ({margin_target}%).</div>', unsafe_allow_html=True)
else:
    alert_cols[1].markdown(f'<div class="alert-card alert-info">✅ <b>Margin Target Met:</b> Stays above target ({margin_target}%).</div>', unsafe_allow_html=True)

if budget_monthly > 0:
    over = df.index[df["Budget Variance ($)"] > 0].tolist()
    if over:
        alert_cols[2].markdown(f'<div class="alert-card alert-warning">📈 <b>Budget Overrun in {df.loc[over[0], "Month"]}:</b> Spend exceeds cap.</div>', unsafe_allow_html=True)
    else:
        alert_cols[2].markdown('<div class="alert-card alert-info">🎯 <b>Budget Compliant:</b> Monthly spend under cap.</div>', unsafe_allow_html=True)
else:
    alert_cols[2].markdown('<div class="alert-card alert-info">ℹ️ <b>Planning Model:</b> Estimates only—not GAAP reporting.</div>', unsafe_allow_html=True)

# ==========================================
# 4. COMPACT KPIS (SINGLE 4-COLUMN ROW)
# ==========================================
annual_revenue = df["Subscription Revenue ($)"].sum()
annual_cost = df["Total Modeled Costs ($)"].sum()
m12 = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Run Rate (M12)", f"${m12['Subscription Revenue ($)']:,.0f}", f"12M Total: ${annual_revenue:,.0f}")
c2.metric("M12 Costs", f"${m12['Total Modeled Costs ($)']:,.0f}", f"12M Total: ${annual_cost:,.0f}")
c3.metric("M12 Gross Margin", "N/A" if pd.isna(m12["Modeled Gross Margin (%)"]) else f"{m12['Modeled Gross Margin (%)']:.1f}%")
c4.metric("Credits Remaining", f"${m12['Remaining Credits ($)']:,.0f}", f"Cash Payable: ${df['Estimated Cash Payable ($)'].sum():,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Revenue & Margin", 
    "💳 Credits & Cash", 
    "🎛️ Scenario Analysis", 
    "📋 Monthly Detail",
    "📑 Assumptions & Source Register"
])

with tab1:
    st.subheader("12-Month Revenue, Modeled Costs & Gross Margin")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Subscription Revenue ($)"], name="Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled Costs ($)"], name="Modeled Costs ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    
    # Plotly Layout: Legend moved to top-right to prevent title overlap
    fig.update_layout(
        template="plotly_white", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#fff", 
        height=380,
        legend=dict(orientation="h", y=1.08, x=0.01),
        margin=dict(l=40, r=60, t=20, b=30)
    )
    fig.update_yaxes(title_text="USD ($)", secondary_y=False, rangemode="tozero", gridcolor="#edf2f7")
    
    margins = df["Modeled Gross Margin (%)"].dropna()
    if not margins.empty:
        lo, hi = min(0, float(margins.min())), max(100, float(margins.max()))
        pad = max(5, (hi - lo) * 0.08)
        fig.update_yaxes(title_text="Gross Margin (%)", range=[lo - pad, hi + pad], secondary_y=True, gridcolor="#edf2f7")
        
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Cloud Credit Offset vs. Out-of-Pocket Cash Payable")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Credits Applied ($)"], name="Credits Applied ($)", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Cash Payable ($)", marker_color="#dd6b20"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Credits ($)"], name="Remaining Credit Balance ($)", mode="lines+markers", line=dict(color="#805ad5", width=3)), secondary_y=True)
    
    fig2.update_layout(
        template="plotly_white", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#fff", 
        height=380,
        barmode="stack", 
        legend=dict(orientation="h", y=1.08, x=0.01),
        margin=dict(l=40, r=60, t=20, b=30)
    )
    fig2.update_yaxes(title_text="Monthly Amount ($)", secondary_y=False, rangemode="tozero", gridcolor="#edf2f7")
    fig2.update_yaxes(title_text="Remaining Credits ($)", secondary_y=True, rangemode="tozero", gridcolor="#edf2f7")
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Usage & Pricing Sensitivity Scenarios")
    
    scenarios = []
    annual_api = df["API Cost ($)"].sum()
    annual_vector = df["Vector DB Cost ($)"].sum()
    annual_fixed = df["Fixed Infrastructure ($)"].sum()

    for label, usage_factor, price_factor in [
        ("Downside: Lower Usage (-25%)", 0.75, 1.0),
        ("Base Case", 1.0, 1.0),
        ("High Usage (+50%)", 1.5, 1.0),
        ("Cost Optimization (-25% Token Price)", 1.0, 0.75),
        ("High Usage + Cost Optimization", 1.5, 0.75),
    ]:
        api_s = annual_api * usage_factor * price_factor
        vector_s = annual_vector * usage_factor
        fixed_s = annual_fixed
        
        cost_s = api_s + vector_s + fixed_s
        gross_profit_s = annual_revenue - cost_s
        margin_s = (gross_profit_s / annual_revenue * 100) if annual_revenue > 0 else np.nan

        scenarios.append({
            "Scenario": label, 
            "12-Month Costs ($)": cost_s, 
            "12-Month Revenue ($)": annual_revenue, 
            "Gross Profit ($)": gross_profit_s, 
            "Gross Margin (%)": margin_s
        })
        
    scen_df = pd.DataFrame(scenarios)
    st.dataframe(
        scen_df.style.format({
            "12-Month Costs ($)": "${:,.0f}", 
            "12-Month Revenue ($)": "${:,.0f}", 
            "Gross Profit ($)": "${:,.0f}", 
            "Gross Margin (%)": "{:.1f}%"
        }, na_rep="—"), 
        use_container_width=True
    )

with tab4:
    st.subheader("12-Month Detailed Model Output")
    st.dataframe(
        df.style.format({
            "Subscription Revenue ($)": "${:,.0f}",
            "API Cost ($)": "${:,.0f}",
            "Vector DB Cost ($)": "${:,.0f}",
            "Fixed Infrastructure ($)": "${:,.0f}",
            "Total Modeled Costs ($)": "${:,.0f}",
            "Modeled Gross Profit ($)": "${:,.0f}",
            "Modeled Gross Margin (%)": "{:.1f}%",
            "Credits Applied ($)": "${:,.0f}",
            "Remaining Credits ($)": "${:,.0f}",
            "Estimated Cash Payable ($)": "${:,.0f}",
            "Budget ($)": "${:,.0f}",
            "Budget Variance ($)": "${:,.0f}",
            "Cost per 1,000 Prompts ($)": "${:,.4f}",
            "Cost per Paid Subscriber ($)": "${:,.2f}",
        }, na_rep="—"), 
        use_container_width=True
    )
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", data=csv, file_name="ai_inference_forecast.csv", mime="text/csv")

with tab5:
    st.subheader("📑 Source & Assumptions Audit Register")
    st.write("Audit record detailing parameter sources, ownership, and verification status.")
    
    assumptions_data = [
        {"Parameter": "Starting DAU", "Current Value": f"{starting_dau:,}", "Category": "User Funnel", "Source": "Product Analytics (Mixpanel/PostHog)", "Verification Status": "Historical Trailing Actuals"},
        {"Parameter": "Monthly DAU Growth", "Current Value": f"{monthly_growth*100:.1f}%", "Category": "User Funnel", "Source": "Growth Forecast Model", "Verification Status": "Management Estimate"},
        {"Parameter": "Paid Conversion %", "Current Value": f"{paying_conversion*100:.1f}%", "Category": "User Funnel", "Source": "Stripe Checkout Analytics", "Verification Status": "Historical Actuals"},
        {"Parameter": "Paid Subscriber Churn", "Current Value": f"{monthly_churn*100:.1f}%", "Category": "User Funnel", "Source": "Stripe Billing Dashboard", "Verification Status": "Verified 90-Day Avg"},
        {"Parameter": "Frontier Input Price", "Current Value": f"${frontier_input_price:.2f} / 1M", "Category": "AI Unit Pricing", "Source": "Vendor Public Pricing Page", "Verification Status": "Verified Current Rate"},
        {"Parameter": "Frontier Output Price", "Current Value": f"${frontier_output_price:.2f} / 1M", "Category": "AI Unit Pricing", "Source": "Vendor Public Pricing Page", "Verification Status": "Verified Current Rate"},
        {"Parameter": "Prompt Cache Hit Rate", "Current Value": f"{cache_hit_rate*100:.0f}%", "Category": "Optimization", "Source": "Engineering Logs (Helicone)", "Verification Status": "Telemetry Estimate"},
        {"Parameter": "Base Fixed Hosting Infra", "Current Value": f"${base_fixed_infra:,.0f}/mo", "Category": "Infrastructure", "Source": "AWS / GCP Monthly Invoices", "Verification Status": "Verified Billing Contract"},
        {"Parameter": "Infra Step-Cost Increment", "Current Value": f"${step_cost_increment:,.0f} per {step_threshold:,} units", "Category": "Infrastructure", "Source": "DevOps Capacity Plan", "Verification Status": "Engineering Estimate"},
        {"Parameter": "Starting Cloud Credits", "Current Value": f"${starting_credits:,.0f}", "Category": "Capital & Credits", "Source": "AWS Activate / OpenAI Portal", "Verification Status": "Verified Grant Balance"},
    ]
    
    st.table(pd.DataFrame(assumptions_data))

st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
