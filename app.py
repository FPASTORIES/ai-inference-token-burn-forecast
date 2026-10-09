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

# Custom Styling
st.markdown("""
<style>
.stApp {background:#fcfbf9;color:#2d3748}
section[data-testid="stSidebar"] {background:#f5f2ed}
div[data-testid="stMetric"] {background:#fff;border:1px solid #e2d9d0;padding:14px;border-radius:10px}
.main-header {background:linear-gradient(135deg,#f5f0eb,#ebe5df);padding:22px;border-radius:12px;border:1px solid #e2d9d0;margin-bottom:18px}
.main-header h1 {color:#2d3748;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
<h1>⚡ AI Inference, Startup Runway & Unit Economics</h1>
<p>Driver-based forecast of token spend, infrastructure costs, subscription economics and cloud-credit usage.</p>
</div>
""", unsafe_allow_html=True)

st.info("Planning model only. Revenue, costs, credits and gross margin are estimates based on the assumptions below—not GAAP reporting or a provider bill.")

# ==========================================
# 1. SIDEBAR INPUTS
# ==========================================
st.sidebar.header("1. Users & Revenue")
starting_dau = st.sidebar.number_input("Starting daily active users (DAU)", min_value=0, value=5000, step=500)
paying_conversion = st.sidebar.slider("Paying subscribers as % of DAU", 0.0, 100.0, 20.0, 1.0) / 100
monthly_growth = st.sidebar.slider("Monthly DAU growth (%)", 0.0, 25.0, 8.0, 0.5) / 100
arpu = st.sidebar.number_input("Monthly revenue per paying subscriber ($)", min_value=0.0, value=30.0, step=5.0)
days_basis = st.sidebar.selectbox("Days per month assumption", ["Actual calendar days", "30-day planning month"])

st.sidebar.header("2. Usage & Context")
baseline_prompts = st.sidebar.slider("Baseline prompts / user / day", 1, 20, 5)
peak_prompts = st.sidebar.slider("Peak prompts / user / day", 1, 100, 35)
peak_month = st.sidebar.slider("Peak usage month", 1, 12, 7)
surge_width = st.sidebar.slider("Peak usage spread (months)", 1.0, 4.0, 2.0, 0.5)
input_tokens_start = st.sidebar.number_input("Initial input tokens / prompt", min_value=0, value=1000, step=100)
output_tokens_start = st.sidebar.number_input("Initial output tokens / prompt", min_value=0, value=400, step=50)
input_growth = st.sidebar.slider("Monthly input-token growth (%)", 0.0, 20.0, 3.0, 0.5) / 100
output_growth = st.sidebar.slider("Monthly output-token growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

st.sidebar.header("3. Model Mix & Token Prices")
frontier_mix = st.sidebar.slider("Frontier model share (%)", 0, 100, 30) / 100
standard_mix = 1.0 - frontier_mix
frontier_input_price = st.sidebar.number_input("Frontier input price ($ / 1M tokens)", min_value=0.0, value=2.50, step=0.25)
frontier_output_price = st.sidebar.number_input("Frontier output price ($ / 1M tokens)", min_value=0.0, value=10.00, step=0.50)
standard_input_price = st.sidebar.number_input("Standard input price ($ / 1M tokens)", min_value=0.0, value=0.15, step=0.05, format="%.4f")
standard_output_price = st.sidebar.number_input("Standard output price ($ / 1M tokens)", min_value=0.0, value=0.60, step=0.05, format="%.4f")
cache_hit_rate = st.sidebar.slider("Eligible input tokens served from cache (%)", 0, 90, 40) / 100
cache_price_ratio = st.sidebar.slider("Cached input price as % of regular input price", 0, 100, 20) / 100
batch_share = st.sidebar.slider("Eligible traffic using batch pricing (%)", 0, 80, 20) / 100
batch_price_ratio = st.sidebar.slider("Batch price as % of standard price", 0, 100, 50) / 100
st.sidebar.caption("Simplified assumption: cache and batch eligibility are independent. Batch discount is applied to the blended eligible token price.")

st.sidebar.header("4. Infrastructure & Credits")
fixed_infra = st.sidebar.number_input("Fixed cloud / GPU infrastructure ($ per month)", min_value=0.0, value=2500.0, step=500.0)
vector_cost_per_user = st.sidebar.number_input("Vector DB / search cost ($ per active user / month)", min_value=0.0, value=0.15, step=0.05)
starting_credits = st.sidebar.number_input("Starting eligible cloud credits ($)", min_value=0.0, value=50000.0, step=5000.0)
credit_eligible_share = st.sidebar.slider("Share of modeled costs eligible for credits (%)", 0, 100, 100) / 100
margin_target = st.sidebar.slider("Target gross margin (%)", 0, 90, 60)
budget_monthly = st.sidebar.number_input("Monthly inference + infrastructure budget ($; 0 = no budget)", min_value=0.0, value=0.0, step=1000.0)

# ==========================================
# 2. FORECAST ENGINE
# ==========================================
months = np.arange(1, 13)
current_year = datetime.datetime.now().year

# Dynamic year calculation for actual calendar days
month_days = [pd.Period(f"{current_year}-{m:02d}").days_in_month for m in months] if days_basis == "Actual calendar days" else [30] * 12
month_names = [f"M{m}" for m in months]
prompt_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month) ** 2) / (2 * surge_width ** 2))

credits_left = float(starting_credits)
rows = []

for i in range(12):
    # Explicit compounding per month
    dau_val = starting_dau * ((1 + monthly_growth) ** i)
    input_tokens_val = input_tokens_start * ((1 + input_growth) ** i)
    output_tokens_val = output_tokens_start * ((1 + output_growth) ** i)

    paying_subscribers = dau_val * paying_conversion
    prompts_per_user_day = float(prompt_curve[i])
    monthly_prompts = dau_val * prompts_per_user_day * month_days[i]
    
    raw_input_m = monthly_prompts * input_tokens_val / 1_000_000
    raw_output_m = monthly_prompts * output_tokens_val / 1_000_000

    # Split input tokens into uncached and cached portions
    uncached_input_m = raw_input_m * (1 - cache_hit_rate)
    cached_input_m = raw_input_m * cache_hit_rate
    
    frontier_input_cost = (uncached_input_m * frontier_input_price + cached_input_m * frontier_input_price * cache_price_ratio) * frontier_mix
    standard_input_cost = (uncached_input_m * standard_input_price + cached_input_m * standard_input_price * cache_price_ratio) * standard_mix
    frontier_output_cost = raw_output_m * frontier_output_price * frontier_mix
    standard_output_cost = raw_output_m * standard_output_price * standard_mix
    
    api_before_batch = frontier_input_cost + standard_input_cost + frontier_output_cost + standard_output_cost
    api_cost = api_before_batch * ((1 - batch_share) + batch_share * batch_price_ratio)

    vector_cost = dau_val * vector_cost_per_user
    infrastructure_cost = fixed_infra
    total_modeled_cost = api_cost + vector_cost + infrastructure_cost
    revenue = paying_subscribers * arpu
    gross_profit = revenue - total_modeled_cost
    gross_margin_pct = (gross_profit / revenue * 100) if revenue > 0 else np.nan

    # Credits offset only the eligible share of modeled costs
    eligible_cost = total_modeled_cost * credit_eligible_share
    credits_applied = min(credits_left, eligible_cost)
    credits_left = max(0.0, credits_left - credits_applied)
    estimated_cash_payable = total_modeled_cost - credits_applied

    rows.append({
        "Month": month_names[i], 
        "Calendar Days": month_days[i],
        "DAU": round(dau_val), 
        "Estimated Paying Subscribers": round(paying_subscribers),
        "Prompts / User / Day": round(prompts_per_user_day, 1),
        "Input Tokens / Prompt": round(input_tokens_val), 
        "Output Tokens / Prompt": round(output_tokens_val),
        "Monthly Prompts": round(monthly_prompts),
        "Raw Input Tokens (M)": round(raw_input_m, 2), 
        "Raw Output Tokens (M)": round(raw_output_m, 2),
        "Subscription Revenue ($)": revenue, 
        "API Cost ($)": api_cost,
        "Vector DB / Search ($)": vector_cost, 
        "Fixed Infrastructure ($)": infrastructure_cost,
        "Modeled Inference + Infrastructure Cost ($)": total_modeled_cost,
        "Modeled Gross Profit ($)": gross_profit, 
        "Modeled Gross Margin (%)": gross_margin_pct,
        "Credits Applied ($)": credits_applied, 
        "Remaining Credits ($)": credits_left,
        "Estimated Cash Payable ($)": estimated_cash_payable,
    })

df = pd.DataFrame(rows)
df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] - budget_monthly if budget_monthly > 0 else np.nan
df["Cost per 1,000 Prompts ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Cost per Paying Subscriber ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] / df["Estimated Paying Subscribers"].
