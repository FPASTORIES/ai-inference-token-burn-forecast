"""
AI Inference, Startup Runway & COGS Unit Economics Model
--------------------------------------------------------
Copyright (c) 2026. Released under the MIT License.

Description: Enterprise FP&A dashboard modeling non-linear token burn, 
prompt caching, model routing, vector database infrastructure COGS, 
and cloud credit cash runway depletion with custom soft beige styling.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set Page Config
st.set_page_config(
    page_title="AI Startup COGS & Runway Model", 
    page_icon="⚡", 
    layout="wide"
)

# ==========================================
# CUSTOM CSS STYLING (SOFT BEIGE & NEUTRAL PALETTE)
# ==========================================
st.markdown("""
<style>
    /* Main Background & Soft Neutral Canvas */
    .stApp {
        background-color: #fcfbf9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #2d3748;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(135deg, #f5f0eb 0%, #ebe5df 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2d9d0;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .main-header h1 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #615a52;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f5f2ed !important;
        border-right: 1px solid #e5dfd7;
    }

    /* Custom Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2d9d0;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetric"] label {
        color: #615a52 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2b6cb0 !important;
        font-weight: 700;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #f0eae1;
        border-radius: 8px 8px 0px 0px;
        padding-left: 16px;
        padding-right: 16px;
        color: #615a52;
        border: 1px solid #e2d9d0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2b6cb0 !important;
        font-weight: 600;
        border-bottom: 2px solid #2b6cb0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1>⚡ AI Startup COGS, Token Burn & Cash Runway Model</h1>
    <p>A GAAP-compliant financial engine modeling LLM API optimization, vector search infrastructure, and cloud credit burn.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR ASSUMPTIONS & CONTROLS
# ==========================================

st.sidebar.header("1. Scale & User Base")
initial_dau = st.sidebar.number_input("Starting DAU", value=5000, step=500)
monthly_growth_rate = st.sidebar.slider("MoM DAU Growth Rate (%)", min_value=0.0, max_value=25.0, value=8.0) / 100
arpu_monthly = st.sidebar.number_input("Monthly Subscription ARPU ($)", value=30.00)

st.sidebar.header("2. Usage & Context Scaling")
baseline_prompts = st.sidebar.slider("Baseline Prompts / User / Day", min_value=1, max_value=20, value=5)
peak_prompts = st.sidebar.slider("Peak Prompts / User / Day (Surge)", min_value=10, max_value=100, value=35)
peak_month = st.sidebar.slider("Peak Usage Month", min_value=1, max_value=12, value=7)
surge_std_dev = st.sidebar.slider("Surge Duration Width (Std Dev)", min_value=1.0, max_value=4.0, value=2.0)

avg_input_tokens = st.sidebar.number_input("Initial Input Tokens / Prompt", value=1000, step=100)
avg_output_tokens = st.sidebar.number_input("Initial Output Tokens / Prompt", value=400, step=50)
context_growth_rate = st.sidebar.slider("MoM Context Expansion Rate (%)", min_value=0.0, max_value=15.0, value=3.0) / 100

st.sidebar.header("3. API Optimization & Routing")
frontier_ratio = st.sidebar.slider("Frontier Model Traffic Blend (%)", min_value=0, max_value=100, value=30) / 100
standard_ratio = 1.0 - frontier_ratio

cache_hit_rate = st.sidebar.slider("Prompt Cache Hit Rate (%)", min_value=0, max_value=90, value=40) / 100
cache_discount = 0.80

batch_volume_rate = st.sidebar.slider("Batch API Processing Share (%)", min_value=0, max_value=80, value=20) / 100
batch_discount = 0.50

st.sidebar.header("4. Infra COGS & Credit Runway")
fixed_infra_monthly = st.sidebar.number_input("Fixed Cloud & GPU Infra ($/mo)", value=2500.0, step=500.0)
vector_db_cost_per_mau = st.sidebar.number_input("Vector DB / Search Cost / MAU ($)", value=0.15, step=0.05)
starting_cloud_credits = st.sidebar.number_input("Starting Cloud Credit Balance ($)", value=50000.0, step=5000.0)

min_margin_target = st.sidebar.slider("Target Minimum Gross Margin (%)", min_value=30, max_value=85, value=60)

# ==========================================
# MODEL CALCULATIONS
# ==========================================

months = np.arange(1, 13)
month_labels = [f"M{i}" for i in months]

frontier_input_rate, frontier_output_rate = 2.50, 10.00
standard_input_rate, standard_output_rate = 0.15, 0.60

prompts_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month)**2) / (2 * surge_std_dev**2))

data = []
current_dau = initial_dau
current_input_tokens = avg_input_tokens
remaining_credits = starting_cloud_credits

for idx, m in enumerate(months):
    if idx > 0:
        current_dau *= (1 + monthly_growth_rate)
        current_input_tokens *= (1 + context_growth_rate)
        
    daily_prompts_per_user = prompts_curve[idx]
    monthly_prompts_per_user = daily_prompts_per_user * 30
    total_monthly_prompts = current_dau * monthly_prompts_per_user
    
    raw_input_tokens = (total_monthly_prompts * current_input_tokens) / 1_000_000
    raw_output_tokens = (total_monthly_prompts * avg_output_tokens) / 1_000_000
    
    effective_input_multiplier = (1.0 - (cache_hit_rate * cache_discount)) * (1.0 - (batch_volume_rate * batch_discount))
    effective_output_multiplier = (1.0 - (batch_volume_rate * batch_discount))
    
    effective_input_tokens = raw_input_tokens * effective_input_multiplier
    effective_output_tokens = raw_output_tokens * effective_output_multiplier
    
    frontier_cost = (effective_input_tokens * frontier_input_rate + effective_output_tokens * frontier_output_rate) * frontier_ratio
    standard_cost = (effective_input_tokens * standard_input_rate + effective_output_tokens * standard_output_rate) * standard_ratio
    total_api_cost = frontier_cost + standard_cost
    
    vector_db_cost = current_dau * vector_db_cost_per_mau
    total_cogs = total_api_cost + vector_db_cost + fixed_infra_monthly
    
    gross_revenue = current_dau * arpu_monthly
    gross_margin = gross_revenue - total_cogs
    gross_margin_pct = (gross_margin / gross_revenue) * 100 if gross_revenue > 0 else 0
    
    credit_applied = min(remaining_credits, total_cogs)
    remaining_credits -= credit_applied
    out_of_pocket_cash_cogs = total_cogs - credit_applied

    data.append({
        "Month": month_labels[idx],
        "DAU": int(current_dau),
        "Input Context/Prompt": int(current_input_tokens),
        "Prompts/User/Day": round(daily_prompts_per_user, 1),
        "Gross Revenue ($)": round(gross_revenue, 2),
        "API Spend ($)": round(total_api_cost, 2),
        "Vector DB Cost ($)": round(vector_db_cost, 2),
        "Fixed Infra ($)": round(fixed_infra_monthly, 2),
        "Total GAAP COGS ($)": round(total_cogs, 2),
        "Gross Margin ($)": round(gross_margin, 2),
        "Gross Margin (%)": round(gross_margin_pct, 1),
        "Credits Used ($)": round(credit_applied, 2),
        "Remaining Credits ($)": round(remaining_credits, 2),
        "Out-of-Pocket Cash Spend ($)": round(out_of_pocket_cash_cogs, 2)
    })

df = pd.DataFrame(data)

# ==========================================
# ALERTS & METRICS
# ==========================================

credit_depletion_df = df[df["Remaining Credits ($)"] == 0]
first_out_of_pocket_month = credit_depletion_df["Month"].iloc[0] if not credit_depletion_df.empty else None

margin_breach_df = df[df["Gross Margin (%)"] < min_margin_target]
first_breach_month = margin_breach_df["Month"].iloc[0] if not margin_breach_df.empty else None

if first_out_of_pocket_month:
    st.warning(f"💳 **CLOUD CREDIT EXPIRATION:** Cloud credits fully burn out in **{first_out_of_pocket_month}**. Cash spend hits bank account directly starting M{first_out_of_pocket_month}.")

if first_breach_month:
    st.error(f"⚠️ **MARGIN COMPRESSION WARNING:** GAAP Gross Margin falls below {min_margin_target}% threshold starting in **{first_breach_month}**.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Run-Rate Revenue (M12)", f"${df['Gross Revenue ($)'].iloc[-1]:,.0f}")
col2.metric("M12 Total GAAP COGS", f"${df['Total GAAP COGS ($)'].iloc[-1]:,.0f}")
col3.metric("M12 Gross Margin %", f"{df['Gross Margin (%)'].iloc[-1]}%")
col4.metric("Credit Balance (M12)", f"${df['Remaining Credits ($)'].iloc[-1]:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# VISUALIZATIONS WITH LIGHT NEUTRAL TEMPLATE
# ==========================================

tab1, tab2, tab3 = st.tabs(["📉 Revenue, GAAP COGS & Gross Margin", "💳 Cloud Credit Depletion & Cash Burn", "📋 Full P&L Breakdown Table"])

with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=df["Month"], y=df["Gross Revenue ($)"], name="Gross Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total GAAP COGS ($)"], name="Total GAAP COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=4)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Gross Margin (%)"], name="Gross Margin (%)", mode="lines+markers", line=dict(color="#28a745", width=3, dash="dash")), secondary_y=True)
    
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        title="12-Month GAAP Revenue vs. Total COGS (API + Vector DB + Hosting)",
        xaxis_title="Month",
        legend=dict(x=0.01, y=0.99)
    )
    fig.update_yaxes(title_text="USD ($)", secondary_y=False, gridcolor="#edf2f7")
    fig.update_yaxes(title_text="Gross Margin (%)", range=[0, 100], secondary_y=True, gridcolor="#edf2f7")
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_credit = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_credit.add_trace(go.Bar(x=df["Month"], y=df["Credits Used ($)"], name="Cloud Credits Applied ($)", marker_color="#38a169"), secondary_y=False)
    fig_credit.add_trace(go.Bar(x=df["Month"], y=df["Out-of-Pocket Cash Spend ($)"], name="Out-of-Pocket Cash COGS ($)", marker_color="#dd6b20"), secondary_y=False)
    fig_credit.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Credits ($)"], name="Remaining Credit Balance ($)", mode="lines+markers", line=dict(color="#805ad5", width=3)), secondary_y=True)
    
    fig_credit.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        title="Cloud Credit Cushion vs. Out-of-Pocket Cash Spend",
        barmode="stack",
        xaxis_title="Month"
    )
    fig_credit.update_yaxes(title_text="Monthly Cost ($)", secondary_y=False, gridcolor="#edf2f7")
    fig_credit.update_yaxes(title_text="Credit Balance ($)", secondary_y=True, gridcolor="#edf2f7")
    
    st.plotly_chart(fig_credit, use_container_width=True)

with tab3:
    st.subheader("Complete 12-Month Financial Output")
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and Startup CFOs evaluating AI unit economics.")
