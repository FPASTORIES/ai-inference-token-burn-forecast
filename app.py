"""
AI Inference & Token Burn Rate Forecast Model
---------------------------------------------
Copyright (c) 2026. Released under the MIT License.

Description: Streamlit model simulating non-linear token usage surges, 
margin compression cliffs, CRO pricing action thresholds, and a dynamic 
pricing mitigation simulator.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="AI Inference & Token Burn Forecast", layout="wide")

st.title("⚡ AI Inference & Margin Erosion Model")
st.caption("Modeling non-linear token usage surges, gross margin compression, CRO pricing action thresholds, and hybrid dynamic pricing recovery.")

# --- SIDEBAR ASSUMPTIONS ---
st.sidebar.header("1. User Base Growth")
initial_dau = st.sidebar.number_input("Starting DAU", value=5000, step=500)
monthly_growth_rate = st.sidebar.slider("MoM DAU Growth Rate (%)", min_value=0.0, max_value=25.0, value=10.0) / 100
arpu_monthly = st.sidebar.number_input("Monthly Subscription Price / User ($)", value=30.00)

st.sidebar.header("2. Usage Surge Dynamics (Bell Curve)")
baseline_prompts = st.sidebar.slider("Baseline Prompts / User / Day", min_value=1, max_value=20, value=5)
peak_prompts = st.sidebar.slider("Peak Prompts / User / Day (Surge)", min_value=10, max_value=100, value=45)
peak_month = st.sidebar.slider("Peak Usage Month (Bell Curve Center)", min_value=1, max_value=12, value=7)
surge_std_dev = st.sidebar.slider("Surge Duration Width (Std Dev)", min_value=1.0, max_value=4.0, value=2.0)

st.sidebar.header("3. Context & Token Pricing")
avg_input_tokens = st.sidebar.number_input("Avg Input Tokens / Prompt", value=1200, step=100)
avg_output_tokens = st.sidebar.number_input("Avg Output Tokens / Prompt", value=500, step=50)

model_choice = st.sidebar.selectbox(
    "LLM Pricing Tier",
    ["Frontier Tier (e.g., GPT-4o)", "Standard Tier (e.g., GPT-4o-mini)", "Custom Pricing"]
)

if model_choice == "Frontier Tier (e.g., GPT-4o)":
    cost_per_m_input = 2.50
    cost_per_m_output = 10.00
elif model_choice == "Standard Tier (e.g., GPT-4o-mini)":
    cost_per_m_input = 0.15
    cost_per_m_output = 0.60
else:
    cost_per_m_input = st.sidebar.number_input("Cost / 1M Input Tokens ($)", value=2.00)
    cost_per_m_output = st.sidebar.number_input("Cost / 1M Output Tokens ($)", value=8.00)

min_margin_target = st.sidebar.slider("Target Minimum Gross Margin (%)", min_value=30, max_value=85, value=60)

# --- DYNAMIC PRICING SIMULATOR TOGGLE ---
st.sidebar.header("4. CRO Pricing Mitigation Strategy")
enable_dynamic_pricing = st.sidebar.checkbox("Enable Dynamic Pricing / Overage Fee", value=False)
included_prompts_limit = st.sidebar.number_input("Included Monthly Prompts / User Limit", value=300, step=50) if enable_dynamic_pricing else 0
overage_fee_per_prompt = st.sidebar.number_input("Overage Fee / Prompt ($)", value=0.05, step=0.01) if enable_dynamic_pricing else 0.0

# --- MODEL CALCULATIONS ---
months = np.arange(1, 13)
month_labels = [f"M{i}" for i in months]

# Gaussian curve for prompts per user
prompts_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month)**2) / (2 * surge_std_dev**2))

data = []
current_dau = initial_dau

for idx, m in enumerate(months):
    if idx > 0:
        current_dau = current_dau * (1 + monthly_growth_rate)
    
    daily_prompts_per_user = prompts_curve[idx]
    monthly_prompts_per_user = daily_prompts_per_user * 30
    total_monthly_prompts = current_dau * monthly_prompts_per_user
    
    total_input_tokens = (total_monthly_prompts * avg_input_tokens) / 1_000_000
    total_output_tokens = (total_monthly_prompts * avg_output_tokens) / 1_000_000
    
    input_cost = total_input_tokens * cost_per_m_input
    output_cost = total_output_tokens * cost_per_m_output
    total_api_cost = input_cost + output_cost
    
    # Revenue Calculation (Base ARPU + Overage Fees)
    base_revenue = current_dau * arpu_monthly
    overage_revenue = 0.0
    
    if enable_dynamic_pricing and monthly_prompts_per_user > included_prompts_limit:
        overage_prompts_per_user = monthly_prompts_per_user - included_prompts_limit
        overage_revenue = current_dau * overage_prompts_per_user * overage_fee_per_prompt
        
    gross_revenue = base_revenue + overage_revenue
    gross_margin = gross_revenue - total_api_cost
    gross_margin_pct = (gross_margin / gross_revenue) * 100 if gross_revenue > 0 else 0
    cost_per_mau = total_api_cost / current_dau if current_dau > 0 else 0

    data.append({
        "Month": month_labels[idx],
        "Month_Num": m,
        "DAU": int(current_dau),
        "Prompts/User/Day": round(daily_prompts_per_user, 1),
        "Base Revenue ($)": round(base_revenue, 2),
        "Overage Revenue ($)": round(overage_revenue, 2),
        "Gross Revenue ($)": round(gross_revenue, 2),
        "API Cost ($)": round(total_api_cost, 2),
        "Gross Margin ($)": round(gross_margin, 2),
        "Gross Margin (%)": round(gross_margin_pct, 1),
        "AI Cost / User / Mo ($)": round(cost_per_mau, 2)
    })

df = pd.DataFrame(data)

# --- TRIGGER DETECTION ---
margin_breach_df = df[df["Gross Margin (%)"] < min_margin_target]
crossover_df = df[df["API Cost ($)"] >= df["Gross Revenue ($)"]]

first_breach_month = margin_breach_df["Month"].iloc[0] if not margin_breach_df.empty else None
first_crossover_month = crossover_df["Month"].iloc[0] if not crossover_df.empty else None

# --- EXECUTIVE ALERT BANNER ---
if first_crossover_month:
    st.error(f"🚨 **CRITICAL CRO ALERT:** API costs exceed total revenue in **{first_crossover_month}**! Flat-rate pricing is cash-negative. Enable Dynamic Pricing in the sidebar.")
elif first_breach_month:
    st.warning(f"⚠️ **MARGIN COMPRESSION WARNING:** Gross margin drops below target threshold ({min_margin_target}%) starting in **{first_breach_month}** due to peak prompt volume.")
else:
    st.success(f"✅ **HEALTHY UNIT ECONOMICS:** Gross margins remain above {min_margin_target}% across all 12 months under current usage assumptions.")

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
peak_idx = df["API Cost ($)"].idxmax()
col1.metric("Peak API Cost Month", f"{df.loc[peak_idx, 'Month']} (${df.loc[peak_idx, 'API Cost ($)']:,.0f})")
col2.metric("Peak Usage (Prompts/User)", f"{df['Prompts/User/Day'].max():.0f} / day")
col3.metric("Lowest Gross Margin %", f"{df['Gross Margin (%)'].min()}%")
col4.metric("Max AI Cost / User", f"${df['AI Cost / User / Mo ($)'].max()}/mo")

st.markdown("---")

# --- VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs(["📉 Revenue vs. Cost Trajectory & Gross Margin Line", "📊 Usage Surge Dynamics", "📋 Monthly Financial Breakdown"])

with tab1:
    # Subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Gross Revenue Bar (Secondary Y: Left Axis)
    fig.add_trace(
        go.Bar(
            x=df["Month"], y=df["Gross Revenue ($)"],
            name="Gross Revenue ($)", marker_color="#1f77b4"
        ),
        secondary_y=False
    )
    
    # API Cost Line (Secondary Y: Left Axis)
    fig.add_trace(
        go.Scatter(
            x=df["Month"], y=df["API Cost ($)"],
            name="AI API Cost ($)", mode="lines+markers",
            line=dict(color="#d62728", width=4)
        ),
        secondary_y=False
    )
    
    # Gross Margin % Line (Secondary Y: Right Axis)
    fig.add_trace(
        go.Scatter(
            x=df["Month"], y=df["Gross Margin (%)"],
            name="Gross Margin (%)", mode="lines+markers",
            line=dict(color="#2ca02c", width=3, dash="dash")
        ),
        secondary_y=True
    )
    
    # Highlight Breach Point
    if first_breach_month:
        breach_row = df[df["Month"] == first_breach_month].iloc[0]
        fig.add_annotation(
            x=first_breach_month, y=breach_row["API Cost ($)"],
            text=f"⚠️ Margin < {min_margin_target}%",
            showarrow=True, arrowhead=2, ax=0, ay=-50,
            bgcolor="#ff7f0e", bordercolor="black", font=dict(color="white", size=12)
        )
        
    fig.update_layout(
        title="12-Month Financial Performance (Revenue, Cost & Gross Margin %)",
        xaxis_title="Month",
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_yaxes(title_text="USD ($)", secondary_y=False)
    fig.update_yaxes(title_text="Gross Margin (%)", range=[0, 100], secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_surge = px.line(
            df, x="Month", y="Prompts/User/Day", markers=True,
            title="User Engagement Curve (Prompts / User / Day)",
            color_discrete_sequence=["#9467bd"]
        )
        st.plotly_chart(fig_surge, use_container_width=True)
        
    with col_b:
        fig_margin = px.line(
            df, x="Month", y="Gross Margin (%)", markers=True,
            title="Gross Margin % Compression View",
            color_discrete_sequence=["#2ca02c"]
        )
        fig_margin.add_hline(y=min_margin_target, line_dash="dash", line_color="red", annotation_text=f"Target ({min_margin_target}%)")
        st.plotly_chart(fig_margin, use_container_width=True)

with tab3:
    st.subheader("Full 12-Month Financial Output")
    st.dataframe(
        df[["Month", "DAU", "Prompts/User/Day", "Base Revenue ($)", "Overage Revenue ($)", "Gross Revenue ($)", "API Cost ($)", "Gross Margin ($)", "Gross Margin (%)", "AI Cost / User / Mo ($)"]],
        use_container_width=True
    )

st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and CROs evaluating AI unit economics.")
