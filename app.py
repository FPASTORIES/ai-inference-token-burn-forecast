"""
AI Inference & Token Burn Rate Forecast Model
---------------------------------------------
Copyright (c) 2026. Released under the MIT License.

Description: Advanced Streamlit model simulating non-linear token usage surges, 
margin compression cliffs, and strategic CRO pricing triggers.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI Inference & Token Burn Forecast", layout="wide")

st.title("⚡ AI Inference & Margin Erosion Model")
st.caption("Modeling non-linear token usage surges, gross margin compression, and CRO pricing action thresholds.")

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

# --- MODEL CALCULATIONS ---
months = np.arange(1, 13)
month_labels = [f"M{i}" for i in months]

# Gaussian curve for prompts per user: baseline + surge * exp(-((m - peak)^2) / (2 * std^2))
prompts_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month)**2) / (2 * surge_std_dev**2))

data = []
current_dau = initial_dau

for idx, m in enumerate(months):
    if idx > 0:
        current_dau = current_dau * (1 + monthly_growth_rate)
    
    daily_prompts_per_user = prompts_curve[idx]
    monthly_prompts = current_dau * daily_prompts_per_user * 30
    
    total_input_tokens = (monthly_prompts * avg_input_tokens) / 1_000_000
    total_output_tokens = (monthly_prompts * avg_output_tokens) / 1_000_000
    
    input_cost = total_input_tokens * cost_per_m_input
    output_cost = total_output_tokens * cost_per_m_output
    total_api_cost = input_cost + output_cost
    
    gross_revenue = current_dau * arpu_monthly
    gross_margin = gross_revenue - total_api_cost
    gross_margin_pct = (gross_margin / gross_revenue) * 100 if gross_revenue > 0 else 0
    cost_per_mau = total_api_cost / current_dau if current_dau > 0 else 0

    data.append({
        "Month": month_labels[idx],
        "Month_Num": m,
        "DAU": int(current_dau),
        "Prompts/User/Day": round(daily_prompts_per_user, 1),
        "Monthly Revenue ($)": round(gross_revenue, 2),
        "API Cost ($)": round(total_api_cost, 2),
        "Gross Margin ($)": round(gross_margin, 2),
        "Gross Margin (%)": round(gross_margin_pct, 1),
        "AI Cost / User / Mo ($)": round(cost_per_mau, 2)
    })

df = pd.DataFrame(data)

# --- TRIGGER DETECTION ---
margin_breach_df = df[df["Gross Margin (%)"] < min_margin_target]
crossover_df = df[df["API Cost ($)"] >= df["Monthly Revenue ($)"]]

first_breach_month = margin_breach_df["Month"].iloc[0] if not margin_breach_df.empty else None
first_crossover_month = crossover_df["Month"].iloc[0] if not crossover_df.empty else None

# --- EXECUTIVE ALERT BANNER ---
if first_crossover_month:
    st.error(f"🚨 **CRITICAL CRO ALERT:** API costs exceed total revenue in **{first_crossover_month}**! Flat-rate pricing is cash-negative. Immediate dynamic pricing or usage capping required.")
elif first_breach_month:
    st.warning(f"⚠️ **MARGIN COMPRESSION WARNING:** Gross margin drops below target threshold ({min_margin_target}%) starting in **{first_breach_month}** due to peak prompt volume. CRO intervention recommended.")
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
tab1, tab2, tab3 = st.tabs(["📉 Revenue vs. Cost Trajectory (CRO Trigger)", "📊 Usage Surge & Cost / User", "📋 Monthly Financials"])

with tab1:
    fig = go.Figure()
    
    # Revenue Bar
    fig.add_trace(go.Bar(
        x=df["Month"], y=df["Monthly Revenue ($)"],
        name="Gross Revenue ($)", marker_color="#1f77b4"
    ))
    
    # API Cost Line
    fig.add_trace(go.Scatter(
        x=df["Month"], y=df["API Cost ($)"],
        name="AI API Cost ($)", mode="lines+markers", line=dict(color="#d62728", width=4)
    ))
    
    # Highlight Breach Point
    if first_breach_month:
        breach_row = df[df["Month"] == first_breach_month].iloc[0]
        fig.add_annotation(
            x=first_breach_month, y=breach_row["API Cost ($)"],
            text=f"⚠️ CRO Action Needed<br>Margin < {min_margin_target}%",
            showarrow=True, arrowhead=2, ax=0, ay=-50,
            bgcolor="#ff7f0e", bordercolor="black", font=dict(color="white", size=12)
        )
        
    fig.update_layout(
        title="Monthly Revenue vs. Direct AI API Cost (Identifying the Pricing Cliff)",
        xaxis_title="Month",
        yaxis_title="USD ($)",
        barmode="group",
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_surge = px.line(
            df, x="Month", y="Prompts/User/Day", markers=True,
            title="User Engagement Curve (Prompts / User / Day)",
            color_discrete_sequence=["#2ca02c"]
        )
        st.plotly_chart(fig_surge, use_container_width=True)
        
    with col_b:
        fig_margin = px.line(
            df, x="Month", y="Gross Margin (%)", markers=True,
            title="Gross Margin % Compression",
            color_discrete_sequence=["#ff7f0e"]
        )
        fig_margin.add_hline(y=min_margin_target, line_dash="dash", line_color="red", annotation_text=f"Target ({min_margin_target}%)")
        st.plotly_chart(fig_margin, use_container_width=True)

with tab3:
    st.subheader("Full 12-Month Model Output")
    st.dataframe(
        df[["Month", "DAU", "Prompts/User/Day", "Monthly Revenue ($)", "API Cost ($)", "Gross Margin ($)", "Gross Margin (%)", "AI Cost / User / Mo ($)"]],
        use_container_width=True
    )

st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders and CROs evaluating AI unit economics.")
