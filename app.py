"""
AI Inference & Token Burn Rate Forecast Model
---------------------------------------------
Copyright (c) 2026. Released under the MIT License.

Author: Pooja Jaju
Description: Streamlit dashboard modeling 12-month LLM token burn, 
API inference costs, and gross margin erosion for SaaS products.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Inference & Token Burn Forecast", layout="wide")

st.title("⚡ AI Inference & Token Burn Rate Forecast")
st.caption("A financial model evaluating unit economics, API burn rate, and gross margin erosion under usage growth.")

# --- SIDEBAR ASSUMPTIONS ---
st.sidebar.header("1. Scale & Usage Drivers")
initial_dau = st.sidebar.number_input("Starting DAU", value=5000, step=500)
monthly_growth_rate = st.sidebar.slider("MoM DAU Growth Rate (%)", min_value=0.0, max_value=25.0, value=8.0) / 100
prompts_per_user_day = st.sidebar.slider("Prompts / User / Day", min_value=1, max_value=50, value=10)

st.sidebar.header("2. Token Parameters")
avg_input_tokens = st.sidebar.number_input("Avg Input Tokens / Prompt", value=800, step=100)
avg_output_tokens = st.sidebar.number_input("Avg Output Tokens / Prompt", value=300, step=50)

st.sidebar.header("3. Pricing & Economics")
model_choice = st.sidebar.selectbox(
    "LLM Tier",
    ["Standard Tier (e.g., GPT-4o-mini)", "Frontier Tier (e.g., GPT-4o)", "Custom Pricing"]
)

if model_choice == "Standard Tier (e.g., GPT-4o-mini)":
    cost_per_m_input = 0.15
    cost_per_m_output = 0.60
elif model_choice == "Frontier Tier (e.g., GPT-4o)":
    cost_per_m_input = 2.50
    cost_per_m_output = 10.00
else:
    cost_per_m_input = st.sidebar.number_input("Cost / 1M Input Tokens ($)", value=1.00)
    cost_per_m_output = st.sidebar.number_input("Cost / 1M Output Tokens ($)", value=4.00)

arpu_monthly = st.sidebar.number_input("Monthly Subscription Price / User ($)", value=30.00)

# --- MODEL CALCULATIONS ---
months = [f"M{i+1}" for i in range(12)]
data = []

current_dau = initial_dau
for m in range(12):
    if m > 0:
        current_dau = current_dau * (1 + monthly_growth_rate)
    
    monthly_prompts = current_dau * prompts_per_user_day * 30
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
        "Month": months[m],
        "DAU": int(current_dau),
        "Monthly Revenue ($)": round(gross_revenue, 2),
        "API Cost ($)": round(total_api_cost, 2),
        "Gross Margin ($)": round(gross_margin, 2),
        "Gross Margin (%)": round(gross_margin_pct, 1),
        "AI Cost / User / Mo ($)": round(cost_per_mau, 2)
    })

df = pd.DataFrame(data)

# --- EXECUTIVE SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Run-Rate Revenue (M12)", f"${df['Monthly Revenue ($)'].iloc[-1]:,.0f}")
col2.metric("Run-Rate API Cost (M12)", f"${df['API Cost ($)'].iloc[-1]:,.0f}")
col3.metric("M12 Gross Margin %", f"{df['Gross Margin (%)'].iloc[-1]}%")
col4.metric("AI Cost per User (M12)", f"${df['AI Cost / User / Mo ($)'].iloc[-1]}/mo")

st.markdown("---")

# --- VISUALIZATIONS ---
tab1, tab2 = st.tabs(["📊 Revenue vs. API Cost Trajectory", "📈 Margin Profile Over Time"])

with tab1:
    fig_costs = px.bar(
        df, 
        x="Month", 
        y=["Monthly Revenue ($)", "API Cost ($)"],
        barmode="group",
        title="12-Month Revenue vs. Direct AI Token Costs",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
    )
    st.plotly_chart(fig_costs, use_container_width=True)

with tab2:
    fig_margin = px.line(
        df, 
        x="Month", 
        y="Gross Margin (%)", 
        markers=True,
        title="Gross Margin % Trajectory",
        range_y=[0, 100]
    )
    fig_margin.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Target SaaS Margin (75%)")
    st.plotly_chart(fig_margin, use_container_width=True)

# --- DETAILED DATA TABLE ---
st.subheader("Monthly Financial Breakdown")
st.dataframe(df, use_container_width=True)

# --- FOOTER & COPYRIGHT ---
st.markdown("---")
st.caption("© 2026. Released under the MIT License. Built for FP&A Leaders evaluating AI economics.")
