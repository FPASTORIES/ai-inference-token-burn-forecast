"""
AI Inference, Startup Runway & Unit Economics Model
---------------------------------------------------
Illustrative FP&A planning model, not an accounting system or vendor quote.
Prices and discount assumptions are user-editable estimates; verify them against
the selected provider's current pricing and billing rules.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="AI Inference & Unit Economics Forecast", page_icon="⚡", layout="wide")

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

# Sidebar inputs
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
standard_mix = 1 - frontier_mix
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

# Forecast engine
months = np.arange(1, 13)
month_days = [pd.Period(f"2027-{m:02d}").days_in_month for m in months] if days_basis == "Actual calendar days" else [30] * 12
month_names = [f"M{m}" for m in months]
prompt_curve = baseline_prompts + (peak_prompts - baseline_prompts) * np.exp(-((months - peak_month) ** 2) / (2 * surge_width ** 2))
dau = float(starting_dau)
input_tokens = float(input_tokens_start)
output_tokens = float(output_tokens_start)
credits_left = float(starting_credits)
rows = []

for i, month in enumerate(months):
    if i:
        dau *= 1 + monthly_growth
        input_tokens *= 1 + input_growth
        output_tokens *= 1 + output_growth

    paying_subscribers = dau * paying_conversion
    prompts_per_user_day = float(prompt_curve[i])
    monthly_prompts = dau * prompts_per_user_day * month_days[i]
    raw_input_m = monthly_prompts * input_tokens / 1_000_000
    raw_output_m = monthly_prompts * output_tokens / 1_000_000

    # Split input tokens into uncached and cached portions. Apply batch pricing
    # to the blended token cost, avoiding discount arithmetic on token counts.
    uncached_input_m = raw_input_m * (1 - cache_hit_rate)
    cached_input_m = raw_input_m * cache_hit_rate
    frontier_input_cost = (uncached_input_m * frontier_input_price + cached_input_m * frontier_input_price * cache_price_ratio) * frontier_mix
    standard_input_cost = (uncached_input_m * standard_input_price + cached_input_m * standard_input_price * cache_price_ratio) * standard_mix
    frontier_output_cost = raw_output_m * frontier_output_price * frontier_mix
    standard_output_cost = raw_output_m * standard_output_price * standard_mix
    api_before_batch = frontier_input_cost + standard_input_cost + frontier_output_cost + standard_output_cost
    # Blended approximation: batch share receives the batch price ratio.
    api_cost = api_before_batch * ((1 - batch_share) + batch_share * batch_price_ratio)

    vector_cost = dau * vector_cost_per_user
    infrastructure_cost = fixed_infra
    total_modeled_cost = api_cost + vector_cost + infrastructure_cost
    revenue = paying_subscribers * arpu
    gross_profit = revenue - total_modeled_cost
    gross_margin_pct = gross_profit / revenue * 100 if revenue > 0 else np.nan

    # Credits offset only the eligible share of modeled costs, up to balance.
    eligible_cost = total_modeled_cost * credit_eligible_share
    credits_applied = min(credits_left, eligible_cost)
    credits_left = max(0.0, credits_left - credits_applied)
    estimated_cash_payable = total_modeled_cost - credits_applied

    rows.append({
        "Month": month_names[i], "Calendar Days": month_days[i],
        "DAU": round(dau), "Estimated Paying Subscribers": round(paying_subscribers),
        "Prompts / User / Day": round(prompts_per_user_day, 1),
        "Input Tokens / Prompt": round(input_tokens), "Output Tokens / Prompt": round(output_tokens),
        "Monthly Prompts": round(monthly_prompts),
        "Raw Input Tokens (M)": round(raw_input_m, 2), "Raw Output Tokens (M)": round(raw_output_m, 2),
        "Subscription Revenue ($)": revenue, "API Cost ($)": api_cost,
        "Vector DB / Search ($)": vector_cost, "Fixed Infrastructure ($)": infrastructure_cost,
        "Modeled Inference + Infrastructure Cost ($)": total_modeled_cost,
        "Modeled Gross Profit ($)": gross_profit, "Modeled Gross Margin (%)": gross_margin_pct,
        "Credits Applied ($)": credits_applied, "Remaining Credits ($)": credits_left,
        "Estimated Cash Payable ($)": estimated_cash_payable,
    })

df = pd.DataFrame(rows)
df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] - budget_monthly if budget_monthly > 0 else np.nan
df["Cost per 1,000 Prompts ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] / df["Monthly Prompts"].replace(0, np.nan) * 1000
df["Cost per Paying Subscriber ($)"] = df["Modeled Inference + Infrastructure Cost ($)"] / df["Estimated Paying Subscribers"].replace(0, np.nan)

# Alerts: distinguish no starting credits from depletion within the forecast.
depleted = df.index[df["Remaining Credits ($)"] <= 0.005].tolist()
if starting_credits <= 0:
    st.warning("No starting cloud credits are available. Modeled costs are not offset by credits.")
elif depleted:
    idx = depleted[0]
    prev_balance = starting_credits if idx == 0 else df.loc[idx - 1, "Remaining Credits ($)"]
    if prev_balance > 0:
        st.warning(f"Cloud credits are exhausted during **{df.loc[idx, 'Month']}**. Estimated cash payable may begin in that month; monthly totals do not identify the exact exhaustion day.")
    else:
        st.warning(f"No cloud credits remain by **{df.loc[idx, 'Month']}**.")

breach = df.index[df["Modeled Gross Margin (%)"].notna() & (df["Modeled Gross Margin (%)"] < margin_target)].tolist()
if breach:
    st.error(f"Modeled gross margin falls below the {margin_target}% target starting in **{df.loc[breach[0], 'Month']}**.")
if budget_monthly > 0:
    over = df.index[df["Budget Variance ($)"] > 0].tolist()
    if over:
        st.warning(f"Modeled monthly costs exceed budget starting in **{df.loc[over[0], 'Month']}**.")

# Summary KPIs
annual_revenue = df["Subscription Revenue ($)"].sum()
annual_cost = df["Modeled Inference + Infrastructure Cost ($)"].sum()
annual_profit = annual_revenue - annual_cost
m12 = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("M12 Revenue Run Rate", f"${m12['Subscription Revenue ($)']:,.0f}")
c2.metric("M12 Modeled Costs", f"${m12['Modeled Inference + Infrastructure Cost ($)']:,.0f}")
c3.metric("M12 Gross Margin", "N/A" if pd.isna(m12["Modeled Gross Margin (%)"]) else f"{m12['Modeled Gross Margin (%)']:.1f}%")
c4.metric("Credits Remaining (M12)", f"${m12['Remaining Credits ($)']:,.0f}")
c5, c6, c7, c8 = st.columns(4)
c5.metric("12-Month Revenue", f"${annual_revenue:,.0f}")
c6.metric("12-Month Modeled Costs", f"${annual_cost:,.0f}")
c7.metric("12-Month Modeled Gross Profit", f"${annual_profit:,.0f}")
c8.metric("12-Month Estimated Cash Payable", f"${df['Estimated Cash Payable ($)'].sum():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Revenue & Margin", "💳 Credits & Cash", "🎛️ Scenario Analysis", "📋 Monthly Detail"])
with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Subscription Revenue ($)"], name="Modeled Subscription Revenue", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Inference + Infrastructure Cost ($)"], name="Modeled Costs", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Modeled Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", title="12-Month Revenue, Modeled Costs & Gross Margin", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="USD ($)", secondary_y=False, rangemode="tozero", gridcolor="#edf2f7")
    # Dynamic range includes negative or >100% margin values.
    margins = df["Modeled Gross Margin (%)"].dropna()
    if not margins.empty:
        lo, hi = min(0, float(margins.min())), max(100, float(margins.max()))
        pad = max(5, (hi - lo) * 0.08)
        fig.update_yaxes(title_text="Modeled Gross Margin (%)", range=[lo - pad, hi + pad], secondary_y=True, gridcolor="#edf2f7")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Margin is modeled as subscription revenue less inference and infrastructure costs, divided by modeled subscription revenue. This is not a GAAP gross-margin determination.")
with tab2:
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Credits Applied ($)"], name="Credits Applied", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Estimated Cash Payable", marker_color="#dd6b20"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Credits ($)"], name="Remaining Credit Balance", mode="lines+markers", line=dict(color="#805ad5", width=3)), secondary_y=True)
    fig2.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", title="Credit Offset vs. Estimated Cash Payable", barmode="stack", legend=dict(orientation="h", y=1.12))
    fig2.update_yaxes(title_text="Monthly Amount ($)", secondary_y=False, rangemode="tozero")
    fig2.update_yaxes(title_text="Remaining Credits ($)", secondary_y=True, rangemode="tozero")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Credits are applied only to the eligible cost share selected in the sidebar. Actual provider credit eligibility and billing timing may differ.")
with tab3:
    st.subheader("Usage and pricing sensitivity")
    st.write("Compare the modeled 12-month costs if average usage or effective token pricing changes. This simplified sensitivity holds user growth, model mix and other infrastructure assumptions constant.")
    scenarios = []
    for label, usage_factor, price_factor in [
        ("Downside: lower usage", 0.75, 1.0),
        ("Base case", 1.0, 1.0),
        ("High usage", 1.5, 1.0),
        ("Cost optimization", 1.0, 0.75),
        ("High usage + optimization", 1.5, 0.75),
    ]:
        api_s = df["API Cost ($)"].sum() * usage_factor * price_factor
        infra_s = (df["Vector DB / Search ($)"] + df["Fixed Infrastructure ($)"]).sum() * usage_factor
        cost_s = api_s + infra_s
        scenarios.append({"Scenario": label, "12-Month Modeled Costs ($)": cost_s, "12-Month Revenue ($)": annual_revenue, "Modeled Gross Profit ($)": annual_revenue - cost_s, "Modeled Gross Margin (%)": (annual_revenue - cost_s) / annual_revenue * 100 if annual_revenue else np.nan})
    scen_df = pd.DataFrame(scenarios)
    st.dataframe(scen_df.style.format({"12-Month Modeled Costs ($)": "${:,.0f}", "12-Month Revenue ($)": "${:,.0f}", "Modeled Gross Profit ($)": "${:,.0f}", "Modeled Gross Margin (%)": "{:.1f}%"}), use_container_width=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=scen_df["Scenario"], y=scen_df["12-Month Modeled Costs ($)"], name="Modeled Costs", marker_color="#3182ce"))
    fig3.update_layout(template="plotly_white", title="Scenario: 12-Month Modeled Costs", yaxis_title="USD ($)", xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Sensitivity scenarios are illustrative and simplified; usage multipliers also affect modeled infrastructure in this comparison. They are not full reforecasts.")
with tab4:
    st.subheader("Monthly forecast detail")
    display_df = df.copy()
    st.dataframe(display_df.style.format({
        "Subscription Revenue ($)": "${:,.0f}",
        "API Cost ($)": "${:,.0f}",
        "Vector DB / Search ($)": "${:,.0f}",
        "Fixed Infrastructure ($)": "${:,.0f}",
        "Modeled Inference + Infrastructure Cost ($)": "${:,.0f}",
        "Modeled Gross Profit ($)": "${:,.0f}",
        "Modeled Gross Margin (%)": "{:.1f}%",
        "Credits Applied ($)": "${:,.0f}",
        "Remaining Credits ($)": "${:,.0f}",
        "Estimated Cash Payable ($)": "${:,.0f}",
        "Budget ($)": "${:,.0f}",
        "Budget Variance ($)": "${:,.0f}",
        "Cost per 1,000 Prompts ($)": "${:,.4f}",
        "Cost per Paying Subscriber ($)": "${:,.2f}",
    }, na_rep="—"), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", data=csv, file_name="ai_inference_forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2026 FPA STORIES | MIT License. Illustrative FP&A planning model. Verify current provider prices, credit terms, tax/accounting treatment and actual billing data before business use.")
