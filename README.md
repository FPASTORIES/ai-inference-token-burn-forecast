# AI Inference & Token Burn Rate Forecast Model
> **Author:** [Pooja Jaju]
> **Contact:** [LinkedIn](https://www.linkedin.com/in/pooja-jaju-9a20b23a/) 

---

## Executive Summary
As enterprise SaaS applications integrate generative AI features, gross margins are no longer static. Variable LLM API costs directly enter Cost of Goods Sold (COGS), pulling margins down if user consumption scales faster than subscription pricing.

This interactive financial model quantifies how **active user growth, prompt frequency, and token context windows** drive 12-month direct API vendor expenditure and impact target gross margins (75%+ benchmark).

## Key Financial Insights
- **Output Sensitivity:** Because LLM output tokens are typically 3x–4x more expensive than input tokens, applications heavy on generation (e.g., long-form text, code summarization) face significantly faster margin compression.
- **Unit Economics Warning:** If AI cost per Monthly Active User (MAU) exceeds 25–30% of average monthly subscription revenue, the feature demands either usage caps, tier gating, or a migration to fine-tuned smaller models.

## Quickstart & Setup

Run these commands in your terminal to clone, install dependencies, and launch the app in one flow:

```bash
# Clone the repo and enter the project folder
git clone [https://github.com/YOUR_USERNAME/ai-inference-token-burn-forecast.git](https://github.com/YOUR_USERNAME/ai-inference-token-burn-forecast.git)
cd ai-inference-token-burn-forecast

# Install dependencies and launch the dashboard
pip install -r requirements.txt
streamlit run app.py
