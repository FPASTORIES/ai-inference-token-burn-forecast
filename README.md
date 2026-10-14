AI Inference Deal Desk

A Streamlit planning model for AI companies. It forecasts what it costs to serve customers (model API calls, vector storage, infrastructure), how promotional credits draw down, what cash is actually payable, and how different pricing structures would hold up on the same customers.

This is a planning model, not an audited financial statement. Every price, rate and preset is an editable estimate. Check them against vendor invoices, billing-system exports and your accounting policy before using any output for a decision. See Limitations.

What it does
Revenue and margin forecast for a Pro (self-serve) tier and an Enterprise (contract) tier, month by month, for 1 to 36 months.
Token cost engine: prompts per user, seasonality, heavy-user skew (multiplier or Pareto), model routing between a frontier and a standard model, caching, batch discounts and overhead.
Enterprise cohorts: contract term, renewal rate, in-term churn, per-account expansion, and a staggered starting book so renewals arrive every month instead of in one cliff.
Credits and cash: AI and cloud credit pools with eligible share and expiry, vendor invoice lag, payment fees and implementation costs paid immediately, and an implementation-fee recognition schedule.
Scenario matrix: 12 scenarios, each moving one driver family (growth, deal velocity, prompt intensity, API price, conversion, churn, or a combined stress).
Fixed-usage pricing comparison: pay-as-you-go, subscription with allowance, hybrid, and prepaid credits, all run on the same customers and one shared cost basis. It is a sensitivity, not a forecast of customer behaviour.
Usage ceiling estimate: how many prompts a Pro subscriber can use before margin falls to your target.
Audit register and integrity checks: assumptions with their expected source and an honest status, plus arithmetic checks that run on every recalculation.
Run it

Requires Python 3.10 or newer.

bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Start with the founder profile (Pre-launch, Early traction or Scaling) and the simple sidebar. Switch on Enable Advanced Deal Desk Parameters to edit every assumption.

Tabs
Tab	What it shows
Revenue & Margin	Revenue, cost, margin against target, budget cap, unit economics
Credits & Cash	Credit drawdown and expiry, cash payable versus cash paid, outstanding invoices
Monthly Detail	Full month-by-month roll-forward, with a CSV download
Scenario Matrix	The 12 scenarios side by side
Audit Register	Assumptions, sources and status, plus the integrity checks
Value & Pricing Strategy	Customer value, the fixed-usage pricing comparison and the usage ceiling
Tests
bash
pip install -r requirements-dev.txt
pytest -q tests/

tests/test_engine.py holds known-answer tests: cases worked out by hand outside the dashboard (for example one month of API cost, credit expiry, invoice-lag schedules, renewal cohorts and the usage ceiling). The in-app integrity checks only prove the arithmetic is internally consistent; these tests check it is right.

The tests load app.py with a small stand-in for Streamlit and a mock for Plotly, so they run fast and need no browser. They cannot show how the page looks. Two optional tests use Streamlit's own AppTest and run only when Streamlit and Plotly are installed. After any change, also run streamlit run app.py once and look at it.

Key modelling conventions
Month-1 DAU is used exactly as entered. Growth starts in month 2, so month 1 has no new-user conversions but does have free-pool conversions.
Revenue uses average subscribers and accounts over the month.
Credits are applied in the month the eligible cost is incurred; the invoice lag applies to what remains payable. Check this against your actual vendor credit agreements.
"Revenue Less Net Vendor Payables & Immediate Costs" is not cash flow. Customer collections and revenue-side timing are not modelled.
Renewal-month exemption: an enterprise cohort at term end faces only the renewal decision that month, and in-term churn is not also applied to it.
The usage ceiling is an estimate from company-wide cost per prompt, not a safe limit to enforce per customer.
Limitations
Net DAU growth is treated as new-user acquisition; gross sign-ups and DAU churn are not modelled.
The pricing comparison holds usage, plan mix and churn constant under every price. It does not model demand response, and its highest-margin row is not a proven best price.
Enterprise accounts renew at one blended rate on one blended contract value; discounts, ramps, minimums and account-level deals are not modelled.
Token mix, caching and batch shares are the same across customer segments.
Infrastructure steps use equal-size capacity blocks.
The change log and limitations are part of the page but hidden from view. Set SHOW_CHANGE_LOG = True near the bottom of app.py to show them.
Project layout
app.py                 the Streamlit app: sidebar, engine, dashboard
tests/test_engine.py   known-answer tests
requirements.txt       runtime dependencies
requirements-dev.txt   runtime plus test dependencies
LICENSE                MIT
Licence

MIT. See LICENSE.
