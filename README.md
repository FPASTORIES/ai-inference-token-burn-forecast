# ⚡ AI Inference, Credit Depletion & Contract Pricing Deal Desk Model (v2.1)

An institutional-grade Financial Planning & Analysis (FP&A) and Deal Desk decision simulator built in Python and Streamlit. Designed specifically to evaluate direct AI model inference costs, vector database storage scaling, infrastructure step-costs, enterprise contract cohorts (staggered renewal books, in-term churn, and expansion), Pareto-skew user distributions, promotional credit burn velocity, and working capital cash lags.

---

## 🎯 Key Features & Capabilities

* **Founder Onboarding & Stage Presets:** Quickly switch between *Pre-launch*, *Early Traction*, and *Scaling* profiles to populate baseline assumptions.
* **Staggered Enterprise Cohort Engine:** Tracks multi-month contracts with vintage cohort retention, account expansion, staggered renewal schedules (preventing end-of-horizon cliffs), and explicit renewal/churn rules.
* **Advanced Inference Cost Routing:** Dynamically models traffic splits between frontier and standard models, cache hit-rate savings, batch pricing ratios, and token growth curves across calendar-month seasonality peaks.
* **Accounts Payable & Cash Lag Ledger:** Reconciles P&L inference costs against promotional credit grants (with expiration tracking) and vendor invoice payment lags (0 to 3 months), while separating immediate cash outflows like payment processing fees and implementation delivery costs.
* **Like-for-Like Pricing Archetype Engine:** Compares *Pay-As-You-Go*, *Subscription + Allowance*, *Hybrid Base + Overage*, and *Prepaid Credit Burndown* models on a consistent segment-by-segment paid volume and shared cost basis.
* **Automated Integrity Checks & Audit Register:** Runs continuous accounting reconciliations on every recalculation and generates an audit register mapped to live model inputs.
* **Interactive Deal Desk Governance Tabs:**
  1. 📊 **Revenue & Margin** (Multi-tier P&L, unit-economics snapshots, and cost telemetry)
  2. 💳 **Credits & Cash** (Grant drawdown, expiration velocity, and working capital cash flow)
  3. 📋 **Monthly Detail** (Full roll-forward ledger table with CSV export)
  4. 📈 **Scenario Matrix** (12-scenario comparative analysis tracking revenue, COGS, ARR, and cash impacts)
  5. 📑 **Audit Register** (Live data governance mapping and automated integrity check validation)
  6. 🎯 **Value & Pricing Strategy** (Value-creation anchor calculator and fixed-usage pricing archetype comparison)

---

## 🛠️ Tech Stack & Requirements

* **Python** (3.9+)
* **Streamlit** (Interactive web application framework)
* **Pandas & NumPy** (Data processing and simulation matrix operations)
* **Plotly** (Interactive data visualization and multi-axis charts)

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/FPASTORIES/ai-inference-token-burn-forecast.git](https://github.com/FPASTORIES/ai-inference-token-burn-forecast.git)
   cd ai-inference-token-burn-forecast
