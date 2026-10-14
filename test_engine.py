"""
Known-answer tests for app.py (AI Inference Deal Desk model).

These tests are deliberately OUTSIDE the dashboard. The in-app integrity checks prove the arithmetic is
internally consistent; these prove it is *right* against cases that can be worked out by hand, so a
consistently wrong assumption cannot pass them.

Run from the repo root:
    pip install pytest pandas numpy jinja2
    pytest -q tests/

Streamlit and Plotly do NOT need to be installed: the tests load app.py with a small stand-in for
Streamlit (inputs return their defaults unless a test overrides them by label) and a mock for Plotly.
That checks logic, formatting and widget arguments. It does NOT check how the page looks in a browser,
so still run `streamlit run app.py` once after changes.
"""
import math
import runpy
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"
ADV = {"🛠️ Enable Advanced Deal Desk Parameters": True}


# --------------------------------------------------------------------------------------------
# Minimal stand-in for Streamlit so app.py can be executed top to bottom without a browser
# --------------------------------------------------------------------------------------------
class _StopApp(Exception):
    pass


def _build_streamlit(overrides, log):
    st = types.ModuleType("streamlit")

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            return getattr(st, name)

    def _seen(kind, label, key):
        log.setdefault("widgets", []).append((label, key))
        log.setdefault("widget_calls", []).append((kind, label, key))

    def _check_declared(label, value, min_value, max_value, step):
        """Real Streamlit raises on mixed int/float arguments and on a default outside [min, max]."""
        nums = [x for x in (value, min_value, max_value, step) if x is not None and not isinstance(x, (tuple, list, str, bool))]
        kinds = {isinstance(x, int) for x in nums}
        assert len(kinds) <= 1, f"StreamlitMixedNumericTypesError: '{label}' mixes int and float arguments {nums}"
        if value is not None and not isinstance(value, (tuple, list, str, bool)):
            if min_value is not None:
                assert value >= min_value, f"'{label}': declared default {value} < min {min_value}"
            if max_value is not None:
                assert value <= max_value, f"'{label}': declared default {value} > max {max_value}"

    def number_input(label, min_value=None, max_value=None, value=None, step=None, key=None, format=None, help=None, **kw):
        _seen("number_input", label, key)
        _check_declared(label, value, min_value, max_value, step)
        v = overrides.get(label, value if value is not None else (min_value if min_value is not None else 0))
        if min_value is not None and v < min_value:
            raise AssertionError(f"number_input '{label}': {v} < min {min_value}")
        if max_value is not None and v > max_value:
            raise AssertionError(f"number_input '{label}': {v} > max {max_value}")
        return v

    def slider(label, min_value=None, max_value=None, value=None, step=None, key=None, help=None, **kw):
        _seen("slider", label, key)
        _check_declared(label, value, min_value, max_value, step)
        if min_value is not None and max_value is not None and min_value >= max_value:
            raise AssertionError(f"StreamlitAPIException: slider '{label}' min {min_value} >= max {max_value}")
        v = overrides.get(label, value if value is not None else min_value)
        if min_value is not None and v < min_value - 1e-9:
            raise AssertionError(f"slider '{label}': {v} < min {min_value}")
        if max_value is not None and v > max_value + 1e-9:
            raise AssertionError(f"slider '{label}': {v} > max {max_value}")
        return v

    def selectbox(label, options, index=0, format_func=None, key=None, help=None, **kw):
        _seen("selectbox", label, key)
        opts = list(options)
        v = overrides.get(label, opts[index])
        assert v in opts, f"selectbox '{label}': {v!r} not in {opts}"
        if format_func:
            [format_func(o) for o in opts]
        return v

    def toggle(label, value=False, key=None, **kw):
        _seen("toggle", label, key)
        return overrides.get(label, value)

    def rec(kind):
        def f(*a, **kw):
            log.setdefault(kind, []).append((a, kw))
        return f

    def dataframe(obj, **kw):
        log.setdefault("dataframe", []).append(obj)
        if hasattr(obj, "to_html"):  # forces pandas Styler formatting to actually run
            try:
                obj.to_html()
            except ImportError:  # jinja2 missing: skip the formatting pass
                pass

    def stop():
        raise _StopApp()

    def cache_data(*a, **kw):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def columns(spec, **kw):
        return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(labels):
        log["tabs"] = list(labels)
        return [Ctx() for _ in labels]

    def expander(label, **kw):
        return Ctx()

    for name in ("set_page_config", "markdown", "caption", "subheader", "header", "info", "warning", "success",
                 "error", "metric", "download_button", "plotly_chart"):
        setattr(st, name, rec(name))
    st.dataframe, st.stop, st.cache_data = dataframe, stop, cache_data
    st.number_input, st.slider, st.selectbox, st.toggle = number_input, slider, selectbox, toggle
    st.columns, st.tabs, st.expander = columns, tabs, expander
    st.sidebar = types.SimpleNamespace(
        header=rec("sb_header"), markdown=rec("sb_markdown"), warning=rec("sb_warning"), caption=rec("sb_caption"),
        number_input=number_input, slider=slider, selectbox=selectbox, toggle=toggle, expander=expander,
    )
    return st


def load_app(overrides=None):
    """Execute app.py top to bottom. Returns (namespace, log, stopped_by_st.stop)."""
    log = {}
    st = _build_streamlit(overrides or {}, log)
    plotly = types.ModuleType("plotly")
    go = mock.MagicMock()
    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = mock.MagicMock()
    plotly.graph_objects = go
    mods = {"streamlit": st, "plotly": plotly, "plotly.graph_objects": go, "plotly.subplots": subplots}
    with mock.patch.dict(sys.modules, mods):
        try:
            ns, stopped = runpy.run_path(str(APP)), False
        except _StopApp:
            ns, stopped = {}, True
    return ns, log, stopped


@pytest.fixture(scope="module")
def app():
    ns, log, stopped = load_app()
    assert not stopped, "app stopped on default inputs"
    return ns


@pytest.fixture(scope="module")
def sim(app):
    return app["simulate"]


@pytest.fixture(scope="module")
def base_params(app):
    return dict(app["P"])


def params(base, **kw):
    p = dict(base)
    p.update(kw)
    return p


ISOLATED = dict(
    # calendar and population: 100 Pro subscribers, 1,000 DAU, nobody else, nothing changes over time
    horizon=1, start_year=2027, start_month=1, dau=1000.0, pro=100.0, ent=0.0, net_new_ent=0.0, dau_growth=0.0,
    new_conv=0.0, base_conv=0.0, churn=0.0, paid_dau_factor=0.70,
    # price and revenue: $30 / month, no expansion, no fees, no implementation fee or cost
    arpu_pro=30.0, arpu_ent=250.0, pro_expansion_pct=0.0, fee_pct=0.0, impl_fee=0.0, impl_cost=0.0, impl_rec_months=1,
    # enterprise lifecycle (inert while ent = 0, pinned so a default change cannot leak in)
    ent_term=12, ent_renewal_rate=1.0, ent_in_term_churn=0.0, ent_expansion=0.0, ent_seats=20, ent_prompts=3000,
    # usage: flat 10 prompts / active user / day, no heavy users, no free-tier traffic, flat tokens
    base_prompts=10.0, p1_vol=10, p1_month=3, p1_width=1.5, p2_vol=10, p2_month=11, p2_width=2.0,
    heavy_mode="Multiplier", heavy_share=0.0, heavy_mult=1.0, pareto_share=0.80, free_mult=0.0, usage_mult=1.0,
    in_tok=1000, out_tok=400, in_grow=0.0, out_grow=0.0,
    # token prices ($ per 1M tokens), standard model only, no cache / batch / overhead
    f_in=2.50, f_out=10.00, s_in=0.15, s_out=0.60, price_mult=1.0, front_mix=0.0,
    cache_rate=0.0, cache_ratio=0.20, batch_share=0.0, batch_ratio=0.50, overhead=0.0,
    # infrastructure and credits: none; no step cost can trigger
    base_infra=0.0, vec_cost=0.0, step_type="DAU Threshold", step_thresh=1e12, step_cost=0.0,
    ai_credits=0.0, cloud_credits=0.0, ai_expiry=0, cloud_expiry=0, ai_elig=1.0, cloud_elig=1.0, invoice_lag=0,
)


def isolated(base, **kw):
    """A world where every number can be worked out by hand. Every input the engine reads is pinned here,
    so a change to an app default can never alter these tests. Callers may override any pinned value."""
    d = dict(ISOLATED)
    d.update(kw)
    return params(base, **d)


# --------------------------------------------------------------------------------------------
# 1. Hand-calculated token-cost example
# --------------------------------------------------------------------------------------------
def test_hand_calculated_month(sim, base_params):
    r = sim(isolated(base_params)).iloc[0]
    prompts = 100 * 0.70 * 10 * 31          # 100 Pro x 70% active x 10 prompts/day x 31 days (Jan)
    api = prompts * (1000 * 0.15 + 400 * 0.60) / 1e6   # all traffic on the standard model, no cache or batch
    assert r["Monthly Prompts"] == pytest.approx(21700.0)
    assert api == pytest.approx(8.463)
    assert r["API Cost ($)"] == pytest.approx(api)
    assert r["Recognized Revenue ($)"] == pytest.approx(100 * 30.0)
    assert r["Total Modeled COGS ($)"] == pytest.approx(api)
    assert r["Modeled Gross Margin (%)"] == pytest.approx((3000 - 8.463) / 3000 * 100)


def test_caching_batching_and_routing_known_answer(sim, base_params):
    # 21,700 prompts; 1,000 in / 400 out tokens. 50% frontier. 40% of input cached at 20% price. 20% batch at 50% price.
    p = isolated(base_params)
    p.update(front_mix=0.5, cache_rate=0.4, cache_ratio=0.2, batch_share=0.2, batch_ratio=0.5)
    in_m, out_m = 21700 * 1000 / 1e6, 21700 * 400 / 1e6
    frontier_in = (in_m * 0.6 * 2.50 + in_m * 0.4 * 2.50 * 0.2) * 0.5
    standard_in = (in_m * 0.6 * 0.15 + in_m * 0.4 * 0.15 * 0.2) * 0.5
    expected = (frontier_in + standard_in + out_m * 10.0 * 0.5 + out_m * 0.60 * 0.5) * (0.8 + 0.2 * 0.5)
    assert sim(p).iloc[0]["API Cost ($)"] == pytest.approx(expected)


# --------------------------------------------------------------------------------------------
# 2. Zero cases
# --------------------------------------------------------------------------------------------
def test_zero_usage(sim, base_params):
    p = params(base_params, usage_mult=0.0, base_infra=2500.0, ai_credits=0.0, cloud_credits=0.0)
    d = sim(p)
    assert (d["Monthly Prompts"] == 0).all()
    assert (d["API Cost ($)"] == 0).all() and (d["Vector DB Cost ($)"] == 0).all()
    expected = 2500.0 + d["Recognized Revenue ($)"] * p["fee_pct"]
    assert d["Total Modeled COGS ($)"].to_numpy() == pytest.approx(expected.to_numpy())


def test_zero_revenue(sim, base_params):
    # Assumptions made explicit: no customers, no usage, no implementation cost, no payment fees on zero revenue,
    # no step cost. The only cost left is $2,500 of fixed infrastructure.
    p = isolated(base_params, horizon=6, dau=0.0, pro=0.0, base_infra=2500.0)
    d = sim(p)
    assert (d["Recognized Revenue ($)"] == 0).all()
    assert d["Modeled Gross Margin (%)"].isna().all()           # undefined, never 0 and never "healthy"
    assert d["Total Modeled COGS ($)"].to_numpy() == pytest.approx(2500.0)


def test_zero_credits(sim, base_params):
    d = sim(params(base_params, ai_credits=0.0, cloud_credits=0.0))
    assert (d["Total Credits Applied ($)"] == 0).all()
    assert d["Estimated Cash Payable ($)"].to_numpy() == pytest.approx(d["Total Modeled COGS ($)"].to_numpy())


def test_one_month_horizon(sim, base_params):
    d = sim(params(base_params, horizon=1))
    assert len(d) == 1
    assert d["Beginning Ent Accounts"].iloc[0] == pytest.approx(base_params["ent"])


# --------------------------------------------------------------------------------------------
# 3. DAU convention: month-1 DAU is used exactly as entered; growth starts in month 2
# --------------------------------------------------------------------------------------------
def test_month1_dau_is_as_entered_and_growth_starts_month2(sim, base_params):
    p = params(base_params, horizon=3, dau=5000.0, dau_growth=0.08, new_conv=0.15, base_conv=0.0, pro=200.0, ent=0.0, net_new_ent=0.0)
    d = sim(p)
    assert d["DAU"].iloc[0] == pytest.approx(5000.0)
    assert d["DAU"].iloc[1] == pytest.approx(5400.0)
    assert d["New-User Conversions"].iloc[0] == 0.0
    assert d["New-User Conversions"].iloc[1] == pytest.approx(400 * 0.15)       # 400 net new DAU x 15%


def test_month1_still_has_free_pool_conversions(sim, base_params):
    p = params(base_params, horizon=1, dau=5000.0, pro=200.0, ent=0.0, net_new_ent=0.0, base_conv=0.02, paid_dau_factor=0.7)
    r = sim(p).iloc[0]
    free_pool = 5000.0 - 200.0 * 0.7
    assert r["Free-Pool Conversions"] == pytest.approx(free_pool * 0.02)


# --------------------------------------------------------------------------------------------
# 4. Enterprise renewal, expiry and expansion
# --------------------------------------------------------------------------------------------
def _ent_world(base, **kw):
    d = dict(ent=120.0, net_new_ent=0.0, ent_expansion=0.0, ent_in_term_churn=0.0, ent_term=12)
    d.update(kw)
    return params(base, **d)


def test_100_percent_renewal_keeps_every_account(sim, base_params):
    d = sim(_ent_world(base_params, horizon=24, ent_renewal_rate=1.0))
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx(120.0)
    assert d["Ent Non-Renewals"].sum() == pytest.approx(0.0)


def test_zero_percent_renewal_empties_the_book_after_one_term(sim, base_params):
    d = sim(_ent_world(base_params, horizon=13, ent_renewal_rate=0.0))
    assert d["Ending Ent Accounts"].iloc[-1] == pytest.approx(0.0)
    assert (d["Ending Ent Accounts"].diff().dropna() <= 1e-9).all()               # only ever falls


def test_partial_renewal_staggered_book_known_answer(sim, base_params):
    d = sim(_ent_world(base_params, horizon=13, ent_renewal_rate=0.75))
    assert d["Ent Accounts Up For Renewal"].iloc[1:13].to_numpy() == pytest.approx(10.0)   # 120 accounts / 12 terms
    assert d["Ent Non-Renewals"].iloc[1:13].to_numpy() == pytest.approx(2.5)              # 25% of 10 lost each month
    assert d["Ending Ent Accounts"].iloc[-1] == pytest.approx(120 - 12 * 2.5)             # exactly 90


def test_one_month_term_known_answer(sim, base_params):
    # Starting book is staggered over remaining terms 1..term, so with term = 1 the whole book renews in month 2.
    d = sim(_ent_world(base_params, horizon=3, ent_term=1, ent_renewal_rate=0.75))
    assert d["Ent Accounts Up For Renewal"].to_numpy() == pytest.approx([0.0, 120.0, 90.0])
    assert d["Ent Non-Renewals"].to_numpy() == pytest.approx([0.0, 30.0, 22.5])
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx([120.0, 90.0, 67.5])


def test_book_starting_at_zero_builds_cohorts_that_renew_on_their_own_dates(sim, base_params):
    # One new deal a month, 3-month term, 50% renewal. Cohort k renews at the start of month k+3.
    p = params(base_params, horizon=7, ent=0.0, net_new_ent=1.0, ent_term=3, ent_renewal_rate=0.5, ent_in_term_churn=0.0, ent_expansion=0.0)
    d = sim(p)
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx([1, 2, 3, 3.5, 4, 4.5, 4.75])
    assert d["Ent Accounts Up For Renewal"].to_numpy() == pytest.approx([0, 0, 0, 1, 1, 1, 1.5])
    assert d["Ent Non-Renewals"].to_numpy() == pytest.approx([0, 0, 0, 0.5, 0.5, 0.5, 0.75])


def test_in_term_churn_and_renewal_churn_apply_to_different_cohorts(sim, base_params):
    # 100 accounts, term 1 (whole book renews in month 2), 90% renewal, 10% in-term churn.
    # Month 1: nobody is at term end, so in-term churn takes 10 -> 90.
    # Month 2: the book is at term end. Renewal loses 9 -> 81. Renewal-month exemption: NO in-term churn that month.
    d = sim(_ent_world(base_params, horizon=3, ent=100.0, ent_term=1, ent_renewal_rate=0.9, ent_in_term_churn=0.10))
    assert d["Ent Non-Renewals"].to_numpy() == pytest.approx([0.0, 9.0, 8.1])
    assert d["Ent In-Term Churn"].to_numpy() == pytest.approx([10.0, 0.0, 0.0])
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx([90.0, 81.0, 72.9])


def test_cohorts_not_at_term_end_still_take_in_term_churn(sim, base_params):
    # 2-month term: the 100-account book is two cohorts of 50 (renewing in months 2 and 3). 10% in-term churn, 100% renewal.
    # Month 1: neither cohort at term end -> 10% of 100 = 10 lost. Month 2: cohort A renews (exempt); cohort B (45) loses 4.5.
    d = sim(_ent_world(base_params, horizon=2, ent=100.0, ent_term=2, ent_renewal_rate=1.0, ent_in_term_churn=0.10))
    assert d["Ent In-Term Churn"].to_numpy() == pytest.approx([10.0, 4.5])
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx([90.0, 85.5])


def test_renewal_month_exemption_is_listed_as_an_assumption():
    _, log, _ = load_app(dict(ADV))
    rows = " ".join(str(df.to_string()) for df in log["dataframe"] if hasattr(df, "to_string") and not hasattr(df, "to_html"))
    md = " ".join(str(a[0]) for a, k in log["markdown"] if a)
    assert "Enterprise renewal-month exemption" in rows or "Enterprise renewal-month exemption" in " ".join(
        str(getattr(df, "data", df).to_string()) for df in log["dataframe"] if hasattr(getattr(df, "data", df), "to_string"))
    assert "Renewal-month exemption (model assumption)" in md


def test_pricing_comparison_states_its_billing_basis(app, sim, base_params):
    pr = app["compare_pricing"](sim(base_params), base_params, app["_default_rules"])
    assert "Billing Basis" in pr.columns and (pr["Billing Basis"].str.len() > 20).all()
    assert "INCLUDES" in pr[pr["Pricing Archetype"].str.startswith("0.")].iloc[0]["Billing Basis"]
    assert "excludes the expansion uplift" in pr[pr["Pricing Archetype"].str.startswith("2.")].iloc[0]["Billing Basis"]


def test_default_no_month_12_cliff(app, sim, base_params):
    ent = sim(base_params)["Ending Ent Accounts"].to_numpy()
    assert (np.diff(ent) > -0.12 * ent[:-1]).all()


def test_expansion_grows_value_not_account_count(sim, base_params):
    p = params(base_params, horizon=12, ent=50.0, net_new_ent=0.0, ent_renewal_rate=1.0, ent_in_term_churn=0.0, ent_expansion=0.02)
    d = sim(p)
    assert d["Ending Ent Accounts"].to_numpy() == pytest.approx(50.0)
    assert d["Ent Expansion Index"].iloc[-1] == pytest.approx(1.02 ** 12)


def test_new_deals_are_not_rounded(sim, base_params):
    d = sim(params(base_params, horizon=2, ent=0.0, net_new_ent=2.5, ent_in_term_churn=0.0))
    assert d["New Ent Deals"].iloc[0] == pytest.approx(2.5)


# --------------------------------------------------------------------------------------------
# 5. Seasonality and heavy users
# --------------------------------------------------------------------------------------------
def test_peak_lands_in_the_calendar_month(app):
    cal = [((10 - 1 + i) % 12) + 1 for i in range(12)]            # forecast starting in October
    c = app["usage_curve"](cal, 5.0, 5, 3, 1.5, 45, 11, 1.0)
    assert int(np.argmax(c)) == 1                                  # November is the second forecast month


def test_peak_below_baseline_is_floored(app):
    c = app["usage_curve"]([1, 2, 3], 10.0, 2, 2, 1.0, 2, 3, 1.0)
    assert (c >= 10.0 - 1e-12).all()


def test_pareto_calibration_known_answer(app, sim, base_params):
    p = params(base_params, heavy_mode="Pareto prompt share", heavy_share=0.2, pareto_share=0.8)
    assert app["heavy_multiplier"](p) == pytest.approx(16.0)       # 20% of users make 80% of prompts => 16x
    d = sim(p)
    share = d["Heavy Pro Prompts"].sum() / (d["Heavy Pro Prompts"].sum() + d["Typical Pro Prompts"].sum())
    assert share == pytest.approx(0.8)


# --------------------------------------------------------------------------------------------
# 6. Credits, invoice lag, cash metrics
# --------------------------------------------------------------------------------------------
def test_credits_exhaust_without_going_negative(sim, base_params):
    d = sim(params(base_params, ai_credits=500.0, cloud_credits=0.0, ai_expiry=0))
    assert (d["Remaining AI Credits ($)"] >= 0).all()
    assert d["Remaining AI Credits ($)"].iloc[-1] == 0
    assert d["AI Credits Applied ($)"].sum() == pytest.approx(500.0)


# Isolated world, Jan-Jun 2027: API cost per month = 100 Pro x 70% x 10 prompts x days x $0.00039 = $0.273 x days.
JAN, FEB, MAR, APR, MAY, JUN = 8.463, 7.644, 8.463, 8.190, 8.463, 8.190


def test_isolated_monthly_api_costs_match_the_hand_values(sim, base_params):
    d = sim(isolated(base_params, horizon=6))
    assert d["API Cost ($)"].to_numpy() == pytest.approx([JAN, FEB, MAR, APR, MAY, JUN])


def test_credits_expire_exactly_at_the_documented_boundary(sim, base_params):
    # $20 of AI credits usable THROUGH forecast month 2. Jan uses 8.463, Feb uses 7.644, 3.893 is left unused.
    d = sim(isolated(base_params, horizon=5, ai_credits=20.0, ai_expiry=2))
    leftover = 20.0 - JAN - FEB
    assert d["Opening AI Credits ($)"].to_numpy() == pytest.approx([20.0, 20.0 - JAN, leftover, 0.0, 0.0])
    assert d["AI Credits Applied ($)"].to_numpy() == pytest.approx([JAN, FEB, 0.0, 0.0, 0.0])
    assert d["AI Credits Expired ($)"].to_numpy() == pytest.approx([0.0, 0.0, leftover, 0.0, 0.0])   # unused balance, recorded
    assert d["Remaining AI Credits ($)"].to_numpy() == pytest.approx([20.0 - JAN, leftover, 0.0, 0.0, 0.0])
    assert d["Vendor Invoices Payable ($)"].iloc[2] == pytest.approx(MAR)          # after expiry the full cost is payable
    assert d["AI Credits Applied ($)"].sum() + d["AI Credits Expired ($)"].sum() == pytest.approx(20.0)


def test_credits_exhausted_before_expiry_are_not_reported_as_expired(sim, base_params):
    d = sim(isolated(base_params, horizon=4, ai_credits=10.0, ai_expiry=4))
    assert d["AI Credits Applied ($)"].to_numpy() == pytest.approx([JAN, 10.0 - JAN, 0.0, 0.0])
    assert d["AI Credits Expired ($)"].to_numpy() == pytest.approx(0.0)
    assert d["Remaining AI Credits ($)"].to_numpy() == pytest.approx([10.0 - JAN, 0.0, 0.0, 0.0])


def test_credit_eligible_share_limits_what_credits_can_cover(sim, base_params):
    d = sim(isolated(base_params, horizon=1, ai_credits=100.0, ai_elig=0.5)).iloc[0]
    assert d["AI Credits Applied ($)"] == pytest.approx(JAN * 0.5)
    assert d["Vendor Invoices Payable ($)"] == pytest.approx(JAN * 0.5)
    assert d["Remaining AI Credits ($)"] == pytest.approx(100.0 - JAN * 0.5)


def test_cloud_credits_cover_infrastructure_and_exhaust_in_month_three(sim, base_params):
    d = sim(isolated(base_params, horizon=3, cloud_credits=250.0, base_infra=100.0))
    assert d["Cloud Credits Applied ($)"].to_numpy() == pytest.approx([100.0, 100.0, 50.0])
    assert d["Remaining Cloud Credits ($)"].to_numpy() == pytest.approx([150.0, 50.0, 0.0])
    assert d["Vendor Invoices Payable ($)"].to_numpy() == pytest.approx([JAN, FEB, MAR + 50.0])


@pytest.mark.parametrize("lag", [0, 1, 2, 3])
def test_cash_paid_plus_outstanding_equals_cash_payable(sim, base_params, lag):
    d = sim(params(base_params, invoice_lag=lag, impl_fee=1000.0, impl_cost=400.0, horizon=12))
    assert (d["Estimated Cash Paid This Month ($)"].sum() + d["Vendor Invoices Outstanding ($)"].iloc[-1]
            == pytest.approx(d["Estimated Cash Payable ($)"].sum()))


@pytest.mark.parametrize("lag", [0, 1, 2, 3])
def test_invoice_lag_monthly_payment_schedule(sim, base_params, lag):
    # Vendor invoices = API cost each month. Payment fees ($300 = 10% of $3,000 revenue) are paid immediately.
    d = sim(isolated(base_params, horizon=6, invoice_lag=lag, fee_pct=0.10))
    invoices = [JAN, FEB, MAR, APR, MAY, JUN]
    paid = [(invoices[i - lag] if i >= lag else 0.0) + 300.0 for i in range(6)]
    outstanding = [sum(invoices[max(0, i - lag + 1):i + 1]) if lag > 0 else 0.0 for i in range(6)]
    assert d["Estimated Cash Paid This Month ($)"].to_numpy() == pytest.approx(paid)
    assert d["Vendor Invoices Outstanding ($)"].to_numpy() == pytest.approx(outstanding)
    # invoices raised in the last `lag` months have not been paid by the horizon and stay outstanding
    assert d["Vendor Invoices Outstanding ($)"].iloc[-1] == pytest.approx(sum(invoices[6 - lag:]) if lag else 0.0)


def test_cash_metric_is_named_for_what_it_is(sim, base_params):
    d = sim(params(base_params, invoice_lag=2))
    col = "Revenue Less Net Vendor Payables & Immediate Costs ($)"
    assert col in d.columns and "Cash Gross Profit After Credits ($)" not in d.columns
    assert d[col].to_numpy() == pytest.approx((d["Recognized Revenue ($)"] - d["Estimated Cash Payable ($)"]).to_numpy())


def test_implementation_fee_recognition_schedule(sim, base_params):
    d = sim(params(base_params, horizon=6, ent=0.0, net_new_ent=1.0, impl_fee=3000.0, impl_rec_months=3, impl_cost=0.0))
    assert d["Implementation Fees Billed ($)"].iloc[0] == pytest.approx(3000.0)
    assert d["Implementation Revenue ($)"].iloc[0] == pytest.approx(1000.0)        # 3,000 over 3 months
    assert d["Deferred Implementation Revenue ($)"].iloc[0] == pytest.approx(2000.0)


# --------------------------------------------------------------------------------------------
# 7. Scenarios, pricing comparison, usage ceiling
# --------------------------------------------------------------------------------------------
def test_default_scenarios_isolate_their_driver(app, sim, base_params):
    scen = dict(app["SCENARIOS"])
    apply = app["apply_scenario"]
    q = apply(base_params, scen["4. Enterprise Deal Velocity High (1.25x)"])
    assert q["net_new_ent"] == pytest.approx(base_params["net_new_ent"] * 1.25)
    assert q["dau_growth"] == pytest.approx(base_params["dau_growth"])
    b = sim(base_params)
    h = sim(apply(base_params, scen["6. Prompt Intensity High (1.5x, all segments)"]))
    assert h["Enterprise Prompts"].sum() / b["Enterprise Prompts"].sum() == pytest.approx(1.5)
    assert h["Recognized Revenue ($)"].sum() == pytest.approx(b["Recognized Revenue ($)"].sum())


def test_default_base_scenario_equals_main_model(app, sim, base_params):
    assert sim(app["apply_scenario"](base_params, {}))["Recognized Revenue ($)"].sum() == pytest.approx(sim(base_params)["Recognized Revenue ($)"].sum())


def test_default_pricing_comparison_shares_one_cost_basis(app, sim, base_params):
    d = sim(base_params)
    pr = app["compare_pricing"](d, base_params, app["_default_rules"])
    assert pr["Paid-Usage Cost ($)"].nunique() == 1
    ref = pr[pr["Pricing Archetype"].str.startswith("0.")].iloc[0]
    assert ref["Simulated Revenue ($)"] == pytest.approx(d["Subscription Revenue ($)"].sum())
    paid_prompts = (d["Typical Pro Prompts"] + d["Heavy Pro Prompts"] + d["Enterprise Prompts"]).sum()
    payg = pr[pr["Pricing Archetype"].str.startswith("1.")].iloc[0]
    assert payg["Simulated Revenue ($)"] == pytest.approx(paid_prompts * app["_default_rules"]["payg_rate"])


def test_cost_allocation_never_loses_idle_cost(app, sim, base_params):
    d = sim(params(base_params, dau=0.0, pro=0.0, ent=0.0, net_new_ent=0.0, base_infra=2500.0))
    paid, free = app["cost_allocation"](d)
    assert paid + free == pytest.approx((d["API Cost ($)"] + d["Vector DB Cost ($)"] + d["Fixed Infrastructure ($)"]).sum())
    assert paid == 0 and free == pytest.approx(2500.0 * len(d))


def test_usage_ceiling_hand_calculated(app, sim, base_params):
    # Isolated month: 21,700 prompts. API $8.463 + vector 21.7 x $0.05 = $1.085  =>  $9.548 / 21,700 = $0.00044 per prompt.
    # Pro price $30, 2.9% payment fee, 60% target margin  =>  $30 x (1 - 0.60 - 0.029) = $11.13 of variable cost allowed.
    p = isolated(base_params, vec_cost=0.05, fee_pct=0.029, base_infra=100.0)
    uc = app["usage_ceiling"](sim(p), p, 60.0)
    assert uc["var_cost"] == pytest.approx(0.00044)
    assert uc["allowed"] == pytest.approx(11.13)
    assert uc["var"] == pytest.approx(11.13 / 0.00044)                                  # about 25,295 prompts / account / month
    assert uc["loaded_cost"] == pytest.approx(0.00044 + 100.0 / 21700)                  # fixed infra spread over prompts
    assert uc["loaded"] == pytest.approx(11.13 / (0.00044 + 100.0 / 21700))
    # the same answer as a margin worked out by hand: revenue 30, fee 0.87, cost 25,295.45 x 0.00044 = 11.13  ->  60%
    assert (30.0 - 30.0 * 0.029 - uc["var"] * 0.00044) / 30.0 == pytest.approx(0.60)


def test_usage_ceiling_counts_pro_expansion_in_revenue(app, sim, base_params):
    p = isolated(base_params, vec_cost=0.05, fee_pct=0.029, pro_expansion_pct=0.10)
    uc = app["usage_ceiling"](sim(p), p, 60.0)
    assert uc["rev"] == pytest.approx(33.0)
    assert uc["allowed"] == pytest.approx(33.0 * 0.371)


def test_usage_ceiling_with_no_or_negative_headroom(app, sim, base_params):
    p = isolated(base_params, vec_cost=0.05, fee_pct=0.029)
    uc = app["usage_ceiling"](sim(p), p, 98.0)               # 98% target + 2.9% fee leaves less than nothing
    assert uc["no_headroom"] and uc["var"] == 0.0 and uc["allowed"] < 0


def test_usage_ceiling_is_unbounded_with_no_traffic(app, sim, base_params):
    p = isolated(base_params, usage_mult=0.0)
    uc = app["usage_ceiling"](sim(p), p, 60.0)
    assert math.isinf(uc["var"]) and not uc["no_headroom"]


# --- default-configuration tests: these may legitimately need new numbers if the baseline assumptions change ---
@pytest.mark.parametrize("target", [40.0, 60.0, 80.0])
def test_default_usage_ceiling_solver_is_self_consistent(app, sim, base_params, target):
    d = sim(base_params)
    uc = app["usage_ceiling"](d, base_params, target)
    assert math.isfinite(uc["var"]) and not uc["no_headroom"]
    revenue_per_pro = uc["rev"]
    margin = (revenue_per_pro * (1 - base_params["fee_pct"]) - uc["var"] * uc["var_cost"]) / revenue_per_pro * 100
    assert margin == pytest.approx(target)


def test_usage_ceiling_uses_the_ending_months_cost_basis(app, sim, base_params):
    d = sim(base_params)
    last = d.iloc[-1]
    uc = app["usage_ceiling"](d, base_params, 60.0)
    assert uc["var_cost"] == pytest.approx((last["API Cost ($)"] + last["Vector DB Cost ($)"]) / last["Monthly Prompts"])


# --------------------------------------------------------------------------------------------
# 8. The dashboard itself: full runs, warnings, labels, integrity checks
# --------------------------------------------------------------------------------------------
SMOKE_CASES = {
    "defaults": {},
    "advanced": dict(ADV),
    "pre-launch": {"Choose a starting profile": "Pre-launch"},
    "pre-launch advanced": {**ADV, "Choose a starting profile": "Pre-launch"},
    "scaling advanced": {**ADV, "Choose a starting profile": "Scaling"},
    "1-month horizon advanced": {**ADV, "Forecast Horizon (Months)": 1},
    "36-month horizon advanced": {**ADV, "Forecast Horizon (Months)": 36},
    "24-month horizon": {"Forecast Horizon (Months)": 24},
    "start in December": {"Forecast Start Month": 12},
    "invoice lag 3": {**ADV, "Vendor Invoice Payment Lag (months)": 3},
    "credits expire": {**ADV, "AI Credits Usable Through Forecast Month (0 = no expiry)": 4},
    "implementation fee": {**ADV, "Implementation Fee per New Deal ($)": 6000.0, "Implementation Fee Recognition Period (months; 1 = upfront)": 3},
    "pareto": {**ADV, "Heavy-User Calibration": "Pareto prompt share"},
    "zero renewal": {**ADV, "Renewal Rate at Term End (%)": 0},
    "1-month contract term": {**ADV, "Contract Term (months)": 1},
    "prompts step trigger": {**ADV, "Step-Cost Trigger Basis": "Monthly Prompts (Millions)"},
    "budget cap": {"Monthly COGS Budget Cap ($; 0 = none)": 5000.0},
    "no headroom": {"Target Gross Margin (%)": 90},
    "zero everything": {"Month-1 Daily Active Users (DAU)": 0, "Starting Pro Subscribers": 0,
                        "Starting Enterprise Accounts (contracts)": 0, "Net New Enterprise Deals / Month": 0.0},
}


@pytest.mark.parametrize("name", list(SMOKE_CASES))
def test_whole_app_runs_and_integrity_checks_pass(name):
    ns, log, stopped = load_app(SMOKE_CASES[name])
    assert not stopped and not log.get("error"), f"{name}: {log.get('error')}"
    assert len(log["tabs"]) == 6
    checks = ns["checks_df"]
    assert not (checks["Result"] == "FAIL").any(), checks[checks["Result"] == "FAIL"]
    assert "Usage ceiling solver: per-account margin at the ceiling equals the target" in set(checks["Check"])   # never silently skipped


@pytest.mark.parametrize("advanced", [False, True])
def test_widget_keys_are_unique_across_the_whole_page(advanced):
    # Streamlit identifies a keyed widget by its key alone, whatever its label, so two widgets with the same
    # explicit key collide (DuplicateWidgetID) even if their labels differ.
    _, log, _ = load_app(dict(ADV) if advanced else {})
    keys = [k for kind, label, k in log["widget_calls"] if k is not None]
    assert len(keys) == len(set(keys)), sorted({k for k in keys if keys.count(k) > 1})


@pytest.mark.parametrize("advanced", [False, True])
def test_keyless_widgets_do_not_share_a_label(advanced):
    # Without a key Streamlit derives identity from the label and arguments; identical labels risk a collision.
    _, log, _ = load_app(dict(ADV) if advanced else {})
    labels = [label for kind, label, k in log["widget_calls"] if k is None]
    assert len(labels) == len(set(labels)), sorted({x for x in labels if labels.count(x) > 1})


def test_every_widget_type_the_app_uses_is_recorded():
    # Guards the harness itself: if app.py starts using a widget the stand-in does not log, say so loudly.
    import re
    used = set(re.findall(r"st(?:\.sidebar)?\.(number_input|slider|selectbox|toggle|checkbox|radio|text_input|multiselect|text_area|date_input)\(", APP.read_text()))
    _, log, _ = load_app(dict(ADV))
    assert used <= {"number_input", "slider", "selectbox", "toggle"}, f"stand-in does not emulate: {used}"
    assert {kind for kind, _, _ in log["widget_calls"]} >= used - {"toggle"}
    assert any(kind == "toggle" for kind, _, _ in log["widget_calls"])


def test_widget_defaults_are_valid_for_real_streamlit():
    # The stand-in raises on mixed int/float arguments and on declared defaults outside [min, max], as Streamlit does.
    # Reaching this line for every smoke configuration means none of them tripped it.
    for name in ("defaults", "advanced", "pre-launch advanced", "scaling advanced"):
        ns, log, stopped = load_app(SMOKE_CASES[name])
        assert not stopped and "widget_calls" in log


def _text(log, kind):
    return " ".join(str(a[0]) for a, k in log.get(kind, []) if a)


def test_zero_revenue_briefing_says_undefined_not_healthy():
    _, log, _ = load_app(SMOKE_CASES["zero everything"])
    briefing = " ".join(a[0] for a, k in log["markdown"] if a and isinstance(a[0], str) and "Governance Briefing" in a[0])
    assert "undefined" in briefing and "robust" not in briefing and "nan%" not in briefing.lower()
    kpi = [a for a, k in log["metric"] if a[0] == "Forecast-Period Gross Margin"][0]
    assert "n/a" in kpi[1]


def test_roi_shows_na_when_price_is_zero():
    _, log, _ = load_app({**ADV, "Pro Tier Monthly ARPU ($)": 0.0})
    md = " ".join(a[0] for a, k in log["markdown"] if a and isinstance(a[0], str) and "Customer ROI multiple" in a[0])
    assert "N/A (price is zero)" in md and "0.0x" not in md


def test_starting_warning_uses_active_paid_users_not_contract_counts():
    # 900 enterprise contracts is fewer than 1,000 DAU, but at 20 seats and 70% activity that is 12,600 active users.
    _, log, _ = load_app({"Month-1 Daily Active Users (DAU)": 1000, "Starting Pro Subscribers": 0,
                          "Starting Enterprise Accounts (contracts)": 900})
    assert "exceed month-1 DAU" in _text(log, "warning")
    _, log, _ = load_app({"Month-1 Daily Active Users (DAU)": 5000, "Starting Pro Subscribers": 100,
                          "Starting Enterprise Accounts (contracts)": 10})
    assert "exceed month-1 DAU" not in _text(log, "warning")


def test_labels_describe_what_the_numbers_are():
    _, log, _ = load_app()
    metrics = [a[0] for a, k in log["metric"]]
    assert any(m.startswith("Est. Pro Usage Ceiling") for m in metrics)
    assert "Revenue Less Net Vendor Payables & Immediate Costs (horizon)" in metrics
    captions = _text(log, "caption")
    assert "sensitivity on constant customer behavior" in captions
    assert "not a guaranteed safe limit" in captions


# --------------------------------------------------------------------------------------------
# 9. OPTIONAL: real Streamlit. Skipped automatically when streamlit is not installed.
# --------------------------------------------------------------------------------------------
def _real_streamlit():
    try:
        from streamlit.testing.v1 import AppTest
        import plotly  # noqa: F401
        return AppTest
    except Exception:
        return None


@pytest.mark.skipif(_real_streamlit() is None, reason="streamlit / plotly not installed")
def test_real_streamlit_app_renders_without_exceptions():
    at = _real_streamlit().from_file(str(APP), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert len(at.tabs) == 6
    assert len(at.metric) > 0


@pytest.mark.skipif(_real_streamlit() is None, reason="streamlit / plotly not installed")
def test_real_streamlit_advanced_toggle_renders_without_exceptions():
    at = _real_streamlit().from_file(str(APP), default_timeout=60).run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]


def test_change_log_is_in_the_page_but_not_visible():
    _, log, _ = load_app()
    hidden = [a[0] for a, k in log["markdown"] if a and isinstance(a[0], str) and "change-log-and-limitations" in a[0]]
    assert len(hidden) == 1
    block = hidden[0]
    assert 'style="display:none"' in block and "\n" not in block          # one HTML block: a blank line would leak it on screen
    assert "Simplifications that remain" in block and "Renewal-month exemption" in block
    assert not any("Change log" in str(e) for e in log.get("expanders", []))
