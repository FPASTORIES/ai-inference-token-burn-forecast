                    '  return ContentService.createTextOutput(JSON.stringify({ok: true})).setMimeType(ContentService.MimeType.JSON);\n}', language="javascript")
    else:
        st.session_state.setdefault("fb_opened", time.time())
        with st.form("feedback_form", clear_on_submit=False):
            fb_category = st.radio("What would you like to share?", FEEDBACK_CATEGORIES, index=0)
            fb_message = st.text_area("Your message", max_chars=2000, height=150, placeholder="What happened, what you expected, or what you would like to see. Steps to reproduce help for bugs.")
            fb_email = st.text_input("Email (optional, only if you'd like a reply)", max_chars=200)
            fb_attach = st.checkbox("Attach my current model settings (no personal data) so a problem is easier to reproduce", value=True)
            st.caption("Your message is used only to improve this project. Please don't include confidential or personal financial details.")
            fb_submit = st.form_submit_button("Send feedback")
        if fb_submit:
            _err = validate_feedback(fb_category, fb_message, fb_email)
            if _err:
                st.error(_err)
            elif st.session_state.get("fb_count", 0) >= 5:
                st.warning("You've sent several messages in this session, thank you! Please try again later.")
            elif time.time() - st.session_state["fb_opened"] < 2:
                st.success("Thank you for your feedback!")   # too fast to be a person: acknowledge, send nothing
            else:
                _settings = None
                if fb_attach:
                    _settings = {"starting_profile": founder_stage, "horizon_months": H, "advanced_mode": bool(advanced_mode), "free_tier_booked_to": P["free_treatment"],
                                 "fixed_infra_to_free_tier": P["infra_alloc"], "trial_share": P["trial_share"], "free_tier_frontier_mix": P["front_mix_free"],
                                 "internal_traffic_pct": P["internal_pct"], "ai_cost_path": ai_path_text(P, H), "other_cogs_lines": len(P["other_cogs"]),
                                 "reported_gm_pct": None if pd.isna(tot["gm"]) else round(tot["gm"], 1), "ending_arr": round(tot["arr"])}
                _payload = {"submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "app": APP_NAME, "app_version": APP_VERSION,
                            "category": fb_category, "message": safe_cell(clean_text(fb_message, 2000)), "email": safe_cell(clean_text(fb_email, 200)), "settings": _settings}
                _ok, _msg = post_feedback(_fb["webhook_url"], _payload)
                if _ok:
                    st.session_state["fb_count"] = st.session_state.get("fb_count", 0) + 1
                    st.success("Thank you! Your feedback was sent.")
                else:
                    st.error(f"{_msg} Your message was not sent." + (f" You can also {' or '.join(_fallbacks)}." if _fallbacks else ""))

# ==========================================
# 5. FOOTER, CHANGE LOG & LIMITATIONS
# ==========================================
st.markdown("---")
SHOW_CHANGE_LOG = False
CHANGE_LOG_MD = """
**v2.9 changes**
1. **Guided story view (default) and Analyst view.** The Guided story opens with a hero, Chapter 0 (pick a profile, see paid-customer, reported and fully loaded margin side by side) and Chapter 1 (what one prompt costs, with the prompts-per-day control). All remaining tabs sit under "Go deeper". Analyst view is the previous layout. Both views use the same engine, so the numbers are identical.
2. **New visual theme** (light grey page, near-white cards, deep navy and one strong blue; matching dark mode). Status pills use separate green, amber and red.
3. **Each setting has one home.** In the guided view the profile picker and the prompts-per-day slider live in the chapters, not the sidebar; in Analyst view they stay in the sidebar.

**v2.8 changes**
1. **Deal Desk tab.** Prices one customer deal (seats, discount, term, billing, usage, optional overage and cap, dedicated cost) with the same per-prompt AI cost as the main model, and returns Approve / Approve with conditions / Escalate against a margin floor.
2. **Break-evens** for maximum prompts per seat, minimum net price and maximum discount, a usage / model-mix / vendor-price stress table, a monthly deal P&L and cash view, and JSON / CSV export.
3. **One AI cost function.** The simulation and the Deal Desk now share `api_cost_for`, so they cannot drift apart. Main-model results are unchanged.

**v2.7 changes**
1. **Traffic mix.** Free-tier traffic can be split into trials and freemium (sidebar 3) and can be routed to a cheaper model mix; internal / eval traffic can be modeled and booked to R&D / G&A or COGS (Advanced > F). A traffic-by-type table shows paid vs trial vs freemium vs internal.
2. **Sensitivity and goal seek.** A tornado shows which assumptions move your chosen metric most; goal seek solves for the lever value that hits a target margin; a usage-curve chart and sanity warnings flag implausible inputs.
3. **CAC, LTV and payback** count free-tier inference as acquisition cost (once), with a simplified, directional method.
4. **Save, compare and export.** Save scenarios in-session, compare them side by side, and export JSON, CSV or an Excel workbook. Text beginning with = + - @ is neutralised so spreadsheets never run it as a formula.
5. **In-app feedback form** (Feedback tab). It sends to a destination you configure in Streamlit secrets; until then it shows a fallback.
6. **Fix:** the per-account usage ceiling now prices paying-customer traffic only, so a cheaper free-tier model no longer understates paid cost per prompt.

**v2.6 changes**
1. **Free / trial inference cost can be booked to COGS or to sales & marketing** (sidebar 3). The default is COGS, so headline numbers are unchanged. Free-tier cost is now its own sub-line, and paying customers' margin never carries it.
2. **Three margins side by side:** paid-customer gross margin, gross margin as reported (after your COGS / S&M choice) and fully loaded gross margin, with a margin bridge on the Revenue & Margin tab.
3. **Other COGS lines for a fully loaded margin.** An editable table above the KPIs takes $ / month or % of revenue lines with start month, end month and monthly growth. Nothing is added by default.
4. **AI cost path (sidebar 3).** A MoM reduction in AI cost per prompt from a chosen month (with a cumulative cap) and a one-time vendor price change (cut or increase). Defaults are off, so prices stay flat unless you set it.
5. **Per-account cost fixed.** "COGS per paid account" now uses paid-customer COGS only; the version that included free-tier cost is shown as a memo. Blended cost per 1,000 prompts uses total direct cost.
6. **Fixed infrastructure** now stays with paid customers by default (Advanced > E lets you split it by prompt share, the v2.5 behaviour). Variable free-tier cost still follows prompt share. This changes the Value & Pricing tab's free-tier figures versus v2.5; headline gross margin is unchanged.
7. **Dark mode.** Colours now follow the active Streamlit theme (or the operating-system setting when the theme is not reported), and charts switch to a dark template.
8. **New integrity checks** cover the cost split, free-tier booking, fully loaded reconciliation, margin ordering and the AI cost path.

**Known limitations**
- **Token prices are illustrative placeholders.** They are flat unless you set an AI cost path; the path is a planning assumption, not a vendor commitment, and applies to AI inference (API) cost only, not vector DB or hosting. Scenarios 8-9 apply one flat % change to every month.
- **Input tokens per prompt grow monthly,** which pushes cost per prompt up unless offset by an efficiency path.
- **Cache and batch discounts** are vendor-dependent and assumed to stay available at the configured levels.
- **Other COGS lines are display-only:** they change the fully loaded margin but not cash payable, credits or the usage ceiling.
- **Accounting treatment is a policy choice.** The COGS / S&M switch is a presentation of the same cost; confirm your policy with your accountant or auditor.
- **Margins exclude any cost you do not enter,** so modeled margin is higher than a fully loaded margin unless you add lines.

**Material changes from v1**
1. **Team / SMB engine rebuilt as cohorts.** Accounts, renewals, non-renewals, in-term churn and expansion are tracked per cohort.
2. **Expansion is no longer double counted.** Team / SMB expansion grows contract value, seats and prompts of existing accounts.
"""
if SHOW_CHANGE_LOG:
    with st.expander("Change log & limitations (v2.9)"):
        st.markdown(CHANGE_LOG_MD)
else:
    _hidden = html.escape(CHANGE_LOG_MD.strip()).replace("\n\n", "<br>").replace("\n", "<br>")
    st.markdown(f'<div id="change-log-and-limitations" aria-hidden="true" style="display:none">{_hidden}</div>', unsafe_allow_html=True)

# Always-visible disclaimers (independent of SHOW_CHANGE_LOG). Built from live inputs so they cannot quote stale values.
_in_mult = (1 + P["in_grow"]) ** max(0, H - 1)
st.markdown(f"""
<div class="assumption-flag"><b>Key assumptions & limitations</b>
<ul>
<li>{token_price_note(P, H)}</li>
<li><b>Token volume:</b> input tokens per prompt grow {P['in_grow'] * 100:.1f}% per month (x{_in_mult:.2f} by month {H}) and output tokens {P['out_grow'] * 100:.1f}% per month, against {'flat prices, so cost per prompt rises over the horizon' if ai_path_is_flat(P) else 'the AI cost path above, which offsets or adds to that growth'}.</li>
<li><b>Discounts:</b> caching ({P['cache_rate'] * 100:.0f}% of eligible input at {P['cache_ratio'] * 100:.0f}% of list price) and batch routing ({P['batch_share'] * 100:.0f}% of traffic at {P['batch_ratio'] * 100:.0f}% of list price) depend on vendor terms and are assumed to stay available throughout.</li>
<li><b>Price scenarios:</b> scenarios 8-9 apply one flat % price change to every month; for a gradual path use the AI cost path in the sidebar.</li>
<li><b>Cost treatment:</b> free / trial inference cost is booked to {TREAT_LABEL[P['free_treatment']]} (a policy choice; confirm with your accountant or auditor). Paying customers' margin never carries it.</li>
<li><b>Fully loaded margin:</b> {('includes ' + str(len(P['other_cogs'])) + ' user-defined COGS line(s); they do not affect cash timing or credits.') if P['other_cogs'] else 'no other COGS lines entered, so margins cover AI-direct costs only.'}</li>
<li><b>Presets:</b> founder presets are planning estimates, not benchmarks. Replace them with actuals from your billing system and LLM gateway logs.</li>
</ul></div>
""", unsafe_allow_html=True)

st.caption("© 2026. The AI Unit Economics Deal Desk. All rights reserved.")
