"""
app.py — Streamlit UI for AutoResearch AI.

Provides two tabs:
    Research       : Single brand URL input → animated 6-step progress → 8-section report.
    Compare Brands : Up to 3 URLs researched sequentially, reports shown side by side.

Live UI updates are driven by LangGraph's stream() output — no polling, no sleep().
All timing is measured with time.time() so durations reflect real network latency.
HTML is rendered inline via st.markdown(unsafe_allow_html=True) because Streamlit
has no native progress timeline component. All HTML strings are produced by pure
functions (_header_html, _timeline_html, etc.) for testability.
"""

import re
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from agent import build_graph, make_initial_state

load_dotenv()

st.set_page_config(page_title="AutoResearch AI", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────────
# COLOR CONSTANTS
# ─────────────────────────────────────────────
C_GREEN  = "#22C55E"
C_PURPLE = "#6C63FF"
C_GRAY   = "#9CA3AF"
C_ORANGE = "#F59E0B"
C_RED    = "#EF4444"
C_BG     = "#0f1117"
C_BORDER = "#262730"

# ─────────────────────────────────────────────
# NODE DEFINITIONS
# ─────────────────────────────────────────────
NODE_ORDER = [
    "identify_brand",
    "search",
    "scrape",
    "check_sufficiency",
    "generate_report",
    "store_memory",
]
NODE_LABELS = {
    "identify_brand":    "Identifying Brand",
    "search":            "Searching Competitors",
    "scrape":            "Scraping Websites",
    "check_sufficiency": "Checking Sufficiency",
    "generate_report":   "Generating Report",
    "store_memory":      "Saving to Memory",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AutoResearch AI")
    st.caption("Autonomous Marketing Intelligence Agent · Built by Bharadwaj R")
    st.divider()
    st.markdown("""
**What it does:**
1. Scrapes brand URL — extracts name & logo
2. Searches competitors via Tavily (parallel)
3. Scrapes competitor pages
4. Loops until data is sufficient (max 3×)
5. Generates 8-section intelligence report

**Observability:**
- 📊 LangSmith — full agent traces
- 📈 Langfuse — LLM logs + prompt management
    """)
    st.divider()
    sidebar_stats_box = st.empty()   # live stats — updated during research
    st.divider()
    st.caption("LangGraph · Groq · Tavily · Chroma · LangSmith · Langfuse")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Research", "⚔️ Compare Brands"])


# ═══════════════════════════════════════════════════════════════
# HTML RENDERERS — all return strings, rendered via markdown()
# ═══════════════════════════════════════════════════════════════

def _header_html(label: str, pct: int, elapsed: float) -> str:
    """Render the top progress bar with current node label and elapsed time."""
    return f"""
<div style="background:{C_BG}; border:1px solid {C_PURPLE}66; border-radius:12px; padding:20px 24px; margin:4px 0;">
  <div style="font-size:20px; font-weight:700; color:{C_PURPLE}; margin-bottom:10px;">
    🔄 &nbsp;{label}
  </div>
  <div style="background:#1e1e2e; border-radius:6px; height:6px; margin-bottom:10px; overflow:hidden;">
    <div style="width:{pct}%; background:linear-gradient(90deg,{C_PURPLE},{C_GREEN}); height:100%; border-radius:6px;"></div>
  </div>
  <table style="width:100%; font-size:13px; color:{C_GRAY};"><tr>
    <td>{pct}% complete</td>
    <td style="text-align:right;">⏱️ {elapsed:.0f}s elapsed</td>
  </tr></table>
</div>"""


def _completion_html(brand_name: str, total: float, result: dict) -> str:
    """Render the green completion banner shown after all nodes finish."""
    sources = len(result.get("search_results", []))
    scraped = len(result.get("scraped_content", []))
    iters   = result.get("iterations", 0)
    return f"""
<div style="background:linear-gradient(135deg,#0d2318,{C_BG}); border:2px solid {C_GREEN}; border-radius:12px; padding:24px; text-align:center; margin:4px 0;">
  <div style="font-size:42px; margin-bottom:6px;">✅</div>
  <div style="font-size:18px; font-weight:700; color:{C_GREEN}; margin-bottom:6px;">Research Complete!</div>
  <div style="color:{C_GRAY}; font-size:13px;">
    Finished in <b style="color:#e2e8f0;">{total:.0f}s</b>
    &nbsp;·&nbsp; {sources} sources
    &nbsp;·&nbsp; {scraped} pages scraped
    &nbsp;·&nbsp; {iters} iteration(s)
  </div>
</div>"""


def _timeline_html(completed: dict, active: str, iterations: int) -> str:
    """
    Render the 6-node step tracker.

    Each node shows: ✅ done (with duration) | 🔄 running | ⏳ pending.
    An iteration badge appears when the agent loops past iteration 1.
    """
    rows = []
    for node in NODE_ORDER:
        label = NODE_LABELS[node]
        if node in completed:
            dur = f"{completed[node]:.1f}s"
            rows.append(f"""
<tr>
  <td style="color:{C_GREEN}; padding:5px 0; font-size:14px;">✅ &nbsp;<b>{label}</b></td>
  <td style="color:{C_GRAY}; text-align:right; font-size:12px; white-space:nowrap;">{dur}</td>
</tr>""")
        elif node == active:
            rows.append(f"""
<tr>
  <td style="color:{C_PURPLE}; padding:5px 0; font-size:14px;">🔄 &nbsp;<b>{label}...</b></td>
  <td style="color:{C_GRAY}; text-align:right; font-size:12px;">running</td>
</tr>""")
        else:
            rows.append(f"""
<tr>
  <td style="color:{C_GRAY}; padding:5px 0; font-size:14px;">⏳ &nbsp;{label}</td>
  <td></td>
</tr>""")

    iter_row = ""
    if iterations > 1:
        iter_row = f"""
<tr><td colspan="2" style="padding-top:8px;">
  <div style="padding:6px 12px; background:{C_ORANGE}22; border-left:3px solid {C_ORANGE}; color:{C_ORANGE}; border-radius:0 4px 4px 0; font-size:13px;">
    🔁 &nbsp;Iteration {iterations} of 3 — collecting more data...
  </div>
</td></tr>"""

    return f"""
<div style="background:{C_BG}; border:1px solid {C_BORDER}; border-radius:8px; padding:12px 16px; margin:4px 0;">
  <table style="width:100%; border-collapse:collapse;">
    {"".join(rows)}{iter_row}
  </table>
</div>"""


def _activity_html(activity_log: list) -> str:
    """
    Render the live activity feed (monospace, last 5 entries).

    Shows timestamped status_log messages from the agent state.
    Capped at 5 lines to avoid the feed growing too tall during long runs.
    """
    if not activity_log:
        return ""
    recent = activity_log[-5:]
    lines = "".join(
        f'<div><span style="color:{C_GRAY};">[{ts}]</span>&nbsp;<span style="color:#cbd5e1;">{msg[:95]}</span></div>'
        for ts, msg in recent
    )
    return f"""
<div style="font-family:monospace; font-size:12px; background:#080b10; border:1px solid {C_BORDER}; border-radius:6px; padding:10px 14px; margin:4px 0; line-height:1.9;">
  <div style="color:{C_GRAY}; font-size:10px; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Live Activity</div>
  {lines}
</div>"""


def _stats_html(stats: dict) -> str:
    """Render the live stats table shown in the sidebar during and after research."""
    rows = [
        ("🔗 URLs searched", stats.get("urls_searched", 0)),
        ("📄 Pages scraped",  stats.get("pages_scraped", 0)),
        ("🔁 Iterations",     stats.get("iterations", 0)),
        ("🤖 LLM calls",      stats.get("llm_calls", 0)),
        ("⏱️ Time elapsed",   f"{stats.get('elapsed', 0):.0f}s"),
    ]
    inner = "".join(
        f'<tr>'
        f'<td style="color:{C_GRAY}; font-size:13px; padding:3px 0;">{lbl}</td>'
        f'<td style="color:#e2e8f0; font-size:13px; font-weight:600; text-align:right;">{val}</td>'
        f'</tr>'
        for lbl, val in rows
    )
    return f"""
<div style="background:{C_BG}; border:1px solid {C_BORDER}; border-radius:8px; padding:12px 14px;">
  <div style="color:{C_PURPLE}; font-weight:700; font-size:13px; margin-bottom:8px;">📊 Research Stats</div>
  <table style="width:100%; border-collapse:collapse;">{inner}</table>
</div>"""


# ─────────────────────────────────────────────
# BRAND CARD RENDERER
# ─────────────────────────────────────────────
def _render_brand_card(container, state: dict):
    brand    = state.get("brand_name", "")
    website  = state.get("brand_website", "")
    industry = state.get("brand_industry", "")
    logo_url = state.get("brand_logo", "")

    def _card_text():
        lines = [f"### ✅ {brand}" if brand else "### ✅ Brand Identified"]
        if website:
            lines.append(f"🌐 **Website:** {website}")
        if industry:
            lines.append(f"🏢 **Industry:** {industry}")
        st.success("  \n".join(lines))

    with container.container():
        if logo_url:
            cols = st.columns([1, 9])
            with cols[0]:
                try:
                    st.image(logo_url, width=64)
                except Exception:
                    pass
            with cols[1]:
                _card_text()
        else:
            _card_text()


# ─────────────────────────────────────────────
# AGENT RUNNER — streams LangGraph, updates all UI containers in real time
# ─────────────────────────────────────────────
def run_agent(brand_url: str, containers: dict, sidebar_stats) -> dict | None:
    """
    containers keys: header, timeline, activity, brand_card
    sidebar_stats   : st.empty() in sidebar for live stats
    """
    graph   = build_graph()
    initial = make_initial_state(brand_url.strip())

    start_time    = time.time()
    last_node_end = start_time
    completed     = {}         # node_name → duration (seconds)
    active_node   = NODE_ORDER[0]
    activity_log  = []         # [(hh:mm:ss, message), ...]
    seen_msgs     = set()      # deduplicate activity feed
    stats = {
        "urls_searched": 0,
        "pages_scraped": 0,
        "iterations":    0,
        "llm_calls":     0,
        "elapsed":       0,
    }

    def _refresh(state: dict, active: str):
        elapsed = time.time() - start_time
        pct     = min(int(len(completed) / len(NODE_ORDER) * 100), 95)
        iters   = state.get("iterations", 0)

        # Collect new status log entries
        for msg in state.get("status_log", []):
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                activity_log.append((datetime.now().strftime("%H:%M:%S"), msg))

        stats.update({
            "urls_searched": len(state.get("search_results", [])),
            "pages_scraped": len(state.get("scraped_content", [])),
            "iterations":    iters,
            "elapsed":       elapsed,
        })

        label = NODE_LABELS.get(active, active) if active else "Finishing..."
        containers["header"].markdown(_header_html(label, pct, elapsed), unsafe_allow_html=True)
        containers["timeline"].markdown(_timeline_html(completed, active, iters), unsafe_allow_html=True)
        if activity_log:
            containers["activity"].markdown(_activity_html(activity_log), unsafe_allow_html=True)
        sidebar_stats.markdown(_stats_html(stats), unsafe_allow_html=True)

    # Initial render
    containers["header"].markdown(_header_html("Starting...", 0, 0), unsafe_allow_html=True)
    containers["timeline"].markdown(_timeline_html({}, active_node, 0), unsafe_allow_html=True)

    result = None
    try:
        for step in graph.stream(initial):
            node_name = list(step.keys())[0]
            state     = list(step.values())[0]
            now       = time.time()

            # Record node duration
            completed[node_name] = now - last_node_end
            last_node_end = now

            # Advance active pointer
            try:
                idx = NODE_ORDER.index(node_name)
                active_node = NODE_ORDER[idx + 1] if idx + 1 < len(NODE_ORDER) else ""
            except ValueError:
                active_node = ""

            # Count LLM calls
            if node_name in ("identify_brand", "check_sufficiency", "generate_report"):
                stats["llm_calls"] += 1

            # Brand card — appears immediately after brand identification
            if node_name == "identify_brand" and state.get("brand_name"):
                _render_brand_card(containers["brand_card"], state)

            _refresh(state, active_node)
            result = state

        # Completion banner
        total = time.time() - start_time
        stats["elapsed"] = total
        containers["header"].markdown(
            _completion_html(result.get("brand_name", "Brand"), total, result),
            unsafe_allow_html=True,
        )
        containers["timeline"].markdown(
            _timeline_html(completed, "", result.get("iterations", 0)),
            unsafe_allow_html=True,
        )
        sidebar_stats.markdown(_stats_html(stats), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error researching '{brand_url}': {e}")
        st.exception(e)

    return result


# ─────────────────────────────────────────────
# REPORT RENDERER
# ─────────────────────────────────────────────
def display_report(result: dict, brand_name: str):
    if not result or not result.get("final_report"):
        st.error("No report generated.")
        return

    _render_brand_card(st, result)

    if result.get("has_live_data"):
        st.success(
            f"Research complete · {result['iterations']} iteration(s) · "
            f"{len(result.get('search_results', []))} sources · "
            f"{len(result.get('scraped_content', []))} pages scraped · "
            f"{len(result.get('reddit_insights', []))} Reddit threads · "
            f"{len(result.get('news_insights', []))} news articles"
        )
    else:
        st.warning("⚠️ No live data retrieved. Report is AI estimate from training knowledge.")

    st.divider()

    SECTIONS = [
        ("1.", "🏆 Top Competitors Identified"),
        ("2.", "🔍 Competitor Analysis"),
        ("3.", "📊 Competitor Scoring"),
        ("4.", "🎯 SWOT Analysis"),
        ("5.", "💬 Reddit Sentiment"),
        ("6.", "🚀 Market Positioning Gaps"),
        ("7.", "💡 Key Insights"),
        ("8.", "✅ Recommended Actions"),
    ]

    report = result["final_report"]
    parts  = re.split(r'\n##\s+', "\n" + report)
    section_map = {}
    for part in parts[1:]:
        for num, _ in SECTIONS:
            if part.strip().startswith(num):
                section_map[num] = part.strip()
                break

    for i, (num, label) in enumerate(SECTIONS):
        content = section_map.get(num, "")
        with st.expander(label, expanded=(i == 0)):
            if content:
                st.markdown("\n".join(content.split("\n")[1:]).strip())
            else:
                st.markdown("*Section not found in report.*")

    st.divider()
    with st.expander(f"News Articles ({len(result.get('news_insights', []))} found)"):
        for r in result.get("news_insights", []):
            st.markdown(f"**[{r['title']}]({r['url']})**")
            st.caption(r["content"][:200])
            st.divider()

    with st.expander(f"Pricing Intelligence ({len(result.get('pricing_data', []))} found)"):
        for r in result.get("pricing_data", []):
            st.markdown(f"**[{r['title']}]({r['url']})**")
            st.caption(r["content"][:200])
            st.divider()


# ═══════════════════════════════════════════════
# TAB 1 — SINGLE BRAND RESEARCH
# ═══════════════════════════════════════════════
with tab1:
    st.header("Brand Competitor Research")
    st.caption("Paste any brand URL — the agent identifies the brand, researches competitors, and generates a full intelligence report.")

    col1, col2 = st.columns([4, 1])
    with col1:
        url_input = st.text_input(
            "Brand URL",
            placeholder="e.g. https://notion.so",
            key="single_url",
        )
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("Run Research", type="primary", use_container_width=True, key="run_single")

    if run_btn:
        raw = url_input.strip()
        if not raw:
            st.warning("Please enter a brand URL.")
        elif " " in raw or ("." not in raw and not raw.startswith(("http://", "https://"))):
            st.error("Please enter a URL (e.g. https://notion.so), not just a brand name.")
        else:
            # Create all live update containers
            header_box   = st.empty()
            timeline_box = st.empty()
            activity_box = st.empty()
            brand_card   = st.empty()

            result = run_agent(
                raw,
                containers={
                    "header":     header_box,
                    "timeline":   timeline_box,
                    "activity":   activity_box,
                    "brand_card": brand_card,
                },
                sidebar_stats=sidebar_stats_box,
            )

            # Clear streaming brand card — display_report renders it permanently
            brand_card.empty()

            if result:
                display_report(result, result.get("brand_name", raw))


# ═══════════════════════════════════════════════
# TAB 2 — COMPARE BRANDS
# ═══════════════════════════════════════════════
with tab2:
    st.header("Compare Up to 3 Brands Side by Side")
    st.caption("Paste each brand's URL — the agent extracts names and researches each independently.")

    c1, c2, c3 = st.columns(3)
    with c1: url_a = st.text_input("Brand 1 URL", placeholder="e.g. https://notion.so",   key="brand_a")
    with c2: url_b = st.text_input("Brand 2 URL", placeholder="e.g. https://linear.app",  key="brand_b")
    with c3: url_c = st.text_input("Brand 3 URL (optional)", placeholder="e.g. https://coda.io", key="brand_c")

    compare_btn = st.button("Compare Brands", type="primary", key="compare")

    if compare_btn:
        urls    = [u.strip() for u in [url_a, url_b, url_c] if u.strip()]
        invalid = [u for u in urls if " " in u or ("." not in u and not u.startswith(("http://", "https://")))]
        if invalid:
            st.error(f"Please enter URLs, not brand names: {', '.join(invalid)}")
        elif len(urls) < 2:
            st.warning("Please enter at least 2 brand URLs to compare.")
        else:
            results = {}
            for url in urls:
                st.subheader(f"Researching: {url}")
                c_header   = st.empty()
                c_timeline = st.empty()
                c_activity = st.empty()
                c_brand    = st.empty()
                c_stats    = st.empty()   # per-brand stats (not sidebar)
                res = run_agent(
                    url,
                    containers={
                        "header":     c_header,
                        "timeline":   c_timeline,
                        "activity":   c_activity,
                        "brand_card": c_brand,
                    },
                    sidebar_stats=sidebar_stats_box,
                )
                # Collapse progress after each brand
                c_header.empty()
                c_timeline.empty()
                c_activity.empty()
                c_brand.empty()
                if res:
                    results[res.get("brand_name", url)] = res
                st.divider()

            if results:
                st.subheader("Comparison Results")
                cols = st.columns(len(results))
                for col, (brand, res) in zip(cols, results.items()):
                    with col:
                        st.markdown(f"### {brand}")
                        if res.get("has_live_data"):
                            st.success(f"{len(res.get('search_results', []))} sources")
                        else:
                            st.warning("AI estimate")
                        with st.expander("Full Report", expanded=True):
                            st.markdown(res["final_report"])


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("AutoResearch AI · Built by Bharadwaj R · LangGraph + LangSmith + Langfuse + Groq + Tavily + Chroma · All data from public web")
