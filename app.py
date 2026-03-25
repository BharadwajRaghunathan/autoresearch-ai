"""
app.py — Streamlit UI for AutoResearch AI.

Three tabs:

    Research          : Single brand URL → animated 6-step progress → 8-section
                        competitor intelligence report.

    Compare Brands    : Up to 3 URLs researched in PARALLEL using Python threading.
                        Total time = slowest single brand, not sum of all.
                        Includes cross-brand LLM summary after all brands complete.

    Ad Creative       : Paste any landing page URL → deep creative extraction →
    Analyzer            7-section creative intelligence report + 5-dimension scorecard.

Live UI updates are driven by LangGraph stream() events (Research / Creative tabs)
and a 0.5s polling loop over a shared status dict (Compare tab).
All HTML strings are produced by pure renderer functions for testability.
"""

import os
import re
import time
import threading
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# ── Secret injection (must happen BEFORE any downstream imports) ─────────────
# Locally   : load_dotenv() in each module reads .env — works as before.
# Cloud     : Streamlit Cloud has no .env file. Secrets live in the app's
#             Secrets panel and are accessed via st.secrets. We push them into
#             os.environ here so every subsequent os.getenv() call in chains.py,
#             tools.py, and memory.py finds the right value.
# Rule      : never overwrite a value that dotenv or the host already set.
load_dotenv()  # no-op on Cloud; populates env on local dev
_SECRETS = [
    "GROQ_API_KEY", "TAVILY_API_KEY",
    "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    "SARVAM_API_KEY",
]
for _k in _SECRETS:
    if not os.environ.get(_k):
        try:
            os.environ[_k] = str(st.secrets[_k])
        except (KeyError, Exception):
            pass  # secret not configured — downstream code degrades gracefully

from langchain_core.messages import HumanMessage
from agent import (
    build_graph, make_initial_state,
    build_creative_graph, make_creative_state,
    voice_summary_node,
)
from chains import llm, langfuse_handler

# Voice deps are optional — imported lazily so missing packages don't crash startup
try:
    from voice import transcribe_sarvam, speak_sarvam
    _VOICE_AVAILABLE = True
except ImportError:
    _VOICE_AVAILABLE = False
    VOICE_OPTIONS = {"Guy (US Male)": "en-US-GuyNeural"}

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
    "trend_compare",
    "store_memory",
]
NODE_LABELS = {
    "identify_brand":    "Identifying Brand",
    "search":            "Searching Competitors",
    "scrape":            "Scraping Websites",
    "check_sufficiency": "Checking Sufficiency",
    "generate_report":   "Generating Report",
    "trend_compare":     "Trend Comparison",
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
**Features:**
- 🔍 **Research** — paste URL → 8-section competitor report
- ⚔️ **Compare** — up to 3 brands in parallel + cross-brand summary
- 🎨 **Creative Decoder** — decode any landing page creative strategy
- 🎙️ **Voice** — speak a brand → agent researches & reads findings aloud

**How the research agent works:**
1. Scrapes brand URL — extracts name, logo & industry
2. Searches competitors via Tavily (parallel)
3. Scrapes competitor pages
4. Loops until data is sufficient (max 3×)
5. Generates 8-section intelligence report
6. Compares with last run — surfaces what changed
7. Saves to Chroma memory for future trend tracking

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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research", "⚔️ Compare Brands", "🎨 Creative Decoder", "🎙️ Voice Research"])


# ═══════════════════════════════════════════════════════════════
# HTML RENDERERS — Tab 1 (single brand)
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


# ═══════════════════════════════════════════════════════════════
# NODE DEFINITIONS — Tab 3 (creative analyzer)
# ═══════════════════════════════════════════════════════════════
CREATIVE_NODE_ORDER = [
    "scrape_creative",
    "analyse_creative",
    "score_creative",
    "verdict_creative",
    "store_creative_memory",
]
CREATIVE_NODE_LABELS = {
    "scrape_creative":       "Scraping Creative Elements",
    "analyse_creative":      "Analysing Creative Strategy",
    "score_creative":        "Scoring Creative",
    "verdict_creative":      "Writing Verdict",
    "store_creative_memory": "Saving to Memory",
}

INDUSTRIES = [
    "AI Marketing",
    "SaaS / Productivity",
    "E-commerce",
    "Fintech",
    "Healthcare",
    "EdTech",
    "Developer Tools",
    "Other",
]


# ═══════════════════════════════════════════════════════════════
# HTML RENDERERS — Tab 3 (creative analyzer)
# ═══════════════════════════════════════════════════════════════

def _confidence_html(result: dict) -> str:
    """
    Render a small data-quality indicator beneath the scorecard.

    Three states:
      ⚠️  word_count < 200 — JS-rendered, Tavily fallback used
      ✅  word_count ≥ 200, single page
      ✅  multi-page (always positive regardless of word count)
    """
    word_count   = result.get("word_count", 0)
    pages        = result.get("pages_scraped", [])
    used_tavily  = result.get("used_tavily_fallback", False)
    pages_label  = " · ".join(pages) if pages else "homepage"
    n_pages      = len(pages)

    if used_tavily:
        icon  = "⚠️"
        color = C_ORANGE
        msg   = f"JS-rendered site — Tavily fallback used · {word_count} words"
    elif n_pages >= 2:
        icon  = "✅"
        color = C_GREEN
        msg   = f"{n_pages} pages analysed — {word_count} words"
    else:
        icon  = "✅" if word_count >= 300 else "⚠️"
        color = C_GREEN if word_count >= 300 else C_ORANGE
        msg   = f"Single page · {word_count} words" + (" — consider pasting the /pricing URL for richer analysis" if word_count < 300 else "")

    return f"""
<div style="display:flex; align-items:center; gap:10px; padding:8px 14px; background:{C_BG}; border:1px solid {color}44; border-radius:6px; margin:4px 0;">
  <span style="font-size:14px;">{icon}</span>
  <span style="color:{C_GRAY}; font-size:12px;">
    <b style="color:{color};">Pages analysed:</b> {pages_label}
    &nbsp;·&nbsp; {msg}
  </span>
</div>"""


def _verdict_html(verdict: str) -> str:
    """Render the one-sentence verdict in a highlighted box above the scorecard."""
    return f"""
<div style="background:linear-gradient(135deg,#1a1040,{C_BG}); border:1px solid {C_PURPLE}88; border-radius:10px; padding:16px 20px; margin:8px 0;">
  <div style="font-size:11px; font-weight:700; color:{C_PURPLE}; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;">⚡ Verdict</div>
  <div style="font-size:15px; color:#e2e8f0; font-style:italic; line-height:1.6;">"{verdict}"</div>
</div>"""


def _scorecard_html(scores: dict) -> str:
    """
    Render the 5-dimension creative scorecard as horizontal progress bars
    with a one-line evidence reason shown below each bar.

    Color coding: green ≥7, orange ≥5, red <5.
    Reasons come from LLM-populated *_reason keys in the scores dict.
    """
    items = [
        ("Clarity of Value Prop",  scores.get("clarity", 0),          scores.get("clarity_reason", "")),
        ("Emotional Impact",       scores.get("emotional_impact", 0),  scores.get("emotional_impact_reason", "")),
        ("CTA Effectiveness",      scores.get("cta_effectiveness", 0), scores.get("cta_effectiveness_reason", "")),
        ("Trust Signals",          scores.get("trust_signals", 0),     scores.get("trust_signals_reason", "")),
        ("Overall Creative Score", scores.get("overall", 0),           ""),
    ]
    rows = ""
    for label, raw_score, reason in items:
        score  = max(0.0, min(10.0, float(raw_score) if raw_score else 0))
        pct    = score / 10 * 100
        color  = C_GREEN if score >= 7 else C_ORANGE if score >= 5 else C_RED
        reason_row = ""
        if reason:
            reason_row = f'<tr><td colspan="3" style="padding:0 0 6px 0; color:{C_GRAY}; font-size:11px; font-style:italic;">&nbsp;&nbsp;"{reason}"</td></tr>'
        rows += f"""
<tr>
  <td style="color:#e2e8f0; font-size:13px; padding:6px 0 3px 0; width:38%; white-space:nowrap;">{label}</td>
  <td style="padding:6px 0 3px 14px;">
    <div style="background:#1e1e2e; border-radius:4px; height:8px; overflow:hidden;">
      <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:4px;"></div>
    </div>
  </td>
  <td style="color:{color}; font-size:13px; font-weight:700; text-align:right; padding:6px 0 3px 10px; white-space:nowrap;">{score:.1f}/10</td>
</tr>{reason_row}"""

    return f"""
<div style="background:{C_BG}; border:1px solid {C_PURPLE}55; border-radius:12px; padding:20px 24px; margin:8px 0;">
  <div style="font-size:16px; font-weight:700; color:{C_PURPLE}; margin-bottom:14px;">🎯 Creative Scorecard</div>
  <table style="width:100%; border-collapse:collapse;">{rows}</table>
</div>"""


# ═══════════════════════════════════════════════════════════════
# HTML RENDERERS — Tab 2 (parallel compare)
# ═══════════════════════════════════════════════════════════════

def _url_label(url: str) -> str:
    """Short display name from URL before the brand name is known."""
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc or url
        return netloc.lstrip("www.")
    except Exception:
        return url


def _overall_progress_html(total: int, done: int, elapsed: float) -> str:
    """Banner showing how many brands have finished and total elapsed time."""
    pct = int(done / total * 100) if total else 0
    return f"""
<div style="background:{C_BG}; border:1px solid {C_PURPLE}66; border-radius:12px; padding:16px 20px; margin:4px 0;">
  <div style="font-size:16px; font-weight:700; color:{C_PURPLE}; margin-bottom:8px;">
    🔄 &nbsp;Researching {total} brand{"s" if total > 1 else ""} simultaneously...
  </div>
  <div style="background:#1e1e2e; border-radius:6px; height:4px; margin-bottom:8px; overflow:hidden;">
    <div style="width:{pct}%; background:linear-gradient(90deg,{C_PURPLE},{C_GREEN}); height:100%; border-radius:6px;"></div>
  </div>
  <div style="color:{C_GRAY}; font-size:13px;">
    {done} of {total} complete &nbsp;·&nbsp; ⏱️ {elapsed:.0f}s elapsed
  </div>
</div>"""


def _all_done_html(total: int, elapsed: float) -> str:
    """Green banner shown once every brand thread has finished."""
    return f"""
<div style="background:linear-gradient(135deg,#0d2318,{C_BG}); border:2px solid {C_GREEN}; border-radius:12px; padding:20px 24px; text-align:center; margin:4px 0;">
  <div style="font-size:36px; margin-bottom:4px;">✅</div>
  <div style="font-size:16px; font-weight:700; color:{C_GREEN}; margin-bottom:4px;">
    All {total} brands researched in {elapsed:.0f}s
  </div>
  <div style="color:{C_GRAY}; font-size:13px;">
    Parallel execution — {total}× faster than sequential research
  </div>
</div>"""


def _compare_col_html(status: dict, url: str) -> str:
    """
    Compact per-column progress card for one brand during parallel research.

    Shows the 6-node timeline with live ✅/🔄/⏳ indicators.
    Error state renders a red card instead of the timeline.
    """
    if not status:
        label = _url_label(url)
        return f"""
<div style="background:{C_BG}; border:1px solid {C_BORDER}; border-radius:8px; padding:12px 14px;">
  <div style="font-size:13px; font-weight:600; color:{C_GRAY};">{label}</div>
  <div style="color:{C_GRAY}; font-size:12px; margin-top:8px;">⏳ Waiting to start...</div>
</div>"""

    error = status.get("error")
    if error:
        label = _url_label(url)
        return f"""
<div style="background:#1a0808; border:1px solid {C_RED}55; border-radius:8px; padding:12px 14px;">
  <div style="font-size:13px; font-weight:600; color:{C_RED};">⚠️ {label}</div>
  <div style="color:{C_GRAY}; font-size:12px; margin-top:6px;">{error[:120]}</div>
</div>"""

    completed  = status.get("completed", {})
    active     = status.get("active", "")
    done       = status.get("done", False)
    brand_name = status.get("state", {}).get("brand_name", "") or _url_label(url)
    elapsed    = time.time() - status.get("start_time", time.time())
    iters      = status.get("state", {}).get("iterations", 0)

    rows = []
    for node in NODE_ORDER:
        label = NODE_LABELS[node]
        if node in completed:
            dur = f"{completed[node]:.1f}s"
            rows.append(
                f'<div style="color:{C_GREEN}; font-size:12px; padding:2px 0;">'
                f'✅ <b>{label}</b> <span style="color:{C_GRAY}; font-size:11px;">{dur}</span></div>'
            )
        elif node == active:
            rows.append(
                f'<div style="color:{C_PURPLE}; font-size:12px; padding:2px 0;">🔄 <b>{label}...</b></div>'
            )
        else:
            rows.append(
                f'<div style="color:{C_GRAY}; font-size:12px; padding:2px 0;">⏳ {label}</div>'
            )

    border_color = C_GREEN if done else C_PURPLE
    status_text  = f"✅ Done · {elapsed:.0f}s" if done else f"🔄 Running · {elapsed:.0f}s"
    iter_badge   = ""
    if iters > 1:
        iter_badge = f'<div style="margin-top:6px; padding:3px 8px; background:{C_ORANGE}22; border-left:2px solid {C_ORANGE}; color:{C_ORANGE}; font-size:11px; border-radius:0 3px 3px 0;">🔁 Iter {iters}/3</div>'

    return f"""
<div style="background:{C_BG}; border:1px solid {border_color}55; border-radius:8px; padding:12px 14px;">
  <div style="font-size:13px; font-weight:700; color:#e2e8f0; margin-bottom:4px;">{brand_name}</div>
  <div style="color:{C_GRAY}; font-size:11px; margin-bottom:8px;">{status_text}</div>
  {"".join(rows)}
  {iter_badge}
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
# AGENT RUNNER (Tab 1) — streams LangGraph, updates UI containers in real time
# ─────────────────────────────────────────────
def run_agent(brand_url: str, containers: dict, sidebar_stats) -> dict | None:
    """
    Run the full research graph for one brand, updating live UI containers at each node.

    containers keys: header, timeline, activity, brand_card
    sidebar_stats   : st.empty() in sidebar for live stats
    """
    graph   = build_graph()
    initial = make_initial_state(brand_url.strip())

    start_time    = time.time()
    last_node_end = start_time
    completed     = {}
    active_node   = NODE_ORDER[0]
    activity_log  = []
    seen_msgs     = set()
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

    containers["header"].markdown(_header_html("Starting...", 0, 0), unsafe_allow_html=True)
    containers["timeline"].markdown(_timeline_html({}, active_node, 0), unsafe_allow_html=True)

    result = None
    try:
        for step in graph.stream(initial):
            node_name = list(step.keys())[0]
            state     = list(step.values())[0]
            now       = time.time()

            completed[node_name] = now - last_node_end
            last_node_end = now

            try:
                idx = NODE_ORDER.index(node_name)
                active_node = NODE_ORDER[idx + 1] if idx + 1 < len(NODE_ORDER) else ""
            except ValueError:
                active_node = ""

            if node_name in ("identify_brand", "check_sufficiency", "generate_report"):
                stats["llm_calls"] += 1

            if node_name == "identify_brand" and state.get("brand_name"):
                _render_brand_card(containers["brand_card"], state)

            _refresh(state, active_node)
            result = state

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
# PDF EXPORT
# ─────────────────────────────────────────────
def _generate_pdf(result: dict, voice_summary: str = "", trend_delta: str = "") -> bytes:
    """
    Build a professional PDF from a completed research result dict.

    Sections:
      - Cover: brand name, website, industry, generated date
      - Voice Summary (if present)
      - Full 8-section report (markdown stripped to plain text)
      - Footer with branding

    Returns raw PDF bytes ready for st.download_button.
    """
    from fpdf import FPDF
    from datetime import datetime
    import re

    def _strip_md(text: str) -> str:
        """Remove markdown and sanitize to Latin-1 safe chars for fpdf Helvetica."""
        text = re.sub(r"#{1,6}\s*", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"---+", "", text)
        text = re.sub(r"^\s*[-•*]\s*", "- ", text, flags=re.MULTILINE)
        # Replace common Unicode chars that break Latin-1 / Helvetica
        replacements = {
            "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
            "\u2014": "-", "\u2013": "-", "\u2022": "-", "\u2026": "...",
            "\u00e2\u0080\u0099": "'", "\u00a0": " ",
        }
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        # Final safety net — drop anything still outside Latin-1
        text = text.encode("latin-1", errors="replace").decode("latin-1")
        return text.strip()

    brand   = result.get("brand_name", "Brand")
    website = result.get("brand_website", result.get("brand_url", ""))
    industry = result.get("brand_industry", "")
    report  = result.get("final_report", "")
    date    = datetime.now().strftime("%B %d, %Y")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 20, 20)

    # ── Cover page ──────────────────────────────────────
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(15, 17, 23)      # dark bg
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(20, 13)
    pdf.cell(0, 10, "AutoResearch AI  ·  Competitor Intelligence Report", ln=True)

    pdf.ln(18)

    # Brand name
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(15, 17, 23)
    pdf.cell(0, 12, brand, ln=True)

    # Meta line
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 110)
    meta_parts = [p for p in [website, industry, date] if p]
    pdf.cell(0, 8, "  ·  ".join(meta_parts), ln=True)

    pdf.ln(4)
    pdf.set_draw_color(108, 99, 255)   # purple accent
    pdf.set_line_width(1.2)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # ── Voice summary box ────────────────────────────────
    if voice_summary:
        pdf.set_fill_color(239, 246, 255)
        pdf.set_draw_color(108, 99, 255)
        pdf.set_line_width(0.4)
        x, y = pdf.get_x(), pdf.get_y()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 6, "AGENT VOICE SUMMARY", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 40)
        pdf.multi_cell(0, 6, _strip_md(voice_summary), border=1, fill=True)
        pdf.ln(6)

    # ── Trend delta box (only on repeat runs) ────────────
    if trend_delta:
        pdf.set_fill_color(236, 253, 245)       # light green tint
        pdf.set_draw_color(34, 197, 94)          # green border
        pdf.set_line_width(0.4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(34, 197, 94)
        pdf.cell(0, 6, "WHAT CHANGED SINCE LAST RUN", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 40)
        pdf.multi_cell(0, 6, _strip_md(trend_delta), border=1, fill=True)
        pdf.ln(6)

    # ── Report sections ──────────────────────────────────
    if report:
        sections = re.split(r"\n(?=#{1,3}\s)", report)
        for section in sections:
            lines = section.strip().splitlines()
            if not lines:
                continue

            first = lines[0].strip()
            is_heading = first.startswith("#")

            if is_heading:
                heading_text = re.sub(r"^#{1,3}\s*", "", first).upper()
                body_lines   = lines[1:]

                # Section heading
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(255, 255, 255)
                pdf.set_fill_color(15, 17, 23)
                pdf.cell(0, 8, f"  {heading_text}", ln=True, fill=True)
                pdf.ln(2)

                body = _strip_md("\n".join(body_lines)).strip()
                if body:
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(30, 30, 40)
                    for line in body.splitlines():
                        line = line.strip()
                        if not line:
                            pdf.ln(2)
                            continue
                        if line.startswith("•"):
                            pdf.set_x(24)
                            pdf.multi_cell(166, 5.5, line)
                        else:
                            pdf.multi_cell(0, 5.5, line)
                pdf.ln(5)
            else:
                # Plain body block (no heading)
                body = _strip_md(section).strip()
                if body:
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(30, 30, 40)
                    pdf.multi_cell(0, 5.5, body)
                    pdf.ln(3)

    # ── Footer on last page ──────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 160)
    pdf.cell(0, 6,
        f"Generated by AutoResearch AI  ·  {date}  ·  LangGraph + Groq + Tavily  ·  Data from public web",
        align="C",
    )

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# REPORT RENDERER (Tab 1)
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

    # ── Trend delta panel (only shown on second+ run) ──
    trend = result.get("trend_delta", "")
    if trend:
        with st.expander("📈 What Changed Since Last Run", expanded=True):
            st.markdown(trend)
        st.divider()
    else:
        st.caption("💡 **Trend Tracker:** Run this brand again later to see what changed in the competitive landscape.")

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


# ─────────────────────────────────────────────
# PARALLEL RESEARCH HELPERS (Tab 2)
# ─────────────────────────────────────────────

def _research_brand_silent(url: str, idx: int, brand_status: dict) -> None:
    """
    Run the full LangGraph research pipeline for one brand in a background thread.

    Writes progress into brand_status[idx] at every node completion so the
    main thread can poll and update the UI independently. Never touches
    Streamlit — all st.* calls must happen on the main thread only.
    """
    try:
        graph   = build_graph()
        initial = make_initial_state(url.strip())
        result  = None

        for step in graph.stream(initial):
            node_name = list(step.keys())[0]
            state     = list(step.values())[0]
            now       = time.time()

            brand_status[idx]["completed"][node_name] = now - brand_status[idx]["last_node_end"]
            brand_status[idx]["last_node_end"] = now

            try:
                node_idx = NODE_ORDER.index(node_name)
                brand_status[idx]["active"] = (
                    NODE_ORDER[node_idx + 1] if node_idx + 1 < len(NODE_ORDER) else ""
                )
            except ValueError:
                brand_status[idx]["active"] = ""

            brand_status[idx]["state"] = state
            result = state

        brand_status[idx]["result"] = result
        brand_status[idx]["done"]   = True

    except Exception as e:
        brand_status[idx]["error"]  = str(e)
        brand_status[idx]["done"]   = True
        brand_status[idx]["result"] = None


def _generate_cross_brand_summary(results: list) -> str:
    """
    Single LLM call that reads all completed reports and produces cross-brand insights.

    This is the unique feature of the Compare tab — no single-brand tool shows
    common competitor overlap, shared vulnerabilities, or a head-to-head verdict.
    """
    names   = [r.get("brand_name", "Brand") for r in results]
    context = ""
    for r in results:
        context += f"\n\n### {r.get('brand_name')} — {r.get('brand_industry', '')}\n"
        context += r.get("final_report", "")[:2500]

    prompt = f"""You are a cross-brand marketing analyst. You have read full competitor intelligence reports for: {', '.join(names)}.

Reports (truncated):
{context[:7500]}

Write a concise Cross-Brand Intelligence Summary with exactly these four sections:

**Common competitor themes:** Which competitors appear across multiple brands? What does that reveal about the market structure?

**Shared vulnerabilities:** What weaknesses or threats do all these brands have in common?

**The biggest market gap:** One major opportunity that none of them are fully capturing right now.

**Head-to-head verdict:** Which brand has the strongest competitive position and why? Be direct and specific.

3-5 sentences per section. Reference actual brand names. No filler."""

    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [langfuse_handler], "run_name": "cross_brand_summary"},
        )
        return response.content
    except Exception as e:
        return f"*Could not generate cross-brand summary: {e}*"


# ─────────────────────────────────────────────
# CREATIVE AGENT RUNNER (Tab 3)
# ─────────────────────────────────────────────
def run_creative_agent(url: str, industry: str, containers: dict, sidebar_stats) -> dict | None:
    """
    Stream the creative analysis graph and update UI containers at each node completion.

    Reuses the same _header_html / _timeline_html / _activity_html renderers as Tab 1
    but driven by CREATIVE_NODE_ORDER instead of NODE_ORDER.

    containers keys: header, timeline, activity
    """
    graph   = build_creative_graph()
    initial = make_creative_state(url.strip(), industry)

    start_time    = time.time()
    last_node_end = start_time
    completed     = {}
    active_node   = CREATIVE_NODE_ORDER[0]
    activity_log  = []
    seen_msgs     = set()
    stats = {"urls_searched": 0, "pages_scraped": 1, "iterations": 1,
             "llm_calls": 0, "elapsed": 0}

    def _refresh(state: dict, active: str):
        elapsed = time.time() - start_time
        pct     = min(int(len(completed) / len(CREATIVE_NODE_ORDER) * 100), 95)

        for msg in state.get("status_log", []):
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                activity_log.append((datetime.now().strftime("%H:%M:%S"), msg))

        stats["elapsed"]   = elapsed
        stats["llm_calls"] = sum(1 for n in completed if n in ("analyse_creative", "score_creative"))

        label = CREATIVE_NODE_LABELS.get(active, active) if active else "Finishing..."
        # Render timeline using a CREATIVE_NODE_ORDER-aware version
        containers["header"].markdown(
            _header_html(label, pct, elapsed), unsafe_allow_html=True
        )
        containers["timeline"].markdown(
            _creative_timeline_html(completed, active), unsafe_allow_html=True
        )
        if activity_log:
            containers["activity"].markdown(
                _activity_html(activity_log), unsafe_allow_html=True
            )
        sidebar_stats.markdown(_stats_html(stats), unsafe_allow_html=True)

    containers["header"].markdown(_header_html("Starting...", 0, 0), unsafe_allow_html=True)
    containers["timeline"].markdown(
        _creative_timeline_html({}, active_node), unsafe_allow_html=True
    )

    result = None
    try:
        for step in graph.stream(initial):
            node_name = list(step.keys())[0]
            state     = list(step.values())[0]
            now       = time.time()

            completed[node_name] = now - last_node_end
            last_node_end = now

            try:
                idx = CREATIVE_NODE_ORDER.index(node_name)
                active_node = CREATIVE_NODE_ORDER[idx + 1] if idx + 1 < len(CREATIVE_NODE_ORDER) else ""
            except ValueError:
                active_node = ""

            _refresh(state, active_node)
            result = state

        total          = time.time() - start_time
        stats["elapsed"] = total
        containers["header"].markdown(
            _completion_html(url, total, {"search_results": [], "scraped_content": [{"url": url}],
                                          "iterations": 1}),
            unsafe_allow_html=True,
        )
        containers["timeline"].markdown(
            _creative_timeline_html(completed, ""), unsafe_allow_html=True
        )
        sidebar_stats.markdown(_stats_html(stats), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error analysing '{url}': {e}")
        st.exception(e)

    return result


def _creative_timeline_html(completed: dict, active: str) -> str:
    """
    Render the 4-node creative analysis step tracker.

    Same visual design as _timeline_html but driven by CREATIVE_NODE_ORDER.
    """
    rows = []
    for node in CREATIVE_NODE_ORDER:
        label = CREATIVE_NODE_LABELS[node]
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

    return f"""
<div style="background:{C_BG}; border:1px solid {C_BORDER}; border-radius:8px; padding:12px 16px; margin:4px 0;">
  <table style="width:100%; border-collapse:collapse;">
    {"".join(rows)}
  </table>
</div>"""


def display_creative_report(result: dict):
    """Render the full creative analysis: verdict + scorecard + raw extractions + 7-section report."""
    if not result or not result.get("creative_report"):
        st.error("No creative analysis generated.")
        return

    # ── Verdict (one sentence above scorecard) ──
    verdict = result.get("creative_verdict", "")
    if verdict:
        st.markdown(_verdict_html(verdict), unsafe_allow_html=True)

    # ── Data confidence indicator ──
    st.markdown(_confidence_html(result), unsafe_allow_html=True)

    # ── Scorecard with per-dimension reasons ──
    scores = result.get("creative_scores", {})
    if scores:
        st.markdown(_scorecard_html(scores), unsafe_allow_html=True)
    else:
        st.warning("Scores not available.")

    st.divider()

    # ── Raw extractions in sidebar-style expander ──
    with st.expander("📦 Raw Extracted Elements", expanded=False):
        col_h, col_c, col_p = st.columns(3)
        with col_h:
            st.markdown("**Marketing Headlines**")
            for h in result.get("headlines", []):
                st.markdown(f"- {h}")
            plan_names = result.get("plan_names", [])
            if plan_names:
                st.markdown("**Pricing Plan Names** *(filtered from headlines)*")
                for p in plan_names:
                    st.markdown(f"- {p}")
        with col_c:
            st.markdown("**CTAs**")
            for c in result.get("ctas", []):
                st.markdown(f"- {c}")
        with col_p:
            st.markdown("**Price Mentions**")
            prices = result.get("price_mentions", [])
            if prices:
                for p in prices:
                    st.markdown(f"- {p}")
            else:
                st.caption("None found")
            st.markdown(f"**Word count:** {result.get('word_count', 0):,}")

    st.divider()

    # ── 7-section report ──
    CREATIVE_SECTIONS = [
        ("1.", "🧠 Creative Strategy Overview"),
        ("2.", "📰 Headline Analysis"),
        ("3.", "🎯 CTA Analysis"),
        ("4.", "🗣️ Tone & Messaging"),
        ("5.", "💰 Pricing Strategy"),
        ("6.", "⚠️ Weaknesses & Gaps"),
        ("7.", "🏆 How to Beat Them"),
    ]

    report = result["creative_report"]
    parts  = re.split(r'\n##\s+', "\n" + report)
    section_map: dict = {}
    for part in parts[1:]:
        for num, _ in CREATIVE_SECTIONS:
            if part.strip().startswith(num):
                section_map[num] = part.strip()
                break

    for i, (num, label) in enumerate(CREATIVE_SECTIONS):
        content = section_map.get(num, "")
        with st.expander(label, expanded=(i == 0)):
            if content:
                st.markdown("\n".join(content.split("\n")[1:]).strip())
            else:
                st.markdown("*Section not found in report.*")


# ─────────────────────────────────────────────
# VOICE RESEARCH HELPER
# ─────────────────────────────────────────────

def _run_voice_research(url: str, sidebar_stats) -> None:
    """
    Run the full research graph from a voice (or text-fallback) URL, then:
    1. Generate a spoken summary via voice_summary_node
    2. Synthesise the summary to WAV bytes via Sarvam Bulbul TTS
    3. Autoplay in the UI
    4. Render the full report below

    Reuses run_agent() and display_report() so voice mode is just a thin wrapper —
    no duplicate research logic.
    """
    raw = url.strip()
    if not raw:
        st.warning("No URL detected in transcript.")
        return
    if " " in raw or ("." not in raw and not raw.startswith(("http://", "https://"))):
        st.error(f"Could not extract a valid URL from transcript: '{raw}'. Try the manual input below.")
        return

    v_header   = st.empty()
    v_timeline = st.empty()
    v_activity = st.empty()
    v_card     = st.empty()

    result = run_agent(
        raw,
        containers={
            "header":     v_header,
            "timeline":   v_timeline,
            "activity":   v_activity,
            "brand_card": v_card,
        },
        sidebar_stats=sidebar_stats,
    )
    v_card.empty()

    if not result:
        st.error("Research failed — no result returned.")
        return

    # ── Generate spoken summary ──
    with st.spinner("Generating spoken summary..."):
        voice_state = voice_summary_node(result)
        summary_text = voice_state.get("voice_summary", "")

    if summary_text:
        st.markdown("### What the agent found")
        st.info(summary_text)

        # ── Synthesise and autoplay via Sarvam Bulbul ──
        try:
            audio_bytes = speak_sarvam(summary_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav", autoplay=True)
        except Exception as e:
            st.warning(f"Text-to-speech failed: {e}")

    # ── Speak trend delta if this is a repeat run ──
    trend_delta = result.get("trend_delta", "")
    if trend_delta:
        # Condense trend delta to a spoken sentence (already plain text from LLM)
        import re as _re
        trend_spoken = _re.sub(r"\*\*(.+?)\*\*", r"\1", trend_delta)  # strip bold
        trend_spoken = " ".join(trend_spoken.split())[:400]            # cap at 400 chars
        trend_spoken = "Here is what changed since your last research. " + trend_spoken
        try:
            trend_audio = speak_sarvam(trend_spoken)
            if trend_audio:
                st.markdown("### What changed since last time")
                st.audio(trend_audio, format="audio/wav", autoplay=False)
        except Exception:
            pass  # silent — trend panel still shows in display_report below

    st.divider()
    display_report(result, result.get("brand_name", raw))

    # ── PDF export ──
    try:
        pdf_bytes = _generate_pdf(result, voice_summary=summary_text, trend_delta=trend_delta)
        brand_slug = result.get("brand_name", "report").lower().replace(" ", "_")
        st.download_button(
            label="⬇️ Download Report as PDF",
            data=pdf_bytes,
            file_name=f"{brand_slug}_competitor_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.caption(f"PDF export unavailable: {e}")


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
            placeholder="https://...",
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
            st.error("Please enter a URL (e.g. https://brandname.com), not just a brand name.")
        else:
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

            # Clear streaming brand card — display_report renders it permanently below
            brand_card.empty()

            if result:
                display_report(result, result.get("brand_name", raw))


# ═══════════════════════════════════════════════
# TAB 2 — COMPARE BRANDS (PARALLEL)
# ═══════════════════════════════════════════════
with tab2:
    st.header("Compare Up to 3 Brands Side by Side")
    st.caption("All brands are researched simultaneously — total time equals the slowest single brand, not the sum.")

    c1, c2, c3 = st.columns(3)
    with c1: url_a = st.text_input("Brand 1 URL", placeholder="https://...", key="brand_a")
    with c2: url_b = st.text_input("Brand 2 URL", placeholder="https://...", key="brand_b")
    with c3: url_c = st.text_input("Brand 3 URL (optional)", placeholder="https://... (optional)", key="brand_c")

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        compare_btn = st.button("Compare Brands", type="primary", key="compare", use_container_width=True)
    with btn_col2:
        reset_btn = st.button("Reset", key="reset_compare", use_container_width=True)

    if reset_btn:
        st.rerun()

    if compare_btn:
        urls    = [u.strip() for u in [url_a, url_b, url_c] if u.strip()]
        invalid = [u for u in urls if " " in u or ("." not in u and not u.startswith(("http://", "https://")))]

        if invalid:
            st.error(f"Please enter URLs, not brand names: {', '.join(invalid)}")
        elif len(urls) < 2:
            st.warning("Please enter at least 2 brand URLs to compare.")
        else:
            n = len(urls)

            # Initialise shared status dicts BEFORE starting threads
            # so the polling loop can safely read from them immediately.
            brand_status: dict = {}
            for i in range(n):
                brand_status[i] = {
                    "completed":    {},
                    "active":       NODE_ORDER[0],
                    "state":        {},
                    "done":         False,
                    "error":        None,
                    "result":       None,
                    "start_time":   time.time(),
                    "last_node_end": time.time(),
                }

            compare_start = time.time()

            # Overall progress banner + per-brand column placeholders
            overall_box   = st.empty()
            st.divider()
            progress_cols = st.columns(n)
            col_containers = [col.empty() for col in progress_cols]

            # ── Launch all threads simultaneously ──
            threads = []
            for i, url in enumerate(urls):
                t = threading.Thread(
                    target=_research_brand_silent,
                    args=(url, i, brand_status),
                    daemon=True,
                )
                threads.append(t)

            for t in threads:
                t.start()

            # ── Polling loop — main thread updates UI every 0.5s ──
            # sleep(0.5) is intentional here: polling interval between UI refreshes.
            while not all(brand_status[i]["done"] for i in range(n)):
                elapsed    = time.time() - compare_start
                done_count = sum(1 for i in range(n) if brand_status[i]["done"])
                overall_box.markdown(
                    _overall_progress_html(n, done_count, elapsed),
                    unsafe_allow_html=True,
                )
                for i in range(n):
                    col_containers[i].markdown(
                        _compare_col_html(brand_status[i], urls[i]),
                        unsafe_allow_html=True,
                    )
                time.sleep(0.5)

            # Ensure all threads have fully exited
            for t in threads:
                t.join()

            # ── Final UI refresh after all done ──
            total = time.time() - compare_start
            overall_box.markdown(_all_done_html(n, total), unsafe_allow_html=True)
            for i in range(n):
                col_containers[i].markdown(
                    _compare_col_html(brand_status[i], urls[i]),
                    unsafe_allow_html=True,
                )

            # ── Update sidebar with combined totals ──
            all_results = [brand_status[i]["result"] for i in range(n) if brand_status[i]["result"]]
            sidebar_stats_box.markdown(_stats_html({
                "urls_searched": sum(len(r.get("search_results", [])) for r in all_results),
                "pages_scraped": sum(len(r.get("scraped_content", [])) for r in all_results),
                "iterations":    sum(r.get("iterations", 0) for r in all_results),
                "llm_calls":     n * 3,
                "elapsed":       total,
            }), unsafe_allow_html=True)

            st.divider()

            # ── Error cards for failed brands ──
            failed     = [(i, brand_status[i]) for i in range(n) if brand_status[i]["error"]]
            successful = [(i, brand_status[i]) for i in range(n)
                          if not brand_status[i]["error"] and brand_status[i]["result"]]

            for i, status in failed:
                st.error(f"⚠️ Could not research `{urls[i]}` — {status['error']}")

            if not successful:
                st.error("All brands failed to research. Check that the URLs are reachable.")
            else:
                # ── SECTION 1: Brand cards row ──
                st.subheader("Brand Overview")
                card_cols = st.columns(len(successful))
                for col, (i, status) in zip(card_cols, successful):
                    r = status["result"]
                    with col:
                        logo = r.get("brand_logo", "")
                        if logo:
                            try:
                                st.image(logo, width=48)
                            except Exception:
                                pass
                        name     = r.get("brand_name", _url_label(urls[i]))
                        industry = r.get("brand_industry", "")
                        sources  = len(r.get("search_results", []))
                        scraped  = len(r.get("scraped_content", []))
                        iters    = r.get("iterations", 0)
                        live_tag = "✅ Live data" if r.get("has_live_data") else "⚠️ AI estimate"
                        st.markdown(f"### {name}")
                        if industry:
                            st.caption(industry)
                        st.markdown(f"{sources} sources · {scraped} pages · {iters} iter(s)  \n{live_tag}")

                # ── SECTION 2: Quick stats comparison table ──
                st.divider()
                st.subheader("Quick Stats Comparison")
                col_names = [brand_status[i]["result"].get("brand_name", _url_label(urls[i])) for i, _ in successful]
                tbl  = "| Metric | " + " | ".join(col_names) + " |\n"
                tbl += "|" + "|".join(["---"] * (len(col_names) + 1)) + "|\n"
                metric_rows = [
                    ("Sources found",       lambda r: str(len(r.get("search_results", [])))),
                    ("Pages scraped",       lambda r: str(len(r.get("scraped_content", [])))),
                    ("Research iterations", lambda r: str(r.get("iterations", 0))),
                    ("Reddit threads",      lambda r: str(len(r.get("reddit_insights", [])))),
                    ("News articles",       lambda r: str(len(r.get("news_insights", [])))),
                    ("Has live data",       lambda r: "✅" if r.get("has_live_data") else "⚠️"),
                ]
                for label, fn in metric_rows:
                    tbl += f"| {label} | " + " | ".join(fn(brand_status[i]["result"]) for i, _ in successful) + " |\n"
                st.markdown(tbl)

                # ── SECTION 3: Full reports side by side ──
                st.divider()
                st.subheader("Full Reports Side by Side")
                SECTIONS_COMPARE = [
                    ("1.", "🏆 Top Competitors"),
                    ("2.", "🔍 Competitor Analysis"),
                    ("3.", "📊 Scoring"),
                    ("4.", "🎯 SWOT"),
                    ("5.", "💬 Reddit Sentiment"),
                    ("6.", "🚀 Positioning Gaps"),
                    ("7.", "💡 Key Insights"),
                    ("8.", "✅ Recommended Actions"),
                ]
                report_cols = st.columns(len(successful))
                for col, (i, status) in zip(report_cols, successful):
                    r      = status["result"]
                    brand  = r.get("brand_name", _url_label(urls[i]))
                    report = r.get("final_report", "")
                    parts  = re.split(r'\n##\s+', "\n" + report)
                    section_map: dict = {}
                    for part in parts[1:]:
                        for num, _ in SECTIONS_COMPARE:
                            if part.strip().startswith(num):
                                section_map[num] = part.strip()
                                break
                    with col:
                        st.markdown(f"#### {brand}")
                        for j, (num, lbl) in enumerate(SECTIONS_COMPARE):
                            content = section_map.get(num, "")
                            with st.expander(lbl, expanded=(j == 0)):
                                if content:
                                    st.markdown("\n".join(content.split("\n")[1:]).strip())
                                else:
                                    st.markdown("*Section not found.*")

                # ── SECTION 4: Cross-brand intelligence summary ──
                if len(successful) >= 2:
                    st.divider()
                    st.subheader("Cross-Brand Intelligence Summary")
                    st.caption(
                        "One LLM call synthesising all reports — common rivals, shared gaps, "
                        "and a head-to-head verdict no single-brand tool provides."
                    )
                    with st.spinner("Generating cross-brand analysis..."):
                        cross_results = [brand_status[i]["result"] for i, _ in successful]
                        summary = _generate_cross_brand_summary(cross_results)
                    st.markdown(summary)


# ═══════════════════════════════════════════════
# TAB 3 — CREATIVE DECODER
# ═══════════════════════════════════════════════
with tab3:
    st.header("Creative Decoder")
    st.caption(
        "Paste any competitor URL — agent decodes their full creative strategy "
        "and tells you exactly how to beat them."
    )

    cr_col1, cr_col2, cr_col3 = st.columns([4, 2, 1])
    with cr_col1:
        creative_url = st.text_input(
            "Competitor Landing Page URL",
            placeholder="https://...",
            key="creative_url",
        )
    with cr_col2:
        creative_industry = st.selectbox(
            "Your Industry",
            options=INDUSTRIES,
            key="creative_industry",
        )
    with cr_col3:
        st.write("")
        st.write("")
        creative_btn = st.button("Analyse", type="primary", use_container_width=True, key="run_creative")

    if creative_btn:
        raw_url = creative_url.strip()
        if not raw_url:
            st.warning("Please enter a URL.")
        elif " " in raw_url or ("." not in raw_url and not raw_url.startswith(("http://", "https://"))):
            st.error("Please enter a valid URL, not just a brand name.")
        else:
            cr_header   = st.empty()
            cr_timeline = st.empty()
            cr_activity = st.empty()

            creative_result = run_creative_agent(
                raw_url,
                creative_industry,
                containers={
                    "header":   cr_header,
                    "timeline": cr_timeline,
                    "activity": cr_activity,
                },
                sidebar_stats=sidebar_stats_box,
            )

            if creative_result:
                st.divider()
                # Show URL + industry badge above report
                st.markdown(
                    f"**Analysed:** `{raw_url}`  &nbsp; | &nbsp;  "
                    f"**Industry context:** {creative_industry}"
                )
                st.divider()
                display_creative_report(creative_result)


# ═══════════════════════════════════════════════
# TAB 4 — VOICE RESEARCH
# ═══════════════════════════════════════════════
with tab4:
    st.header("Voice Research")
    st.caption(
        "Speak a brand name — the agent researches it and reads the key findings aloud. "
        "Full report renders below."
    )

    if not _VOICE_AVAILABLE:
        st.warning("Voice module failed to import — check voice.py.")
    elif not os.environ.get("SARVAM_API_KEY"):
        st.warning("SARVAM_API_KEY is not set. Add it to your .env or Streamlit secrets.")
    else:
        v_col1, v_col2 = st.columns([4, 1])
        with v_col1:
            audio_input = st.audio_input("Speak a brand name or URL")
        with v_col2:
            st.write("")
            voice_btn = st.button(
                "Transcribe & Research",
                type="primary",
                use_container_width=True,
                key="run_voice",
                disabled=(audio_input is None),
            )

        # ── Process: mic path ──
        if voice_btn and audio_input is not None:
            with st.spinner("Transcribing audio..."):
                try:
                    transcript = transcribe_sarvam(audio_input.getvalue())
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    transcript = ""

            if transcript:
                st.info(f"Heard: **{transcript}**")

                # Extract URL — prefer a bare domain/URL in the transcript,
                # fall back to treating the whole transcript as a search hint
                words     = transcript.split()
                url_token = next(
                    (w for w in words if "." in w and " " not in w),
                    transcript.strip(),
                )
                if not url_token.startswith(("http://", "https://")):
                    url_token = "https://" + url_token.lstrip("/")

                # ── Acknowledgment — agent confirms it heard the request ──
                brand_hint = url_token.replace("https://", "").replace("http://", "").split("/")[0]
                try:
                    ack_bytes = speak_sarvam(
                        f"Sure! I'll go and research {brand_hint} for you right now. Give me a moment."
                    )
                    if ack_bytes:
                        st.audio(ack_bytes, format="audio/wav", autoplay=True)
                except Exception:
                    pass  # ack failure is silent

                _run_voice_research(url_token, sidebar_stats_box)
            else:
                st.warning("No speech detected. Try speaking louder or use the manual URL input below.")



# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("AutoResearch AI · Built by Bharadwaj R · LangGraph + LangSmith + Langfuse + Groq + Tavily + Chroma · All data from public web")
