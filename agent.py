"""
agent.py — LangGraph state machine for AutoResearch AI.

Graph topology:
    START → identify_brand → search → scrape → check_sufficiency
                                                      │
                              ┌──── loop (max 3) ─────┘
                              ▼
                       generate_report → store_memory → END

Node responsibilities:
    identify_brand    : Verify the brand's official website; extract industry,
                        description, and logo. Never guesses — scrapes real HTML.
    search            : Parallel Tavily queries (competitors + Reddit + news + pricing).
    scrape            : BeautifulSoup scrape of every competitor URL found.
    check_sufficiency : LLM self-evaluation — loop back if data is thin (max 3×).
    generate_report   : Produce the 8-section markdown intelligence report.
    store_memory      : Persist the report to Chroma for future recall.

Anti-hallucination design:
    brand_context and brand_industry are extracted from the real website in
    identify_brand and injected into every subsequent LLM prompt. The report
    prompt explicitly forbids listing competitors outside that industry.

Exports:
    build_graph()           : Compile and return the LangGraph app (now 7 nodes incl. trend_compare).
    make_initial_state(...) : Construct a zero-value ResearchState dict.
    ResearchState           : TypedDict for the shared state.
    voice_summary_node(...) : Standalone — condense report for TTS (not in graph).
    trend_compare_node(...) : Node 6 — diff current vs previous Chroma report.
"""

import json
import re
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from chains import llm, langfuse_handler, get_langfuse_prompt
from tools import scrape_website, search_reddit, search_news, search_pricing, _tavily_search, extract_brand_identity, scrape_creative_page
from memory import save_research, save_creative, get_previous_research

load_dotenv()


# ─────────────────────────────────────────────
# JSON EXTRACTION HELPER
# ─────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM response text."""
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 2. Find first {...} block (handles LLM wrapping in markdown)
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class ResearchState(TypedDict):
    brand_name: str         # Extracted from URL (or provided)
    brand_url: str          # URL provided by user — primary input
    brand_logo: str         # og:image or logo img src
    brand_context: str      # Full description from their website
    brand_website: str      # Verified official URL
    brand_industry: str     # e.g. "AI Marketing Technology"
    search_results: list
    scraped_content: list
    reddit_insights: list
    news_insights: list
    pricing_data: list
    iterations: int
    is_sufficient: bool
    final_report: str
    status_log: list
    current_node: str
    has_live_data: bool
    voice_summary: str      # Spoken-style condensed report (≤100 words, no markdown)
    is_voice_mode: bool     # True when request came from the Voice tab
    trend_delta: str        # What changed vs last Chroma run for this brand (empty on first run)


# ─────────────────────────────────────────────
# NODE 0: identify_brand
# Smart discovery — uses Tavily to find the brand, verifies via LLM,
# never guesses a URL from the brand name alone.
# ─────────────────────────────────────────────

# Sites to skip when looking for official brand homepage
_SKIP_DOMAINS = [
    "wikipedia.org", "linkedin.com", "crunchbase.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "youtube.com", "glassdoor.com",
    "indeed.com", "g2.com", "capterra.com", "trustpilot.com", "reddit.com",
    "quora.com", "medium.com", "techcrunch.com", "forbes.com", "bloomberg.com",
    "tracxn.com", "pitchbook.com", "angellist.com",
]

# TLD priority — lower index = higher priority
_TLD_PRIORITY = [".com", ".io", ".ai", ".co", ".in", ".app", ".net", ".org"]


def _score_candidate(hit: dict, brand_lower: str) -> int:
    """Score a Tavily result for likelihood of being the official brand site."""
    url   = hit.get("url", "").lower()
    title = hit.get("title", "").lower()

    # Hard skip for aggregator/social sites
    if any(skip in url for skip in _SKIP_DOMAINS):
        return -1

    score = 0

    # Brand name appears in domain
    if brand_lower in url:
        score += 15

    # Brand name appears in page title
    if brand_lower in title:
        score += 8

    # Preferred TLDs
    for priority, tld in enumerate(_TLD_PRIORITY):
        if tld in url:
            score += max(8 - priority, 1)
            break

    return score


# ─────────────────────────────────────────────
# PROMPT FALLBACKS
# These are used when the named prompt doesn't exist in Langfuse yet.
# Create them in Langfuse UI to override dynamically (use {{variable}} syntax).
# ─────────────────────────────────────────────
_VERIFY_BRAND_FALLBACK = """\
You are verifying whether a webpage is the official website of a brand.

Brand name: "{{brand}}"
URL: {{url}}
Page content: {{page_content}}

Answer in valid JSON only — no other text, no markdown fences:
{
  "is_correct_site": true or false,
  "company_description": "2-3 sentences describing what this company does and who they serve",
  "industry": "specific industry category e.g. AI Marketing Technology, Food Delivery, Sportswear",
  "location": "city, country — primary HQ location",
  "founded": "year founded or unknown",
  "official_url": "{{url}}"
}

Set is_correct_site to true ONLY if this page clearly belongs to "{{brand}}" the company/product.\
"""

_CHECK_SUFFICIENCY_FALLBACK = """\
You are a marketing analyst. We are researching competitors for:
{{summary}}

Do we have enough data to write a full competitor report with SWOT and positioning gaps?
Answer YES or NO on the first line, then explain briefly.\
"""

_GENERATE_REPORT_FALLBACK = """\
You are an expert marketing strategist.

BRAND: {{brand_name}}
OFFICIAL WEBSITE: {{brand_website}}
INDUSTRY: {{brand_industry}}
WHAT THEY DO: {{brand_context}}

CRITICAL RULE: {{data_note}}
Only list competitors that operate in the SAME industry: "{{brand_industry}}"
Never suggest companies from unrelated industries.

LIVE RESEARCH DATA:
{{context}}

REDDIT SENTIMENT:
{{reddit_text}}

RECENT NEWS:
{{news_text}}

PRICING INTELLIGENCE:
{{pricing_text}}

Write the full report using EXACTLY these 8 section headers (## prefix):

## 1. TOP COMPETITORS IDENTIFIED
4-6 real competitors in the same space. Name, one-line description, website.

## 2. COMPETITOR ANALYSIS
For each competitor — Positioning | Strengths | Weaknesses | Target Audience.

## 3. COMPETITOR SCORING (1-10 table)
| Competitor | Features | Pricing | Market Presence | Growth | Overall |
|---|---|---|---|---|---|

## 4. SWOT ANALYSIS for {{brand_name}}
| | Strengths | Weaknesses |
|---|---|---|
| Internal | ... | ... |

| | Opportunities | Threats |
|---|---|---|
| External | ... | ... |

## 5. REDDIT SENTIMENT SUMMARY
What real users say. Quote specific opinions if available from the Reddit data.

## 6. MARKET POSITIONING GAPS
Specific opportunities competitors are missing that {{brand_name}} can capture.

## 7. KEY INSIGHTS
7 bullet point findings.

## 8. RECOMMENDED ACTIONS
5 specific, actionable steps for {{brand_name}} to win market share.\
"""


def _normalize_url(url: str) -> str:
    """Ensure URL has a scheme. 'notion.so' → 'https://notion.so'."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _brand_name_from_url(url: str) -> str:
    """Extract a human-readable brand name from a domain. 'https://notion.so' → 'Notion'."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or urlparse("https://" + url).netloc
        # Strip www., take first part before first dot
        domain = domain.lstrip("www.").split(".")[0]
        return domain.capitalize()
    except Exception:
        return ""


def identify_brand_node(state: ResearchState) -> ResearchState:
    """
    Verify the brand's official website and extract its identity.

    Fast path: user supplied a URL — scrape it directly and verify via LLM.
    Standard path: run 3 parallel Tavily queries, score candidates, then verify
    each one with the LLM until a match is found (max 4 attempts).

    Populates: brand_name, brand_logo, brand_context, brand_website, brand_industry.
    All downstream nodes depend on brand_industry to stay in the right market segment.
    """
    brand = state["brand_name"]
    provided_url = _normalize_url(state.get("brand_url", ""))

    # If brand name not provided, derive a preliminary one from the URL domain
    if not brand and provided_url:
        brand = _brand_name_from_url(provided_url)

    brand_lower = brand.lower().replace(" ", "")

    official_url = ""
    brand_info   = {}

    # Fields populated by extract_brand_identity
    extracted_name = ""
    extracted_logo = ""

    if provided_url:
        # ── Fast path: user provided URL — extract identity + verify ──
        log_entry = f"Identifying brand — scraping provided URL: {provided_url}"
        print(log_entry)

        # Extract name and logo directly from HTML
        identity = extract_brand_identity(provided_url)
        extracted_name = identity.get("name", "")
        extracted_logo = identity.get("logo", "")

        # Use extracted name if brand_name not set
        if not brand and extracted_name:
            brand = extracted_name
            brand_lower = brand.lower().replace(" ", "")

        page_text = scrape_website.invoke({"url": provided_url})
        if page_text and len(page_text) >= 80:
            verify_prompt = get_langfuse_prompt(
                "verify-brand",
                _VERIFY_BRAND_FALLBACK,
                brand=brand or "unknown",
                url=provided_url,
                page_content=page_text[:1800],
            )
            response = llm.invoke(
                [HumanMessage(content=verify_prompt)],
                config={"callbacks": [langfuse_handler], "run_name": "verify_brand"},
            )
            brand_info   = extract_json(response.content)
            official_url = provided_url
            print(f"[identify_brand] Direct URL verified — industry={brand_info.get('industry','?')}")
        else:
            print(f"[identify_brand] Could not scrape provided URL {provided_url} — falling back to search")

    if not brand_info:
        # ── Standard path: search for the website ──
        if not provided_url:
            log_entry = f"Identifying brand '{brand}' — searching for official website..."
        else:
            log_entry = f"Provided URL failed — falling back to search for '{brand}'..."
        print(log_entry)

        # Step 1: Tavily searches (3 queries, parallel)
        queries = [
            f"{brand} official website",
            f"{brand} company homepage",
            f"{brand} technology product startup",
        ]
        all_hits, seen_urls = [], set()
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(_tavily_search, q, 4) for q in queries]
            for f in as_completed(futures):
                for hit in f.result():
                    if hit["url"] not in seen_urls:
                        seen_urls.add(hit["url"])
                        all_hits.append(hit)

        # Step 2: Score and rank candidates
        scored = [(hit, _score_candidate(hit, brand_lower)) for hit in all_hits]
        scored = [(h, s) for h, s in scored if s >= 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [h for h, _ in scored]

        print(f"[identify_brand] {len(candidates)} candidates to verify")

        # Step 3: Verify each candidate with LLM
        for attempt, candidate in enumerate(candidates[:4]):
            url = candidate["url"]
            print(f"[identify_brand] Attempt {attempt + 1}: verifying {url}")

            page_text = scrape_website.invoke({"url": url})
            if not page_text or len(page_text) < 80:
                print(f"[identify_brand] Could not scrape {url} — skipping")
                continue

            verify_prompt = get_langfuse_prompt(
                "verify-brand",
                _VERIFY_BRAND_FALLBACK,
                brand=brand,
                url=url,
                page_content=page_text[:1800],
            )
            response = llm.invoke(
                [HumanMessage(content=verify_prompt)],
                config={"callbacks": [langfuse_handler], "run_name": "verify_brand"},
            )

            info = extract_json(response.content)
            print(f"[identify_brand] LLM verdict for {url}: is_correct={info.get('is_correct_site')}, industry={info.get('industry','?')}")

            if info.get("is_correct_site"):
                official_url = url
                brand_info   = info
                break

    # ── Build brand context from verified info ──
    if brand_info:
        brand_context = (
            f"{brand} is a {brand_info.get('industry', 'company')}. "
            f"{brand_info.get('company_description', '')}"
        )
        industry = brand_info.get("industry", "")
    else:
        # Fallback: use best Tavily snippet
        fallback = candidates[0] if candidates else {}
        brand_context = f"{brand}: {fallback.get('content', '')[:300]}" if fallback else f"Brand: {brand}"
        industry = ""
        official_url = fallback.get("url", "") if fallback else ""
        print(f"[identify_brand] Could not verify any site — using fallback context")

    log_entry2 = f"Brand identified: {industry or 'unknown industry'} — {official_url}"
    print(log_entry2)

    return {
        **state,
        "brand_name":     brand or state["brand_name"],
        "brand_logo":     extracted_logo,
        "brand_context":  brand_context,
        "brand_website":  official_url,
        "brand_industry": industry,
        "current_node":   "identify_brand",
        "status_log":     state.get("status_log", []) + [log_entry, log_entry2],
    }


# ─────────────────────────────────────────────
# NODE 1: search
# Uses brand_industry to make queries specific to the right market.
# Runs competitor search + Reddit + news + pricing ALL in parallel.
# ─────────────────────────────────────────────
def search_node(state: ResearchState) -> ResearchState:
    """
    Run all research queries in parallel and accumulate results into state.

    Queries are industry-aware when brand_industry is known — this produces
    tighter competitor results than generic brand-name searches.
    Four data streams run simultaneously: competitors, Reddit, news, pricing.
    New URLs are deduplicated against already-collected search_results.
    """
    brand    = state["brand_name"]
    industry = state.get("brand_industry", "")
    iteration = state["iterations"] + 1
    log_entry = f"[Iteration {iteration}] Searching using industry context: '{industry or brand}'..."
    print(log_entry)

    # Industry-aware competitor queries — fully dynamic, nothing hardcoded
    if industry:
        competitor_queries = [
            f"{industry} top competitors 2026",
            f"{brand} vs alternatives {industry}",
            f"best tools similar to {brand} {industry}",
        ]
    else:
        competitor_queries = [
            f"{brand} top competitors alternatives",
            f"{brand} vs similar products",
            f"best alternatives to {brand}",
        ]

    def run_competitor_search():
        results, seen = [], set()
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(_tavily_search, q, 3) for q in competitor_queries]
            for f in as_completed(futures):
                for hit in f.result():
                    if hit["url"] not in seen:
                        seen.add(hit["url"])
                        results.append(hit)
        print(f"[search] {len(results)} competitor results")
        return results

    # All 4 data streams in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_comp    = executor.submit(run_competitor_search)
        f_reddit  = executor.submit(search_reddit, brand)
        f_news    = executor.submit(search_news, brand)
        f_pricing = executor.submit(search_pricing, brand)

        new_results  = f_comp.result()
        reddit_data  = f_reddit.result()
        news_data    = f_news.result()
        pricing_data = f_pricing.result()

    existing_urls = {r["url"] for r in state.get("search_results", [])}
    unique_new = [r for r in new_results if r["url"] not in existing_urls]

    return {
        **state,
        "search_results":  state.get("search_results", []) + unique_new,
        "reddit_insights": state.get("reddit_insights", []) + reddit_data,
        "news_insights":   state.get("news_insights", []) + news_data,
        "pricing_data":    state.get("pricing_data", []) + pricing_data,
        "iterations":      iteration,
        "current_node":    "search",
        "status_log":      state.get("status_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────
# NODE 2: scrape
# ─────────────────────────────────────────────
def scrape_node(state: ResearchState) -> ResearchState:
    """
    Scrape all competitor URLs discovered in the last search pass.

    Uses the already_scraped offset so repeated loop iterations don't
    re-scrape URLs processed in earlier iterations.
    """
    already_scraped = len(state.get("scraped_content", []))
    urls = [r["url"] for r in state["search_results"][already_scraped:]]
    log_entry = f"[Iteration {state['iterations']}] Scraping {len(urls)} pages..."
    print(log_entry)

    new_scraped = []
    for url in urls:
        text = scrape_website.invoke({"url": url})
        if text:
            new_scraped.append({"url": url, "text": text})

    return {
        **state,
        "scraped_content": state.get("scraped_content", []) + new_scraped,
        "current_node":    "scrape",
        "status_log":      state.get("status_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────
# NODE 3: check_sufficiency
# ─────────────────────────────────────────────
def check_sufficiency_node(state: ResearchState) -> ResearchState:
    """
    Ask the LLM whether the collected data is sufficient for a full report.

    Passes a structured summary (counts + sample snippets) rather than raw data
    so the LLM can make a reliable YES/NO judgement without hitting token limits.
    If the answer is NO and iterations < 3, the graph loops back to search_node.
    """
    log_entry = f"[Iteration {state['iterations']}] Checking data sufficiency..."
    print(log_entry)

    if len(state.get("search_results", [])) == 0:
        warning = "WARNING: 0 search results — retrying..."
        print(warning)
        return {
            **state,
            "is_sufficient": False,
            "current_node":  "check_sufficiency",
            "status_log":    state.get("status_log", []) + [log_entry, warning],
        }

    summary = (
        f"Brand: {state['brand_name']}\n"
        f"Industry: {state.get('brand_industry', 'unknown')}\n"
        f"What they do: {state.get('brand_context', '')[:200]}\n"
        f"Competitor results: {len(state.get('search_results', []))}\n"
        f"Pages scraped: {len(state.get('scraped_content', []))}\n"
        f"Reddit threads: {len(state.get('reddit_insights', []))}\n"
        f"News articles: {len(state.get('news_insights', []))}\n"
    )
    for item in state.get("scraped_content", [])[:2]:
        summary += f"\n--- {item['url']} ---\n{item['text'][:200]}\n"

    prompt = get_langfuse_prompt(
        "check-sufficiency",
        _CHECK_SUFFICIENCY_FALLBACK,
        summary=summary,
    )
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse_handler], "run_name": "check_sufficiency"},
    )
    is_sufficient = response.content.strip().upper().startswith("YES")
    log_entry2 = f"[Iteration {state['iterations']}] {'SUFFICIENT ✓' if is_sufficient else 'Need more — looping...'}"
    print(log_entry2)

    return {
        **state,
        "is_sufficient": is_sufficient,
        "current_node":  "check_sufficiency",
        "status_log":    state.get("status_log", []) + [log_entry, log_entry2],
    }


def should_continue(state: ResearchState) -> str:
    """
    Conditional edge: proceed to report generation or loop back for more data.

    The iteration cap (3) prevents infinite loops when Tavily returns sparse
    results or the LLM keeps asking for more data despite sufficient context.
    """
    return "generate" if (state["is_sufficient"] or state["iterations"] >= 3) else "search"


# ─────────────────────────────────────────────
# NODE 4: generate_report
# brand_context + brand_industry injected into prompt.
# LLM has zero excuse to hallucinate the wrong industry.
# ─────────────────────────────────────────────
def generate_report_node(state: ResearchState) -> ResearchState:
    """
    Generate the 8-section markdown intelligence report.

    brand_context and brand_industry are injected into the prompt to prevent
    the LLM from suggesting competitors in unrelated industries.
    When no live data was retrieved, has_live_data=False triggers a warning
    label in the UI and the prompt tells the LLM to mark sections as estimates.
    """
    log_entry = "Generating marketing intelligence report..."
    print(log_entry)

    scraped  = state.get("scraped_content", [])
    searches = state.get("search_results", [])
    has_live_data = bool(scraped or searches)

    context = ""
    for item in scraped:
        context += f"\n[SCRAPED] {item['url']}\n{item['text']}\n"
    for r in searches:
        context += f"\n[WEB] {r['title']} ({r['url']})\n{r['content']}\n"

    reddit_text  = "\n".join(f"- [{r['title']}] {r['content'][:300]}" for r in state.get("reddit_insights", [])) or "No Reddit data."
    news_text    = "\n".join(f"- [{r['title']}] {r['content'][:200]}" for r in state.get("news_insights", [])) or "No news data."
    pricing_text = "\n".join(f"- [{r['title']}] {r['content'][:200]}" for r in state.get("pricing_data", [])) or "No pricing data."

    data_note = (
        "Use the LIVE research data below as your primary source. Do NOT hallucinate or invent competitors."
        if has_live_data else
        "No live data was retrieved. Use training knowledge but label every section '⚠️ AI Estimate'."
    )

    prompt = get_langfuse_prompt(
        "generate-report",
        _GENERATE_REPORT_FALLBACK,
        brand_name=state["brand_name"],
        brand_website=state.get("brand_website", "unknown"),
        brand_industry=state.get("brand_industry", "unknown"),
        brand_context=state.get("brand_context", "unknown"),
        data_note=data_note,
        context=context[:4000] if context else "None.",
        reddit_text=reddit_text[:1200],
        news_text=news_text[:800],
        pricing_text=pricing_text[:800],
    )
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse_handler], "run_name": "generate_report"},
    )

    return {
        **state,
        "final_report": response.content,
        "has_live_data": has_live_data,
        "current_node":  "generate_report",
        "status_log":    state.get("status_log", []) + [log_entry, "Report generated successfully."],
    }


# ─────────────────────────────────────────────
# NODE 5: store_memory
# ─────────────────────────────────────────────
def store_memory_node(state: ResearchState) -> ResearchState:
    """
    Persist the final report to Chroma for cross-session recall.

    Wrapped in try/except so a Chroma failure never blocks report delivery.
    """
    try:
        save_research(state["brand_name"], state["final_report"])
    except Exception as e:
        print(f"[store_memory] Warning: {e}")
    return {
        **state,
        "current_node": "store_memory",
        "status_log":   state.get("status_log", []) + ["Saved to memory."],
    }


# ─────────────────────────────────────────────
# TREND TRACKER NODE — wired into research graph
# Runs after generate_report, before store_memory.
# On first run for a brand: no-op (trend_delta stays "").
# On subsequent runs: LLM diffs current vs previous report.
# ─────────────────────────────────────────────

_TREND_COMPARE_FALLBACK = """\
You are a competitive intelligence analyst comparing two research snapshots for the same brand.

Brand: {{brand}}

PREVIOUS REPORT (last run):
{{previous}}

CURRENT REPORT (this run):
{{current}}

Write a concise "What Changed" summary with exactly these three parts:

**New threats:** Any competitors that appeared this run but not last time. If none, say "No new entrants detected."

**Disappeared or weakened:** Any competitors from last run that are no longer prominent. If none, say "No exits detected."

**Shift in market dynamics:** One sentence on the biggest strategic change between the two snapshots.

Be specific — name actual companies. 2-3 sentences per part. No filler.\
"""


def trend_compare_node(state: ResearchState) -> dict:
    """
    Compare the freshly generated report against the most recent Chroma entry
    for the same brand and surface what changed.

    No-op on first run (no previous report in memory). On subsequent runs,
    produces a plain-language diff the UI renders as a "What Changed" panel.
    Chroma is queried BEFORE store_memory runs so the result is always the
    previous run, never the current one.
    """
    log_entry = "Comparing with previous research..."
    print(log_entry)

    previous = get_previous_research(state.get("brand_name", ""))
    if not previous:
        print("[trend_compare] No previous report found — first run for this brand.")
        return {
            "trend_delta":  "",
            "current_node": "trend_compare",
            "status_log":   state.get("status_log", []) + ["No previous data — trend comparison skipped."],
        }

    prompt = get_langfuse_prompt(
        "trend-compare",
        _TREND_COMPARE_FALLBACK,
        brand=state.get("brand_name", ""),
        previous=previous[:3000],
        current=state.get("final_report", "")[:3000],
    )
    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [langfuse_handler], "run_name": "trend_compare"},
        )
        delta = response.content.strip()
    except Exception as e:
        print(f"[trend_compare] LLM error: {e}")
        delta = ""

    print(f"[trend_compare] Delta generated ({len(delta)} chars)")
    return {
        "trend_delta":  delta,
        "current_node": "trend_compare",
        "status_log":   state.get("status_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────
# VOICE SUMMARY — standalone callable (not in graph)
# Called by app.py after run_agent() when is_voice_mode=True
# ─────────────────────────────────────────────

_VOICE_SUMMARY_FALLBACK = """\
You are summarising a competitor intelligence report to be read aloud.

Rules — follow exactly:
- No markdown, no bullet points, no section headers.
- Plain spoken sentences only.
- Under 100 words total.
- Cover: top 2 competitors found, biggest threat, one market gap, one recommended action.

Report:
{{report}}

Output only the spoken summary. Nothing else.\
"""


def voice_summary_node(state: ResearchState) -> dict:
    """
    Condense the full research report into a short spoken-style paragraph.

    Called directly (not as a graph node) from app.py when is_voice_mode is True.
    Produces ≤100 words of plain prose with no markdown — safe to pass straight
    to edge-tts for speech synthesis.

    Returns a partial state dict with voice_summary populated.
    """
    report = state.get("final_report", "")
    if not report:
        return {"voice_summary": "No report available to summarise."}

    prompt = get_langfuse_prompt(
        "voice-summary",
        _VOICE_SUMMARY_FALLBACK,
        report=report[:4000],
    )
    try:
        response = llm.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [langfuse_handler], "run_name": "voice_summary"},
        )
        summary = response.content.strip()
    except Exception as e:
        print(f"[voice_summary_node] LLM error: {e}")
        summary = "Could not generate voice summary."

    return {"voice_summary": summary}


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────
def build_graph():
    """
    Compile and return the LangGraph StateGraph for a single research run.

    Returns a compiled graph that can be invoked or streamed. A new graph is
    compiled per request so each Streamlit session gets an independent run.
    """
    graph = StateGraph(ResearchState)
    graph.add_node("identify_brand",    identify_brand_node)
    graph.add_node("search",            search_node)
    graph.add_node("scrape",            scrape_node)
    graph.add_node("check_sufficiency", check_sufficiency_node)
    graph.add_node("generate_report",   generate_report_node)
    graph.add_node("trend_compare",     trend_compare_node)
    graph.add_node("store_memory",      store_memory_node)

    graph.add_edge(START, "identify_brand")
    graph.add_edge("identify_brand", "search")
    graph.add_edge("search", "scrape")
    graph.add_edge("scrape", "check_sufficiency")
    graph.add_conditional_edges("check_sufficiency", should_continue,
                                {"search": "search", "generate": "generate_report"})
    graph.add_edge("generate_report", "trend_compare")
    graph.add_edge("trend_compare",   "store_memory")
    graph.add_edge("store_memory", END)
    return graph.compile()


def make_initial_state(brand_url: str, brand_name: str = "") -> ResearchState:
    """
    Construct a zero-value ResearchState for the start of a research run.

    URL normalisation happens here so identify_brand_node never receives
    a bare domain like "notion.so" that would fail requests.get().
    """
    # Ensure the URL has a scheme — requests.get() requires it
    if brand_url and not brand_url.startswith(("http://", "https://")):
        brand_url = "https://" + brand_url
    return {
        "brand_name":     brand_name,
        "brand_url":      brand_url,
        "brand_logo":     "",
        "brand_context":  "",
        "brand_website":  "",
        "brand_industry": "",
        "search_results":  [],
        "scraped_content": [],
        "reddit_insights": [],
        "news_insights":   [],
        "pricing_data":    [],
        "iterations":      0,
        "is_sufficient":   False,
        "final_report":    "",
        "status_log":      [],
        "current_node":    "",
        "has_live_data":   False,
        "voice_summary":   "",
        "is_voice_mode":   False,
        "trend_delta":     "",
    }


# ═══════════════════════════════════════════════════════════════════════
# AD CREATIVE ANALYZER — separate LangGraph graph
#
# Graph topology:
#   START → scrape_creative → analyse_creative → score_creative
#                                                       │
#                                             store_creative_memory → END
#
# Kept fully separate from the competitor-research graph so the two
# features can evolve independently and run concurrently without sharing
# state.
# ═══════════════════════════════════════════════════════════════════════


class CreativeState(TypedDict):
    """Shared state for the creative analysis graph."""
    url:              str
    industry:         str
    headlines:        list    # h1/h2/h3 marketing copy (plan names filtered out)
    plan_names:       list    # pricing tier labels separated from real headlines
    ctas:             list    # button and CTA anchor text
    meta_title:       str
    meta_description: str
    image_alts:       list    # non-empty <img> alt attributes
    price_mentions:   list    # regex-matched currency strings
    raw_content:          str     # labelled multi-page content, capped at 2400 chars
    word_count:           int
    pages_scraped:        list   # e.g. ["homepage", "pricing", "features"]
    used_tavily_fallback: bool   # True when static word count < 200
    creative_report:      str    # 7-section markdown from analyse node
    creative_scores:  dict    # {clarity, reason, emotional_impact, reason, cta_effectiveness, reason, trust_signals, reason, overall}
    creative_verdict: str     # one-sentence summary of biggest strength + weakness
    status_log:       list
    current_node:     str


# ── Prompt fallbacks (override in Langfuse UI as "analyse-creative" / "score-creative") ──

_ANALYSE_CREATIVE_FALLBACK = """\
You are an expert ad creative strategist analysing a competitor's landing page.

URL: {{url}}
Industry: {{industry}}

EXTRACTED CREATIVE ELEMENTS:
Marketing headlines:  {{headlines}}
Pricing plan names:   {{plan_names}}
CTA buttons / links:  {{ctas}}
Page title:           {{meta_title}}
Meta description:     {{meta_description}}
Image alt texts:      {{images}}
Price mentions:       {{prices}}
Word count:           {{word_count}}
Page content:         {{content}}

INDUSTRY BENCHMARK — {{industry}}:
Compare against what top companies in {{industry}} typically do:
- SaaS / Productivity: free trial CTA, transparent pricing, social proof (logos + testimonials), integration list
- AI Marketing: ROI metrics, case studies, before/after demos, speed claims
- E-commerce: urgency (limited stock/time), user reviews, clear returns policy, trust badges
- Fintech: regulatory trust signals, security badges, plain-language fee explanation
- Healthcare: credentials, privacy compliance signals, empathy-led copy
- Developer Tools: code examples, GitHub stars, API documentation links, open-source signals
- Other: apply general best practice (clarity, trust, conversion intent)
If this page is missing standard signals for {{industry}} — explicitly call it out as a weakness.

Write a structured creative intelligence report with EXACTLY these 7 sections (## prefix).
Be specific — quote actual copy from the page. Never make generic observations.

## 1. CREATIVE STRATEGY OVERVIEW
What is their core message? Which emotion or pain point are they targeting?
What makes a visitor stay on this page?

## 2. HEADLINE ANALYSIS
Break down each marketing headline. Label the psychological trigger in [brackets]:
[urgency] [social proof] [curiosity] [fear] [aspiration] [authority] [clarity]
If plan names were extracted, note how they are positioned (aspirational vs functional).

## 3. CTA ANALYSIS
What CTAs are they using? What action do they want?
Rate conversion pressure: soft / medium / aggressive.
What friction is present or absent?

## 4. TONE & MESSAGING
Formal or casual? Feature-focused or benefit-focused?
What words and phrases repeat most? What brand voice archetype applies?

## 5. PRICING STRATEGY
How is pricing presented? Free trial / freemium / enterprise-only?
What psychological anchoring or framing techniques are visible?
If no pricing is shown — what does that signal about their sales motion?

## 6. WEAKNESSES & GAPS
What is missing vs the {{industry}} benchmark?
Which objections are NOT addressed?
Where could a competitor beat them on messaging?

## 7. HOW TO BEAT THEM
5 specific, actionable recommendations. Be direct:
"Instead of X, do Y — because Z."\
"""

_SCORE_CREATIVE_FALLBACK = """\
You are scoring the ad creative quality of a landing page based ONLY on evidence in the data below.

Creative analysis report:
{{report}}

Extracted data summary:
  CTAs found:         {{cta_count}}
  Trust signals found (testimonials/logos/reviews in content): {{trust_found}}
  Price mentions:     {{price_count}}
  Headlines count:    {{headline_count}}

CRITICAL RULES — scores must be grounded in extracted evidence:
- If trust signals (testimonials, reviews, logos, guarantees) were NOT found → trust_signals MUST be below 5
- If price mentions were NOT found on the page → pricing transparency score should reflect that gap
- If fewer than 3 CTAs were extracted → cta_effectiveness CANNOT exceed 6
- NEVER score higher than the evidence supports
- Each score MUST include a one-line reason citing specific evidence

Respond with ONLY valid JSON — no other text, no markdown fences:
{
  "clarity": <integer 1-10>,
  "clarity_reason": "<one line citing specific evidence>",
  "emotional_impact": <integer 1-10>,
  "emotional_impact_reason": "<one line citing specific evidence>",
  "cta_effectiveness": <integer 1-10>,
  "cta_effectiveness_reason": "<one line citing specific evidence>",
  "trust_signals": <integer 1-10>,
  "trust_signals_reason": "<one line citing specific evidence>",
  "overall": <float 1.0-10.0>
}

Scoring guide:
  clarity           : How clear is the value proposition? (1=confusing, 10=crystal clear)
  emotional_impact  : Does it trigger desire or urgency? (1=flat, 10=compelling)
  cta_effectiveness : Are CTAs strong and action-driving? (1=weak, 10=irresistible)
  trust_signals     : Testimonials, logos, guarantees, social proof? (1=none found, 10=abundant)
  overall           : Weighted holistic score — clarity 30%, emotional 20%, CTA 25%, trust 25%\
"""

_VERDICT_FALLBACK = """\
You are writing a one-sentence creative verdict.

Scores:
  Clarity:          {{clarity}}/10
  Emotional Impact: {{emotional_impact}}/10
  CTA Effectiveness:{{cta_effectiveness}}/10
  Trust Signals:    {{trust_signals}}/10
  Overall:          {{overall}}/10

Write EXACTLY ONE sentence (max 25 words) that identifies the biggest strength AND the biggest weakness.
Format: "[strength] — [weakness and how a competitor could exploit it]"
No markdown. No labels. Just the sentence.\
"""


# ─────────────────────────────────────────────
# CREATIVE NODE 1: scrape_creative
# ─────────────────────────────────────────────
def scrape_creative_node(state: CreativeState) -> CreativeState:
    """
    Scrape the target URL and extract all creative signal elements.

    Delegates to scrape_creative_page() in tools.py which does the deep
    BeautifulSoup extraction. Populates every raw-data field in CreativeState.
    """
    url = state["url"]
    log_entry = f"Scraping creative elements from {url}..."
    print(log_entry)

    data = scrape_creative_page(url)

    pages        = data["pages_scraped"]
    used_tavily  = data["used_tavily_fallback"]
    log_entry2   = (
        f"Scraped {len(pages)} page(s): {' · '.join(pages)}"
        + (" + Tavily fallback" if used_tavily else "")
    )
    print(log_entry2)

    return {
        **state,
        "headlines":            data["headlines"],
        "plan_names":           data["plan_names"],
        "ctas":                 data["ctas"],
        "meta_title":           data["meta_title"],
        "meta_description":     data["meta_description"],
        "image_alts":           data["image_alts"],
        "price_mentions":       data["price_mentions"],
        "raw_content":          data["raw_content"],
        "word_count":           data["word_count"],
        "pages_scraped":        pages,
        "used_tavily_fallback": used_tavily,
        "current_node":         "scrape_creative",
        "status_log":           state.get("status_log", []) + [log_entry, log_entry2],
    }


# ─────────────────────────────────────────────
# CREATIVE NODE 2: analyse_creative
# ─────────────────────────────────────────────
def analyse_creative_node(state: CreativeState) -> CreativeState:
    """
    Send all extracted creative elements to the LLM for structured analysis.

    Feeds each extracted field into the prompt individually so the LLM
    can reason about each signal type before synthesising the report.
    """
    log_entry = "Analysing creative strategy..."
    print(log_entry)

    prompt = get_langfuse_prompt(
        "analyse-creative",
        _ANALYSE_CREATIVE_FALLBACK,
        url=state["url"],
        industry=state.get("industry", "unknown"),
        headlines="\n".join(state.get("headlines", [])) or "None extracted.",
        plan_names=", ".join(state.get("plan_names", [])) or "None found.",
        ctas="\n".join(state.get("ctas", [])) or "None extracted.",
        meta_title=state.get("meta_title", ""),
        meta_description=state.get("meta_description", ""),
        images="\n".join(state.get("image_alts", [])) or "None extracted.",
        prices=", ".join(state.get("price_mentions", [])) or "None extracted.",
        word_count=state.get("word_count", 0),
        content=state.get("raw_content", "")[:2500],
    )
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse_handler], "run_name": "analyse_creative"},
    )
    log_entry2 = "Creative analysis complete."
    print(log_entry2)

    return {
        **state,
        "creative_report": response.content,
        "current_node":    "analyse_creative",
        "status_log":      state.get("status_log", []) + [log_entry, log_entry2],
    }


# ─────────────────────────────────────────────
# CREATIVE NODE 3: score_creative
# ─────────────────────────────────────────────
def score_creative_node(state: CreativeState) -> CreativeState:
    """
    Ask the LLM to produce a 5-dimension JSON scorecard for the creative.

    Passes the analysis report rather than raw extracted data — the LLM
    can score more accurately from a synthesised view than raw HTML snippets.
    Reuses extract_json() for robust JSON parsing with markdown-fence stripping.
    """
    log_entry = "Scoring creative dimensions..."
    print(log_entry)

    ctas       = state.get("ctas", [])
    prices     = state.get("price_mentions", [])
    headlines  = state.get("headlines", [])
    report_txt = state.get("creative_report", "")

    # Detect trust signals from the report text so the scoring prompt knows
    # whether testimonials/logos were actually mentioned in the analysis.
    trust_keywords = ("testimonial", "review", "logo", "customer", "case study",
                      "guarantee", "rating", "stars", "clients", "trusted by")
    trust_found = any(kw in report_txt.lower() for kw in trust_keywords)

    prompt = get_langfuse_prompt(
        "score-creative",
        _SCORE_CREATIVE_FALLBACK,
        report=report_txt[:3000],
        url=state["url"],
        cta_count=len(ctas),
        trust_found="YES" if trust_found else "NO — trust signals not found in extracted content",
        price_count=len(prices),
        headline_count=len(headlines),
    )
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse_handler], "run_name": "score_creative"},
    )

    scores = extract_json(response.content)

    # Clamp numeric scores to 1-10; enforce evidence-based caps
    for key in ("clarity", "emotional_impact", "cta_effectiveness", "trust_signals"):
        if key in scores:
            scores[key] = max(1, min(10, int(scores[key])))
    if "overall" in scores:
        scores["overall"] = max(1.0, min(10.0, float(scores["overall"])))

    # Hard-enforce the evidence rules even if LLM ignores them
    if not trust_found and scores.get("trust_signals", 0) >= 5:
        scores["trust_signals"] = 4
        scores["trust_signals_reason"] = scores.get(
            "trust_signals_reason", "No testimonials or trust signals found in extracted content"
        )
    if len(ctas) < 3 and scores.get("cta_effectiveness", 0) > 6:
        scores["cta_effectiveness"] = 6
        scores["cta_effectiveness_reason"] = scores.get(
            "cta_effectiveness_reason", f"Only {len(ctas)} CTA(s) extracted — capped at 6"
        )

    print(f"[score_creative] Scores: {scores}")

    return {
        **state,
        "creative_scores": scores,
        "current_node":    "score_creative",
        "status_log":      state.get("status_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────
# CREATIVE NODE 4: verdict_creative
# ─────────────────────────────────────────────
def verdict_creative_node(state: CreativeState) -> CreativeState:
    """
    Produce a single-sentence verdict summarising the creative's biggest
    strength and biggest weakness.

    Called after scoring so the verdict is grounded in the evidence-based
    scores rather than raw extractions. A small, fast LLM call — output
    is ≤25 words and displayed prominently above the scorecard.
    """
    log_entry = "Writing creative verdict..."
    print(log_entry)

    scores = state.get("creative_scores", {})
    prompt = get_langfuse_prompt(
        "verdict-creative",
        _VERDICT_FALLBACK,
        clarity=scores.get("clarity", 0),
        emotional_impact=scores.get("emotional_impact", 0),
        cta_effectiveness=scores.get("cta_effectiveness", 0),
        trust_signals=scores.get("trust_signals", 0),
        overall=scores.get("overall", 0),
    )
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [langfuse_handler], "run_name": "verdict_creative"},
    )
    verdict = response.content.strip().strip('"')
    print(f"[verdict_creative] {verdict}")

    return {
        **state,
        "creative_verdict": verdict,
        "current_node":     "verdict_creative",
        "status_log":       state.get("status_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────
# CREATIVE NODE 5: store_creative_memory
# ─────────────────────────────────────────────
def store_creative_memory_node(state: CreativeState) -> CreativeState:
    """
    Persist the creative report to the dedicated creative_memory Chroma collection.

    Kept in a separate collection from research_memory so competitive research
    and creative analyses can be recalled independently.
    """
    try:
        save_creative(state["url"], state.get("creative_report", ""))
    except Exception as e:
        print(f"[store_creative_memory] Warning: {e}")
    return {
        **state,
        "current_node": "store_creative_memory",
        "status_log":   state.get("status_log", []) + ["Creative analysis saved to memory."],
    }


# ─────────────────────────────────────────────
# CREATIVE GRAPH
# ─────────────────────────────────────────────
def build_creative_graph():
    """
    Compile and return the LangGraph StateGraph for a single creative analysis run.

    Linear 4-node pipeline — no conditional loops needed since creative analysis
    is always deterministic (scrape → analyse → score → store).
    """
    graph = StateGraph(CreativeState)
    graph.add_node("scrape_creative",        scrape_creative_node)
    graph.add_node("analyse_creative",       analyse_creative_node)
    graph.add_node("score_creative",         score_creative_node)
    graph.add_node("verdict_creative",       verdict_creative_node)
    graph.add_node("store_creative_memory",  store_creative_memory_node)

    graph.add_edge(START, "scrape_creative")
    graph.add_edge("scrape_creative",       "analyse_creative")
    graph.add_edge("analyse_creative",      "score_creative")
    graph.add_edge("score_creative",        "verdict_creative")
    graph.add_edge("verdict_creative",      "store_creative_memory")
    graph.add_edge("store_creative_memory", END)
    return graph.compile()


def make_creative_state(url: str, industry: str = "") -> CreativeState:
    """Construct a zero-value CreativeState for the start of a creative analysis run."""
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return {
        "url":              url,
        "industry":         industry,
        "headlines":        [],
        "plan_names":       [],
        "ctas":             [],
        "meta_title":       "",
        "meta_description": "",
        "image_alts":       [],
        "price_mentions":   [],
        "raw_content":          "",
        "word_count":           0,
        "pages_scraped":        [],
        "used_tavily_fallback": False,
        "creative_report":      "",
        "creative_scores":  {},
        "creative_verdict": "",
        "status_log":       [],
        "current_node":     "",
    }
