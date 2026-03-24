# AutoResearch AI — Claude Code Context

## Project Overview
Autonomous marketing intelligence agent. Three features:
1. **Brand Research** — paste a URL → 8-section competitor intelligence report
2. **Compare Brands** — paste up to 3 URLs → parallel research + cross-brand summary
3. **Creative Decoder** — paste a competitor landing page → 7-section creative analysis + scorecard + verdict

Stack: Python 3.13, LangGraph, LangChain, Groq (llama-3.3-70b-versatile), Tavily, BeautifulSoup, Chroma, Streamlit, LangSmith, Langfuse.

---

## File Responsibilities

| File | What it owns |
|---|---|
| `app.py` | Streamlit UI — 3 tabs, HTML renderers, threading for Compare tab, streaming LangGraph events |
| `agent.py` | Both LangGraph graphs (`build_graph` + `build_creative_graph`), all node functions, both TypedDict state schemas |
| `chains.py` | Groq LLM init, Langfuse callback handler, `get_langfuse_prompt()` with inline fallback |
| `tools.py` | Tavily search, BeautifulSoup scraper (`scrape_website`), creative extractor (`scrape_creative_page`) |
| `memory.py` | Chroma — two collections: `research_memory` (brands) + `creative_memory` (URLs) |

---

## LangGraph Graphs

### build_graph() — Research (6 nodes)
```
identify_brand → search → scrape → check_sufficiency → generate_report → store_memory
                              ↑__________________|  (loop if not sufficient, max 3 iter)
```
- State: `ResearchState` (brand_name, brand_url, brand_industry, search_results, scraped_content, iterations, is_sufficient, final_report, status_log, current_node)
- Streaming: every node emits `current_node` + `status_log` entries — app.py polls these

### build_creative_graph() — Creative Decoder (5 nodes)
```
scrape_creative → analyse_creative → score_creative → verdict_creative → store_creative_memory
```
- State: `CreativeState` (url, industry, headlines, plan_names, ctas, meta_title, meta_description, image_alts, price_mentions, raw_content, word_count, creative_report, creative_scores, creative_verdict, pages_scraped, used_tavily_fallback, status_log, current_node)
- Three sequential LLM calls: analyse → score → verdict

---

## LLM & Prompts

**Model**: `llama-3.3-70b-versatile` via Groq, `temperature=0.3`

**Langfuse prompts** (6 total — editable live in Langfuse UI):
| Prompt name | Node | Graph |
|---|---|---|
| `verify-brand` | identify_brand | Research |
| `check-sufficiency` | check_sufficiency | Research |
| `generate-report` | generate_report | Research |
| `analyse-creative` | analyse_creative | Creative |
| `score-creative` | score_creative | Creative |
| `verdict-creative` | verdict_creative | Creative |

Every prompt has an inline fallback constant in `agent.py` (prefixed `_*_FALLBACK`). If Langfuse is unreachable the agent continues normally.

---

## Anti-Hallucination Rules

These are load-bearing design decisions — do not remove them:

1. **URL-first identity** — agent never guesses a URL from a name; always scrapes provided URL or LLM-verifies Tavily candidates
2. **Industry injection** — `brand_industry` extracted from real HTML and injected into every report prompt; LLM explicitly blocked from listing competitors outside that industry
3. **Evidence-based creative scoring** — `cta_count`, `trust_found`, `price_count` passed to score LLM as context. Python hard-overrides after: no trust signals found → trust score capped at 4 regardless of LLM output
4. **Score override logic** is in `score_creative_node` in `agent.py` — do not remove the `if not state["trust_found"]` block

---

## Scraping Rules (tools.py)

- `scrape_creative_page(url)` auto-discovers `/pricing` and `/features` with 3s timeout
- If `word_count < 200` after BS4 → Tavily fallback (for JS-rendered sites)
- Per-page content cap: 800 chars; total cap: 2400 chars (Groq free tier compatibility)
- Plan names filtered via `_is_plan_name()` — keeps `plan_names` separate from `headlines`
- Content labeled by page: `[HOMEPAGE]`, `[PRICING]`, `[FEATURES]`

---

## Parallel Compare Brands (app.py)

- Each brand runs in its own `threading.Thread` via `_research_brand_silent(url, idx, brand_status)`
- Threads write ONLY to shared `brand_status` dict — never touch Streamlit widgets
- Main thread polls `brand_status` every 0.5s and updates 3 columns
- Cross-brand summary generated after all threads join via `_generate_cross_brand_summary(results)`

---

## Memory (memory.py)

- Chroma init is wrapped in try/except — failure makes all functions no-ops, agent continues
- Two collections: `research_memory` (one doc per brand per run) + `creative_memory` (one doc per URL per run)
- IDs include timestamps so re-runs don't overwrite

---

## Streamlit Cloud Deployment

- App reads API keys from `st.secrets` (set in App Settings → Secrets as TOML)
- `app.py` injects `st.secrets` into `os.environ` at module level BEFORE any agent/chains imports
- This is required because `chains.py` reads env vars at import time

---

## Adding a New Node

1. Add field(s) to the relevant TypedDict in `agent.py`
2. Write the node function (takes state, returns partial state dict)
3. Add `graph.add_node("node_name", node_function)`
4. Add `graph.add_edge(...)` in the right place
5. If the node calls the LLM: add a Langfuse prompt (or inline fallback constant)
6. If the node scrapes: add tool function to `tools.py`, import in `agent.py`
7. Update `status_log` in the node so the UI progress tracker shows it

## Adding a New Feature Tab

1. Add the tab in `app.py` (`tab1, tab2, tab3, tab4 = st.tabs([...])`)
2. Build a new LangGraph state + graph in `agent.py` (follow CreativeState pattern)
3. If it needs memory: add a new Chroma collection in `memory.py`
4. If it scrapes: add extractor function in `tools.py`
5. If it uses new prompts: add to Langfuse + inline fallback in `agent.py`
