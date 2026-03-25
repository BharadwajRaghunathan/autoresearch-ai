# AutoResearch AI — Claude Code Context

## Project Overview
Autonomous marketing intelligence agent. Four features:
1. **Brand Research** — paste a URL → 8-section competitor intelligence report
2. **Compare Brands** — paste up to 3 URLs → parallel research + cross-brand summary
3. **Creative Decoder** — paste a competitor landing page → 7-section creative analysis + scorecard + verdict
4. **Voice Research** — speak a brand name → agent researches it → reads key findings aloud

Stack: Python 3.13, LangGraph, LangChain, Groq (llama-3.3-70b-versatile), Tavily, BeautifulSoup, Chroma, Streamlit, LangSmith, Langfuse, faster-whisper (STT), edge-tts (TTS).

---

## File Responsibilities

| File | What it owns |
|---|---|
| `app.py` | Streamlit UI — 3 tabs, HTML renderers, threading for Compare tab, streaming LangGraph events |
| `agent.py` | Both LangGraph graphs (`build_graph` + `build_creative_graph`), all node functions, both TypedDict state schemas |
| `chains.py` | Groq LLM init, Langfuse callback handler, `get_langfuse_prompt()` with inline fallback |
| `tools.py` | Tavily search, BeautifulSoup scraper (`scrape_website`), creative extractor (`scrape_creative_page`) |
| `memory.py` | Chroma — two collections: `research_memory` (brands) + `creative_memory` (URLs) |
| `voice.py` | Sarvam AI STT (`transcribe_sarvam`) + TTS (`speak_sarvam`). Pure functions only — never calls `st.*`. Greeting/ack text constants live here too. |

---

## LangGraph Graphs

### build_graph() — Research (7 nodes)
```
identify_brand → search → scrape → check_sufficiency → generate_report → trend_compare → store_memory
                              ↑__________________|  (loop if not sufficient, max 3 iter)
```
- State: `ResearchState` (brand_name, brand_url, brand_industry, search_results, scraped_content, iterations, is_sufficient, final_report, trend_delta, voice_summary, is_voice_mode, status_log, current_node)
- `trend_compare` node: no-op on first run; diffs current vs previous Chroma report on repeat runs
- Streaming: every node emits `current_node` + `status_log` entries — app.py polls these

### build_creative_graph() — Creative Decoder (5 nodes)
```
scrape_creative → analyse_creative → score_creative → verdict_creative → store_creative_memory
```
- State: `CreativeState` (url, industry, headlines, plan_names, ctas, meta_title, meta_description, image_alts, price_mentions, raw_content, word_count, creative_report, creative_scores, creative_verdict, pages_scraped, used_tavily_fallback, status_log, current_node)
- Three sequential LLM calls: analyse → score → verdict

### Voice Research — not a separate graph
```
st.audio_input → transcribe() → run_agent() [existing graph] → voice_summary_node() → speak_sync()
```
- `voice_summary_node` is a **standalone function** (not wired into any graph) — called directly in `app.py` after research completes
- `ResearchState` has `voice_summary: str` and `is_voice_mode: bool` fields
- STT: Sarvam Saarika API (`transcribe_sarvam`) — requires `SARVAM_API_KEY`
- TTS: Sarvam Bulbul API (`speak_sarvam`) — returns WAV bytes, capped at 500 chars
- Voice degrades gracefully if `SARVAM_API_KEY` is missing — warning shown in UI
- Langfuse prompt: `voice-summary` (override live in UI)

---

## LLM & Prompts

**Model**: `llama-3.3-70b-versatile` via Groq, `temperature=0.3`

**Langfuse prompts** (7 total — editable live in Langfuse UI):
| Prompt name | Node / Function | Graph |
|---|---|---|
| `verify-brand` | identify_brand | Research |
| `check-sufficiency` | check_sufficiency | Research |
| `generate-report` | generate_report | Research |
| `analyse-creative` | analyse_creative | Creative |
| `score-creative` | score_creative | Creative |
| `verdict-creative` | verdict_creative | Creative |
| `voice-summary` | voice_summary_node | Voice (standalone) |
| `trend-compare` | trend_compare_node | Research (node 6) |

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

## Dependency Rule — ALWAYS do this when adding a feature

Before finishing any new feature, check `requirements.txt`:
1. Is every new `import` covered by a pinned package in requirements.txt?
2. Use the package's actual PyPI name (e.g. `sarvamai`, not `sarvam-ai`)
3. Add to the correct section with a comment explaining which feature uses it
4. Do NOT add packages that are already in the stdlib or already listed
5. Test with `pip install -r requirements.txt` before committing

Current package → feature map:
| Package | Used by |
|---|---|
| `langchain*`, `langgraph` | Research + Creative graphs |
| `langfuse`, `langsmith` | Observability |
| `chromadb`, `sentence-transformers`, `langchain-chroma` | Chroma memory |
| `tavily-python` | Competitor search |
| `beautifulsoup4`, `requests` | Web scraping |
| `streamlit` | UI |
| `sarvamai` | Voice tab — Sarvam STT + TTS |
| `python-dotenv` | .env loading |
| `fpdf2` | PDF export (if used) |

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

1. Add the tab in `app.py` (`tab1, tab2, tab3, tab4, tab5 = st.tabs([...])`)
2. Build a new LangGraph state + graph in `agent.py` (follow CreativeState pattern)
3. If it needs memory: add a new Chroma collection in `memory.py`
4. If it scrapes: add extractor function in `tools.py`
5. If it uses new prompts: add to Langfuse + inline fallback in `agent.py`

## Voice Feature Rules

- `voice.py` is optional — `_VOICE_AVAILABLE` flag in `app.py` guards all voice-dependent code
- `voice_summary_node` must NOT be added to any LangGraph graph — call it directly from `app.py`
- TTS output file `voice_response.mp3` is ephemeral — do not persist or commit it (add to `.gitignore` if needed)
- `faster-whisper` base model downloads ~150MB on first call — expected behaviour
- If `speak_sync` fails (network issue with edge-tts), the text summary is still shown in the UI
