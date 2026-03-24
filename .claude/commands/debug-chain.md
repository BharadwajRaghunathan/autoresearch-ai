# /debug-chain

Debug a failing or misbehaving LangGraph node or LLM call.

## Instructions

When the user reports an error or unexpected output, follow this diagnostic flow:

### Step 1 — Identify the failure point
- Which node is failing? (check `status_log` in state or the LangSmith trace)
- Is it a scraping failure, LLM failure, or scoring/output failure?

### Step 2 — Scraping issues
- `word_count < 200` after BS4? → Check if Tavily fallback triggered (`used_tavily_fallback`)
- Empty headlines/CTAs? → Check if the target page is JS-rendered (Tavily fallback should handle it)
- Price mentions missing? → Check if `/pricing` auto-discovery succeeded (`pages_scraped` list)
- Plan names showing up in headlines? → Check `_is_plan_name()` filter in `tools.py`

### Step 3 — LLM failures
- Groq 413/context error? → Check total content length; cap is 2400 chars in `scrape_creative_page()`
- Hallucinated competitors? → Check `brand_industry` is extracted and injected into the prompt
- Wrong company in report? → Check `identify_brand` node — was the URL scraped successfully?
- Langfuse prompt not found? → Fallback constant in `agent.py` should have activated — check logs for `[langfuse_prompt]`

### Step 4 — Scoring issues
- Trust score too high despite no trust signals? → Check Python hard-override block in `score_creative_node`
- Scores not grounded? → Check `cta_count`, `trust_found`, `price_count` are passed to score prompt
- JSON parse error from score LLM? → The node wraps in try/except and defaults to 5s — check agent.py score_creative_node

### Step 5 — Streamlit Cloud deployment
- `GROQ_API_KEY not found`? → Check that `app.py` injects `st.secrets` into `os.environ` BEFORE importing from agent/chains
- Chroma error on Cloud? → Chroma uses `./chroma_db/` local path — Cloud filesystem is ephemeral, this is expected and handled (memory becomes no-op)

### Step 6 — Parallel Compare tab
- UI not updating? → Threads must write to `brand_status` dict ONLY; never call `st.*` from background threads
- One brand hanging? → Check `_research_brand_silent` has proper try/except; failed brands write `status: "error"` to dict

## Example usage
`/debug-chain — Creative Decoder returns empty scorecard for notion.com`
