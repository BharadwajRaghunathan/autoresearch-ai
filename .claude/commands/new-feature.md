# /new-feature

Scaffold a complete new feature tab in AutoResearch AI.

## Instructions

The user will describe a new intelligence feature (e.g. "SEO Analyzer", "Pricing Tracker"). Follow this checklist:

### 1. app.py — Add tab
- Add tab name to `st.tabs([...])`
- Build the tab UI inside a `with tabN:` block
- Follow the existing tab patterns: `st.text_input` → button → spinner → progress tracker → results
- Inject `st.secrets` into `os.environ` if the new tab needs a new API key

### 2. agent.py — New graph
- Define a new `TypedDict` state (e.g. `SeoState`) with all fields needed
- Write node functions following the existing pattern (return partial state dict, update `status_log` + `current_node`)
- Define `_*_FALLBACK` constants for every LLM call
- Build the graph with `build_seo_graph()` (or similar) using `StateGraph`
- Export a `make_*_state(url)` initializer

### 3. tools.py — New scraper/search (if needed)
- Add extractor function following `scrape_creative_page()` pattern
- Apply the 800-char per-page cap and 2400-char total cap
- Add Tavily fallback if the feature may hit JS-rendered sites

### 4. memory.py — New collection (if needed)
- Add a new `_*_store = None` module variable
- Init the Chroma collection inside the existing try/except block
- Add `save_*()` function following `save_creative()` pattern

### 5. Langfuse prompts
- List every LLM call in the new graph and the prompt name it should use
- Remind the user to create those prompts in Langfuse UI
- Inline fallback constants are already in agent.py (step 2)

### Anti-hallucination checklist
- [ ] Industry or context injected into every report prompt?
- [ ] Evidence counts passed to scoring LLM?
- [ ] Python hard-override for any score rules?
- [ ] URL-first: agent scrapes provided URL, never guesses?

## Example usage
`/new-feature — Build an SEO Gap Analyzer tab that compares meta tags, heading structure, and keyword density`
