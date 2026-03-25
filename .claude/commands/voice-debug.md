# /voice-debug

Debug issues with the Voice Research tab.

## Diagnostics

### STT (Speech to Text) issues
- `ImportError: faster-whisper not installed` → Run `pip install faster-whisper`
- First run downloads ~150MB Whisper base model — slow once, cached after
- Empty transcript → Audio too short/silent; try speaking louder
- Wrong URL extracted → Transcript parsed by splitting on whitespace and finding token with "." — if brand name has no dot, user should say the full URL ("notion dot so")
- `_VOICE_AVAILABLE = False` banner shown → Import failed silently at startup — check `pip list | grep faster-whisper`

### TTS (Text to Speech) issues
- `ImportError: edge-tts not installed` → Run `pip install edge-tts`
- `edge_tts` requires internet (streams from Microsoft) — fails offline
- `asyncio.run()` error in Jupyter/existing event loop → Not an issue in Streamlit (runs in a thread), but flag if seen
- `voice_response.mp3` already open → Streamlit autoplay holds the file; harmless, overwrites on next run

### voice_summary_node issues
- Empty summary → `final_report` was empty — check research completed successfully first
- Generic summary → `voice-summary` prompt not in Langfuse; inline fallback active — create the prompt in Langfuse UI
- Summary longer than 100 words → LLM ignored the word limit; tighten the Langfuse prompt with explicit "HARD LIMIT: 100 words"

### UI issues
- Button disabled when audio recorded → `st.audio_input` returns None until user records; this is expected
- `_run_voice_research` not defined error → Function must be defined BEFORE the Tab 4 block in app.py — check ordering
- Report renders but no audio plays → `st.audio(autoplay=True)` requires modern browser; test in Chrome

## Voice feature architecture reminder
- `voice.py` — STT + TTS only; no LangGraph, no LLM
- `voice_summary_node` — lives in agent.py, called directly from app.py (NOT in any graph)
- Full research uses existing `build_graph()` — voice mode is just a post-processing layer
- `is_voice_mode: bool` field on ResearchState exists but is not currently used by any node — reserved for future conditional behaviour
