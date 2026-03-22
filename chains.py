"""
chains.py — LLM and observability setup for AutoResearch AI.

Exports:
    llm               : ChatGroq instance wired with LangSmith auto-tracing.
    langfuse_handler  : Langfuse callback handler attached to every LLM call.
    get_langfuse_prompt: Fetch a live prompt from Langfuse UI; fall back to inline.

Observability strategy:
    - LangSmith activates automatically when LANGCHAIN_TRACING_V2=true and
      LANGCHAIN_API_KEY are set — no extra code needed.
    - Langfuse provides a secondary trace + prompt management layer via callback.
      All three agent prompts (verify-brand, check-sufficiency, generate-report)
      can be edited live in the Langfuse UI without redeploying.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

load_dotenv()

# Langfuse v4 SDK reads LANGFUSE_HOST from env, but our .env exposes
# LANGFUSE_BASE_URL (the legacy name). Bridge the gap at import time
# so downstream code never needs to know about this inconsistency.
if not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# One handler instance per process — reused for every LLM call.
langfuse_handler = LangfuseCallbackHandler()

# Separate client for prompt management (get_prompt is not on the callback handler).
langfuse_client = Langfuse()


def get_langfuse_prompt(name: str, fallback: str, **variables) -> str:
    """
    Fetch a named prompt from Langfuse and compile its template variables.

    Why this pattern: prompts can be updated in Langfuse UI without redeploying.
    The fallback ensures the agent still works if Langfuse is unreachable or the
    prompt hasn't been created yet. Both Langfuse and inline prompts use the same
    {{variable_name}} syntax, so switching between them is transparent.

    Args:
        name      : Langfuse prompt name (e.g. "generate-report").
        fallback  : Inline prompt string with {{variable}} placeholders.
        **variables: Values to substitute into the template.

    Returns:
        Compiled prompt string ready to pass to the LLM.
    """
    try:
        prompt_obj = langfuse_client.get_prompt(name)
        return prompt_obj.compile(**variables)
    except Exception as e:
        print(f"[langfuse_prompt] '{name}' not in Langfuse ({e}), using inline fallback.")
        result = fallback
        for k, v in variables.items():
            result = result.replace("{{" + k + "}}", str(v))
        return result


# llama-3.3-70b-versatile: active, instruction-following, strong JSON output.
# temperature=0.3 keeps reports factual while still producing fluent prose.
# LangSmith tracing is activated automatically via LANGCHAIN_TRACING_V2 env var.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)
