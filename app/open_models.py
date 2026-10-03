"""Map the brief's short model names onto open-weight slugs.

Backboard's API defaults to a closed model when `llm_provider` is omitted.
Every call in this app sets an open provider and resolves these names first.
"""

# Closed providers the challenge disqualifies. Refused before any HTTP call.
CLOSED_PROVIDERS = {"openai", "anthropic", "google", "xai"}

# Short names used in the comparison script and .env.example.
OPEN_MODEL_SLUGS = {
    "llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct": "qwen/qwen-2.5-7b-instruct",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct",
}


def resolve_open_model(name: str) -> str:
    """Return an OpenRouter slug. Full slugs (containing /) pass through."""
    cleaned = (name or "").strip()
    return OPEN_MODEL_SLUGS.get(cleaned, cleaned)


def assert_open_provider(provider: str) -> str:
    cleaned = (provider or "").strip().lower()
    if cleaned in CLOSED_PROVIDERS or not cleaned:
        raise RuntimeError(
            f"Refusing closed or empty LLM provider {provider!r}. "
            "Use an open provider such as openrouter."
        )
    return cleaned
