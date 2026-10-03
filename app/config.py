from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment configuration. Closed model keys are intentionally absent."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./duster.db"
    analyzer: str = "heuristic"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Backboard fronts open-weight models. The provider must stay open —
    # the API defaults to OpenAI if this is omitted, which this app refuses.
    backboard_api_key: str = ""
    backboard_base_url: str = "https://app.backboard.io/api"
    backboard_llm_provider: str = "openrouter"
    backboard_model: str = "qwen/qwen-2.5-7b-instruct"
    compare_models: str = "llama-3.1-8b-instruct,qwen2.5-7b-instruct,mistral-7b-instruct"

    tinker_api_key: str = ""
    tinker_model_name: str = ""

    # faster-whisper model size. "base" is the CPU-friendly default.
    whisper_model: str = "base"


settings = Settings()
