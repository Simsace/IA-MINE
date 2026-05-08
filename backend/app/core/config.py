from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Mines Mali RAG API"
    app_version: str = "1.0.0"
    environment: str = "local"
    log_level: str = "INFO"

    # CORS: use ["*"] for local development, restrict this in production.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Data and vector store paths.
    chunks_dir: Path = Path("../data_mines_mali/chunks")
    data_dir: Path = Path("data")
    faiss_index_path: Path = Path("data/mines_index.faiss")
    metadata_path: Path = Path("data/metadata.json")

    # Embeddings.
    embedding_model_name: str = "all-MiniLM-L6-v2"
    hf_cache_dir: Path = Path("data/.hf-cache")
    local_models_dir: Path = Path("data/.models")

    # OpenAI-compatible LLM.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 60
    llm_max_tokens: int = 600
    engine_name: str = "openai-rag"
    # LLM provider: 'openai' or 'gemini'
    llm_provider: str = "openai"
    gemini_api_key: str | None = None

    default_top_k: int = 5
    max_top_k: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.local_models_dir.mkdir(parents=True, exist_ok=True)
    return settings

