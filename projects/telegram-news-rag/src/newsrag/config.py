from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env and data/ from the project folder, so commands work from any directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    telegram_api_id: int
    telegram_api_hash: str
    telegram_channel: str
    data_dir: Path = PROJECT_ROOT / "data"

    # RAG
    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_model: str = "qwen3:14b"
    anthropic_model: str = "claude-sonnet-5"
    anthropic_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chunk_max_tokens: int = 512
    qdrant_collection: str = "news"

    @field_validator("data_dir")
    @classmethod
    def _anchor_to_project(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def qdrant_path(self) -> Path:
        # Qdrant "local mode": same API as the server, stored in a folder, no Docker needed
        return self.data_dir / "qdrant"

    @property
    def session_path(self) -> Path:
        return self.data_dir / "telegram"


@lru_cache
def get_settings() -> Settings:
    return Settings()
