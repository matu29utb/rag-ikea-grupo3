"""
Configuración central con Pydantic Settings.
Lee variables del .env y las valida.
Especializado para RAG de documentos bancarios.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ─────────────────────────────────────────────────────────────────
    # AWS Credentials
    # ─────────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")
    aws_region: str = Field(
        "eu-west-1", validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION")
    )

    # ─────────────────────────────────────────────────────────────────
    # Bedrock Models
    # ─────────────────────────────────────────────────────────────────
    llm_model_id: str = Field(
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0", alias="LLM_MODEL_ID"
    )
    embedding_model_id: str = Field(
        "amazon.titan-embed-text-v2:0", alias="EMBED_MODEL_ID"
    )

    # ─────────────────────────────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────────────────────────────
    top_k: int = Field(5, alias="TOP_K")

    # ─────────────────────────────────────────────────────────────────
    # ChromaDB
    # ─────────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("ikea_catalog", alias="CHROMA_COLLECTION_NAME")


# Instancia global
settings = Settings()  # type: ignore[call-arg]
