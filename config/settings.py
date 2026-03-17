"""
Configuración central con Pydantic Settings.
Lee variables del .env y las valida.
Especializado para RAG de documentos bancarios.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─────────────────────────────────────────────────────────────────
    # AWS Credentials
    # ─────────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, alias="AWS_SESSION_TOKEN")
    aws_region: str = Field(
        "eu-west-1",
        validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION")
    )


# Instancia global
settings = Settings()
