from __future__ import annotations
from typing import TYPE_CHECKING
from langchain_aws import BedrockEmbeddings
if TYPE_CHECKING:
    from config.settings import Settings

def get_embeddings(settings: "Settings") -> BedrockEmbeddings:
    """
    Crear y devolver una instancia de embeddings de AWS Bedrock.
    """
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_default_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
