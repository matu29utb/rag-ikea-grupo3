# from __future__ import annotations
# from langchain_aws import BedrockEmbeddings

<<<<<<< HEAD
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
=======
# if TYPE_CHECKING:
#   from config.settings import Settings


# def get_embeddings(settings: "Settings") -> BedrockEmbeddings:
#     """Create and return an AWS Bedrock embeddings instance.

#     Credentials are read from the environment variables configured in .env
#     (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
#     """
#     return BedrockEmbeddings(
#         model_id=settings.embedding_model_id,
#         region_name=settings.aws_default_region,
#         aws_access_key_id=settings.aws_access_key_id,
#         aws_secret_access_key=settings.aws_secret_access_key,
#     )
>>>>>>> 82e65035b7759d14da1d5b1f31ebd1de3e3a442d
