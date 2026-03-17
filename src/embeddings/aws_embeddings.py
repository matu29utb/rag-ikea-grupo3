# from __future__ import annotations
# from langchain_aws import BedrockEmbeddings

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
