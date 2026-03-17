from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings
    from src.vectorstore.chroma_store import ChromaVectorStore


class SmartRetriever:
    """Retriever configurable con soporte para filtrado por metadatos."""

    def __init__(self, vector_store: "ChromaVectorStore", settings: "Settings") -> None:
        self.vector_store = vector_store
        self.settings = settings

    def get_retriever(
        self,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity",
    ):
        """
        Devuelve un BaseRetriever de LangChain.

        Args:
            k: Número de documentos a recuperar.
            filter_dict: Filtro de metadatos para ChromaDB (ej. {"category": "sofas"}).
            search_type: Tipo de búsqueda ("similarity", "mmr", "similarity_score_threshold").
        """
        top_k = k or self.settings.top_k

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        logger.debug(
            f"Building retriever — search_type={search_type}, k={top_k}, filter={filter_dict}"
        )
        return self.vector_store.store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Recupera documentos para una consulta con filtro opcional."""
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search(query, k=top_k, filter=filter_dict)

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Recupera documentos junto con sus puntuaciones de relevancia."""
        top_k = k or self.settings.top_k
        return self.vector_store.similarity_search_with_score(
            query, k=top_k, filter=filter_dict
        )
