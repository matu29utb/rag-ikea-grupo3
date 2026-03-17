"""
ChromaDB vector store wrapper.

Persists embeddings to disk so they survive restarts.
Provides CRUD helpers and convenience methods consumed by the retrieval layer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


class ChromaVectorStore:
    """Thin wrapper around LangChain's Chroma integration."""

    def __init__(self, embeddings, settings: "Settings") -> None:
        self.embeddings = embeddings
        self.settings = settings
        self._store: Optional[Chroma] = None

    # ──────────────────────────────────────────────────────────────────────
    # Internal store (lazy-initialised)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.settings.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
            )
        return self._store

    # ──────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents. Returns the list of assigned IDs."""
        ids = self.store.add_documents(documents)
        logger.info(f"Indexed {len(documents)} chunk(s) into ChromaDB")
        return ids

    # ──────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Return the *k* most relevant documents for *query*."""
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents together with their cosine-relevance scores."""
        return self.store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )

    def as_retriever(
        self,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity",
    ):
        """Expose a LangChain-compatible BaseRetriever interface."""
        search_kwargs: Dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    # ──────────────────────────────────────────────────────────────────────
    # Admin
    # ──────────────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the current collection."""
        col = self.store._collection
        return {"name": col.name, "count": col.count()}

    def delete_collection(self) -> None:
        """Drop the entire collection (irreversible!)."""
        self.store.delete_collection()
        self._store = None
        logger.warning("ChromaDB collection deleted")