from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from config.settings import Settings


class ChromaVectorStore:
    """Wrapper sobre Chroma con operaciones de gestión de colección."""

    def __init__(self, embeddings: Embeddings, settings: Settings) -> None:
        self.store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    def add_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)

    def delete_collection(self) -> None:
        self.store.delete_collection()

    def get_collection_stats(self) -> dict[str, Any]:
        collection = self.store._collection  # type: ignore[attr-defined]
        return {"name": collection.name, "count": collection.count()}

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k, filter=filter)
