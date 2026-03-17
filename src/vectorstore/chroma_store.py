# """
# Wrapper para el almacenamiento vectorial en ChromaDB

# Guarda las incrustaciones en disco para que se conserven tras los reinicios.
# Implementa métodos de búsqueda por similitud de coseno y filtrado de metadatos.
# """

# from __future__ import annotations

# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from loguru import logger

# if TYPE_CHECKING:
#     from config.settings import Settings


# class ChromaVectorStore:
#     def __init__(self, embeddings, settings: "Settings") -> None:
#         """Integración de Chroma con LangChain."""
#         self.embeddings = embeddings
#         self.settings = settings
#         self._store: Optional[Chroma] = None

#     # Almacenamiento interno

#     @property
#     def store(self) -> Chroma:
#         """
#         Crea el almacenamiento interno de Chroma con los parámetros de configuración
#         """
#         if self._store is None:
#             self._store = Chroma(
#                 collection_name=self.settings.chroma_collection_name,
#                 embedding_function=self.embeddings,
#                 persist_directory=self.settings.chroma_persist_dir,
#             )
#         return self._store

#     # Vectorización

#     def add_documents(self, documents: List[Document]) -> List[str]:
#         """
#         Convierte documentos a vectores y los guarda. Devuelve los IDs generados.
#         """
#         ids = self.store.add_documents(documents)
#         logger.info(f"Indexed {len(documents)} chunk(s) into ChromaDB")
#         return ids

#     # Búsqueda

#     def similarity_search(
#         self,
#         query: str,
#         k: int = 5,
#         filter: Optional[Dict[str, Any]] = None,
#     ) -> List[Document]:
#         """Realiza una búsqueda de similitud y devuelve los k documentos más relevantes"""
#         return self.store.similarity_search(query, k=k, filter=filter)

#     def similarity_search_with_score(
#         self,
#         query: str,
#         k: int = 5,
#         filter: Optional[Dict[str, Any]] = None,
#     ) -> List[Tuple[Document, float]]:
#         """
#         Realiza la búsqueda y devuelve los documentos junto con su puntuación de relevancia en distancia coseno
#         """
#         return self.store.similarity_search_with_relevance_scores(
#             query, k=k, filter=filter
#         )

#     def as_retriever(
#         self,
#         k: int = 5,
#         filter: Optional[Dict[str, Any]] = None,
#         search_type: str = "similarity",
#     ):
#         """
#         Expone una interfaz BaseRetriever compatible con LangChain.
#         """
#         search_kwargs: Dict[str, Any] = {"k": k}
#         if filter:
#             search_kwargs["filter"] = filter
#         return self.store.as_retriever(
#             search_type=search_type, search_kwargs=search_kwargs
#         )


#     # Admin

#     def get_collection_stats(self) -> Dict[str, Any]:
#         """
#         Devuelve estadísticas básicas de la colección
#         """
#         col = self.store._collection
#         return {"name": col.name, "count": col.count()}

#     def delete_collection(self) -> None:
#         """
#         Elimina la colección, es irreversible
#         """
#         self.store.delete_collection()
#         self._store = None
#         logger.warning("ChromaDB collection deleted")
