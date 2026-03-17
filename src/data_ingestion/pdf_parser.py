import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class IKEAPDFParser:
    """
    Clase para ingerir catalogos en PDF, extraer su texto y fragmentarlo
    conservando metadatos para la base de datos vectorial.
    """
    def __init__(self, pdf_path: str, source_name: str):
        self.pdf_path = pdf_path
        self.source_name = source_name

        # Estrategia de chunking: caracteres recursivos
        # 1000 caracteres
        # 150 de solapamiento
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """
        Carga el PDF utilizando la estrategia seleccionada.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"No se encontro el PDF en: {self.pdf_path}")

        # Para usar textract, se comentaria la linea de PyMuPDFLoader y se descomentaria la de Amazon.
        loader = PyMuPDFLoader(self.pdf_path)
        # loader = AmazonTextractPDFLoader(self.pdf_path)

        return loader.load()

    def parse_and_chunk(self) -> List[Document]:
        """
        Ejecuta el pipeline: carga -> limpieza -> fragmentacion
        """
        raw_docs = self.load_documents()

        # Enriquecemos metadatos antes de fragmentar
        for doc in raw_docs:
            # Homogeneizamos los metadatos para que no choquen con los del CSV
            doc.metadata = {
                "source": self.source_name,
                "page": doc.metadata.get("page", 0),
                "category": "Inspiracion/Catalogo",
                # Rellenamos los campos matematicos a 0 para no romper el filtro de ChromaDB
                "price": 0.0,
                "width": 0.0,
                "height": 0.0,
                "depth": 0.0
            }

        # Aplicamos el chunking
        chunks = self.text_splitter.split_documents(raw_docs)
        return chunks
