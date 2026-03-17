"""
Script para indexar documentos desde un directorio en ChromaDB.
Lee archivos CSV y PDF, los procesa y los indexa utilizando embeddings de AWS.
Uso:
    python scripts/index_documents.py --dir data/raw --clear
Opciones:
    --dir: Directorio raíz de los documentos (default: data/raw)
    --clear: Limpiar la colección existente antes de indexar
    --no-recursive: Solo indexar archivos en el directorio raíz (sin subcarpetas)
    """
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import loader

# Añadir el directorio raíz del proyecto al sys.path para permitir imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import Settings
from src.embeddings.aws_embeddings import get_embeddings
from src.data_ingestion.csv_parser import IKEACatalogParser
from src.data_ingestion.pdf_parser import IKEAPDFParser
from src.vectorstore.chroma_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index documents from a directory into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", # Directorio raíz de los documentos (data/raw/)
        type=str,
        default="data/raw",
        required=True,
        metavar="PATH",
        help="Directory containing the documents to index",
    )
    parser.add_argument(
        "--clear", # Opción para limpiar la colección existente antes de indexar
        action="store_true",
        help="⚠️  Clear the existing ChromaDB collection before indexing",
    )
    parser.add_argument(
        "--no-recursive", # Opción para no procesar subdirectorios
        action="store_true",
        help="Only index files in the top-level directory (skip sub-folders)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()

    logger.info("Initialising embeddings and vector store…")

    # Obtener las embeddings utilizando la función get_embeddings() con la configuración proporcionada
    embeddings = get_embeddings(settings)

    # Instancia de ChromaVectorStore con las embeddings y configuración
    vector_store = ChromaVectorStore(embeddings, settings)

    if args.clear:
        logger.warning("Clearing existing collection…")
        vector_store.delete_collection()

    base_dir = Path(args.dir)

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        sys.exit(1)

    csv_dir = base_dir / "csv"
    pdf_dir = base_dir / "pdf"

    all_documents = []

    # Ingestión de CSVs
    if csv_dir.exists():
        logger.info(f"Processing CSV directory: {csv_dir}")

        for csv_file in csv_dir.glob("*.csv"):
            try:
                logger.info(f"Parsing CSV: {csv_file.name}")

                # Instancia de IKEACatalogParser con la ruta del archivo CSV
                parser = IKEACatalogParser(str(csv_file))

                # Llamada al método parse_to_documents() para obtener los documentos a partir del CSV
                docs = parser.parse_to_documents()
                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"Error in CSV {csv_file.name}: {e}")
    else:
        logger.warning("CSV directory not found, skipping...")

    # Ingestión de PDFs
    if pdf_dir.exists():
        logger.info(f"Processing PDF directory: {pdf_dir}")

        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                logger.info(f"Parsing PDF: {pdf_file.name}")

                # Instancia de IKEAPDFParser con la ruta del archivo PDF y el nombre de la fuente
                parser = IKEAPDFParser( 
                    pdf_path=str(pdf_file),
                    source_name=pdf_file.name
                )

                # Llamada al método parse_and_chunk() para obtener los documentos a partir del PDF
                docs = parser.parse_and_chunk()
                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"Error in PDF {pdf_file.name}: {e}")
    else:
        logger.warning("PDF directory not found, skipping...")

    # Indexación en ChromaDB
    if not all_documents:
        logger.warning("No documents found to index.")
        return

    logger.info(f"Indexing {len(all_documents)} documents into ChromaDB…")
    
    # Indexar todos los documentos obtenidos en la colección de ChromaDB utilizando el método add_documents() del vector_store
    vector_store.add_documents(all_documents)

    # Obtener estadísticas de la colección después de la indexación utilizando el método get_collection_stats() del vector_store
    stats = vector_store.get_collection_stats()

    logger.success(
        f"Done! Collection '{stats['name']}' contains {stats['count']} documents."
    )


if __name__ == "__main__":
    main()