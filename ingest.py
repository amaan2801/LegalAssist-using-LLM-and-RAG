import logging
from src.ingestion.loader import load_legal_documents
from src.ingestion.splitter import split_into_chunks
from src.ingestion.embedder import build_vectorstore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ingestion pipeline...")
    documents = load_legal_documents()
    chunks = split_into_chunks(documents)
    build_vectorstore(chunks)
    logger.info("Ingestion complete. Vectorstore is ready.")


if __name__ == "__main__":
    main()