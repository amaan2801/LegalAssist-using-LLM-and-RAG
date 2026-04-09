import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


def load_legal_documents() -> list[Document]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    logger.info(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}")

    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} pages total")
    return documents