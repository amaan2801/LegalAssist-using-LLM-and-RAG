import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import EMBED_MODEL, VECTOR_DIR

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    logger.info(f"Vectorstore saved to {VECTOR_DIR} with {len(chunks)} chunks")
    return vectorstore


def load_vectorstore() -> Chroma:
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            "Vectorstore not found. Run `python ingest.py` first."
        )

    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )