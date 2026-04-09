from langchain_community.vectorstores import Chroma
from src.ingestion.embedder import load_vectorstore
from config.settings import TOP_K_RESULTS


def build_retriever(vectorstore: Chroma = None):
    """
    Returns a retriever that finds the top-k most
    relevant legal document chunks for a given query.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )