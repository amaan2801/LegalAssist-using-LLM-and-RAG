from dataclasses import dataclass, field
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.retrieval.retriever import build_retriever
from src.chain.prompt import get_legal_prompt
from config.settings import LLM_MODEL, LLM_TEMPERATURE


@dataclass
class LegalAnswer:
    question: str
    answer: str
    source_documents: list[Document] = field(default_factory=list)

    @property
    def unique_sources(self) -> list[str]:
        seen = set()
        sources = []
        for doc in self.source_documents:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain():
    """
    Builds the RAG pipeline using LCEL:
    retriever → prompt → LLM → output parser
    """
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    retriever = build_retriever()
    prompt = get_legal_prompt()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(chain, question: str) -> LegalAnswer:
    """
    Runs a question through the RAG chain.
    Retrieves source documents separately for citation display.
    """
    retriever = build_retriever()
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    return LegalAnswer(
        question=question,
        answer=answer,
        source_documents=source_docs,
    )