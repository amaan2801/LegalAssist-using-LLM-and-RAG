import streamlit as st
import tempfile
import shutil
from pathlib import Path
from src.chain.qa_chain import build_qa_chain, ask
from src.ingestion.loader import load_legal_documents
from src.ingestion.splitter import split_into_chunks
from src.ingestion.embedder import build_vectorstore

st.set_page_config(
    page_title="LegalAssist",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LegalAssist")
    st.caption("Powered by DeepSeek 1.5B + RAG")
    st.divider()

    # PDF Upload Section
    st.markdown("### Upload Legal Documents")
    uploaded_files = st.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload legal PDFs to add them to the knowledge base",
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                raw_dir = Path("data/raw")
                raw_dir.mkdir(parents=True, exist_ok=True)

                for file in uploaded_files:
                    dest = raw_dir / file.name
                    with open(dest, "wb") as f:
                        f.write(file.read())

                docs = load_legal_documents()
                chunks = split_into_chunks(docs)
                build_vectorstore(chunks)

                st.cache_resource.clear()

            st.success(f"Processed {len(uploaded_files)} document(s). Knowledge base updated.")

    st.divider()

    # Mode selector
    st.markdown("### Mode")
    mode = st.radio(
        label="Select what you want to do",
        options=["Q&A — Ask a question", "Draft — Generate a legal document"],
        label_visibility="collapsed",
    )

    st.divider()

    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Tech Stack**")
    st.code(
        "LangChain · ChromaDB\n"
        "DeepSeek 1.5B · Ollama\n"
        "HuggingFace · Streamlit",
        language=None,
    )
    st.caption("All processing is local. No data leaves your machine.")


# ── Load Chain ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading legal knowledge base...")
def load_chain():
    return build_qa_chain()


chain = load_chain()

# ── Chat History Setup ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main Area ─────────────────────────────────────────────────
if "Q&A" in mode:
    st.title("Legal Q&A")
    st.caption("Ask questions about your uploaded legal documents. Every answer cites its source.")
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources:**")
                for src in msg["sources"]:
                    st.markdown(f"- `{Path(src).name}`")

    # Chat input
    question = st.chat_input("Ask a legal question...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching legal database..."):
                result = ask(chain, question)

            st.write(result.answer)

            if result.unique_sources:
                st.markdown("**Sources:**")
                for src in result.unique_sources:
                    st.markdown(f"- `{Path(src).name}`")

            with st.expander("View retrieved chunks"):
                for i, doc in enumerate(result.source_documents, 1):
                    st.markdown(f"**Chunk {i}** — `{Path(doc.metadata.get('source', 'unknown')).name}`")
                    st.text(doc.page_content[:400] + "...")
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources": result.unique_sources,
        })

else:
    # ── Legal Document Drafting Mode ──────────────────────────
    st.title("Legal Document Drafting")
    st.caption("Describe what you need and the assistant will draft a structured legal document.")
    st.divider()

    doc_type = st.selectbox(
        "Document type",
        options=[
            "Legal Notice",
            "Non-Disclosure Agreement (NDA)",
            "Employment Contract",
            "Cease and Desist Letter",
            "Partnership Agreement",
            "Patent Filing Summary",
        ],
    )

    context = st.text_area(
        label="Describe the situation",
        placeholder=(
            "e.g. I am Amaan Ahmed, residing at XYZ address. I need to send a "
            "legal notice to ABC Company for non-payment of dues worth ₹50,000 "
            "for services rendered in January 2025."
        ),
        height=160,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        draft = st.button("Draft Document", type="primary", use_container_width=True)
    with col2:
        st.caption("Takes 30–60 seconds on CPU.")

    if draft:
        if not context.strip():
            st.warning("Please describe the situation first.")
        else:
            draft_question = (
                f"Draft a professional {doc_type} based on the following situation. "
                f"Format it properly with all standard legal sections, clauses, and signatures. "
                f"Situation: {context}"
            )

            with st.spinner(f"Drafting your {doc_type}..."):
                result = ask(chain, draft_question)

            st.divider()
            st.markdown(f"### {doc_type}")
            st.write(result.answer)

            st.download_button(
                label="Download as .txt",
                data=result.answer,
                file_name=f"{doc_type.replace(' ', '_').lower()}.txt",
                mime="text/plain",
            )
