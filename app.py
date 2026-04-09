import streamlit as st
from src.chain.qa_chain import build_qa_chain, ask

st.set_page_config(
    page_title="Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal Assistant")
    st.caption("Powered by DeepSeek 1.5B + RAG")
    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        "1. Your question is converted into a vector\n"
        "2. Most relevant legal sections are retrieved\n"
        "3. DeepSeek answers using *only* those sections\n"
        "4. Source documents are shown for verification"
    )
    st.divider()

    st.markdown("**Tech Stack**")
    st.code(
        "LangChain · ChromaDB\n"
        "DeepSeek 1.5B · Ollama\n"
        "HuggingFace · Streamlit",
        language=None
    )
    st.divider()
    st.caption("All processing is local. No data leaves your machine.")


# ── Main ─────────────────────────────────────────────
st.title("Patent Law Legal Assistant")
st.caption(
    "Ask questions grounded in verified legal documents. "
    "Every answer cites its source."
)
st.divider()


@st.cache_resource(show_spinner="Loading legal knowledge base...")
def load_chain():
    return build_qa_chain()


chain = load_chain()

question = st.text_area(
    label="Enter your legal question",
    placeholder="e.g. What deductions are allowed under Section 35 of the Income Tax Act?",
    height=120,
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    st.caption("Response may take 20–40 seconds on CPU.")

if submit:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Searching legal database and generating answer..."):
            result = ask(chain, question)

        st.divider()

        st.markdown("### Answer")
        st.write(result.answer)

        st.markdown("### Source Documents")
        if result.unique_sources:
            for src in result.unique_sources:
                st.markdown(f"- `{src}`")
        else:
            st.caption("No source metadata available.")

        with st.expander("View retrieved chunks"):
            for i, doc in enumerate(result.source_documents, 1):
                st.markdown(
                    f"**Chunk {i}** — "
                    f"`{doc.metadata.get('source', 'unknown')}`"
                )
                st.text(doc.page_content[:500] + "...")
                st.divider()