# LegalAssist: Local RAG for Indian Patent & Tax Law

A privacy-focused legal research tool that uses Retrieval-Augmented Generation (RAG) to query Indian legal documents. By running a local LLM (DeepSeek) via Ollama, it ensures that sensitive legal queries never leave your machine while providing grounded, cited answers.

## Why this exists
Generic LLMs are notorious for "hallucinating" (inventing) case laws and statutes. For legal professionals, this is a dealbreaker. **LegalAssist** bridges this gap by:
- **Restricting Knowledge:** The model only answers based on the PDFs you provide.
- **Verification:** Every response includes direct citations and raw text chunks for manual verification.
- **Zero Cost/API:** Uses local compute via Ollama and ChromaDB.

---

## Tech Stack
- **LLM:** DeepSeek R1 1.5B (via Ollama)
- **Orchestration:** LangChain (LCEL)
- **Vector Store:** ChromaDB
- **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
- **Frontend:** Streamlit

---

## Getting Started

### 1. Requirements
- Python 3.11+
- [Ollama](https://ollama.com) (must be running locally)

### 2. Installation
```bash
# Clone the repo
git clone https://github.com/amaan2801/LegalAssist-using-LLM-and-RAG.git
cd LegalAssist-using-LLM-and-RAG

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the model
ollama pull deepseek-r1:1.5b