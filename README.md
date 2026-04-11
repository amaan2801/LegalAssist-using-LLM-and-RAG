# LegalAssist — AI Legal Assistant Using RAG

A local AI assistant that answers questions about Indian patent and tax law,
drafts legal documents, and maintains conversation history — all running
privately on your machine without any API keys or cloud services.

Built as my B.Tech final year project at USAR, GGSIPU Delhi.

---

## The Problem

Legal research is expensive and slow. A lawyer charges thousands per hour.
General-purpose AI chatbots confidently make up case laws and statutes that
don't exist — this is called hallucination, and in legal contexts it is
genuinely dangerous.

This project addresses that by forcing the AI to answer only from verified
legal documents you provide. If the answer isn't in the documents, it says so.

---

## What It Can Do

**1. Legal Q&A with Citations**
Ask any question about the documents in the knowledge base. The system
retrieves the most relevant sections and answers strictly from them. Every
response shows which document and chunk it came from.

**2. Upload Documents Through the UI**
Drag and drop any legal PDF directly into the sidebar. The system processes
it, builds embeddings, and updates the knowledge base instantly — no
command line needed.

**3. Chat History**
The conversation stays visible as you ask follow-up questions, just like
a real chat interface. You can clear the history anytime from the sidebar.

**4. Legal Document Drafting**
Switch to Draft mode, pick a document type (legal notice, NDA, employment
contract, cease and desist, etc.), describe your situation, and the assistant
generates a complete formatted legal document. You can download it as a .txt
file.

---

## How RAG Works in This Project
Your PDF
↓
PDF Loader → splits into 800-token chunks → converted to embeddings
↓
ChromaDB stores all embeddings locally on disk
↓
You ask a question
↓
Your question → converted to embedding → similarity search in ChromaDB
↓
Top 4 most relevant chunks retrieved
↓
Chunks + your question → passed to DeepSeek 1.5B with strict instructions
↓
"Answer ONLY from the context provided. Cite the source."
↓
Answer appears in the UI with source document name

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | DeepSeek R1 1.5B |
| Local LLM runner | Ollama |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| RAG framework | LangChain (LCEL pipeline) |
| Frontend | Streamlit |
| Language | Python 3.11 |

---

---

## Setup

### Requirements
- Python 3.11
- [Ollama](https://ollama.com) installed

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amaan2801/LegalAssist-using-LLM-and-RAG.git
cd LegalAssist-using-LLM-and-RAG

# 2. Create and activate virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the model (downloads around 1 GB)
ollama pull deepseek-r1:1.5b

# 5. Add legal PDFs to data/raw/ folder
# then build the vector database
python ingest.py

# 6. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Every time you restart your computer

```bash
cd C:\Users\AMAAN ANEES\Desktop\LegalAssist-using-LLM-and-RAG
venv\Scripts\activate
streamlit run app.py
```

Make sure the Ollama app is running in your system tray before launching.

---

## Features Walkthrough

### Uploading a document
1. Open the sidebar
2. Click "Upload Legal Documents"
3. Select one or more PDF files
4. Click "Process Documents"
5. Wait for the success message — the knowledge base is now updated

### Asking a question
1. Make sure mode is set to "Q&A" in the sidebar
2. Type your question in the chat box at the bottom
3. The answer appears with source citations
4. Click "View retrieved chunks" to see exactly what text was used

### Drafting a legal document
1. Switch mode to "Draft" in the sidebar
2. Select the document type from the dropdown
3. Describe your situation in plain English
4. Click "Draft Document"
5. Download the result as a .txt file

---

## Comparison With General LLMs

| Aspect | ChatGPT / General LLM | LegalAssist |
|---|---|---|
| Data privacy | Sent to cloud servers | Fully local, nothing leaves your machine |
| Hallucination | Invents laws and case numbers | Answers only from verified documents |
| Source tracing | No citations | Every answer shows source document |
| Cost | API subscription required | Free, runs on your hardware |
| Internet required | Yes | No (after initial model download) |

---

## Known Limitations

- Responses take 20–60 seconds on CPU. A GPU would make it much faster.
- Answer quality depends on the documents provided. Garbage in, garbage out.
- DeepSeek 1.5B is a small model. For complex legal analysis a larger model
  would give better results.
- Currently no support for scanned PDFs (image-based PDFs won't extract text).

---

## Future Improvements

- RAGAS evaluation for automated answer quality measurement
- Support for scanned PDFs using OCR
- Multi-document comparison (ask questions across two contracts at once)
- User authentication for multi-user deployment
- Faster inference using a quantized model
