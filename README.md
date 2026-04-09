# LegalAssist — RAG-Based Legal Assistant

A local AI assistant that answers questions about Indian patent and tax law.
Built for my B.Tech final year project at USAR, GGSIPU.

The core idea is simple — instead of asking an LLM to answer from memory
(which leads to hallucinated case laws and fake statutes), this system first
retrieves the actual relevant sections from real legal documents, then uses
the LLM to explain only what it finds. Every answer comes with a source.

---

## Why I built this

Legal research is expensive and slow. A lawyer charges thousands per hour.
A general chatbot confidently invents laws that don't exist. This project
tries to fix the second problem — making AI actually reliable for legal queries
by grounding it in real documents.

---

## Demo

![Demo](assets_working_ss.png)

---

## How it works

When you ask a question:

1. The question is converted into a vector (a list of numbers representing meaning)
2. ChromaDB searches for the most similar chunks from the legal PDFs
3. Those chunks are handed to DeepSeek 1.5B with strict instructions — answer only from this
4. The answer appears in the UI along with the source document it came from

This approach is called RAG — Retrieval Augmented Generation.

---

## Tech used

- **DeepSeek R1 1.5B** — the LLM, runs locally via Ollama (no API key needed)
- **ChromaDB** — stores and searches document embeddings
- **LangChain** — connects retrieval and generation into a pipeline
- **HuggingFace sentence-transformers** — converts text to vectors
- **Streamlit** — the web interface
- **Python 3.11**

---

---

## Setup

You need Python 3.11 and [Ollama](https://ollama.com) installed.

```bash
# clone the repo
git clone https://github.com/amaan2801/LegalAssist-using-LLM-and-RAG.git
cd LegalAssist-using-LLM-and-RAG

# create and activate virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# pull the model (downloads ~1GB)
ollama pull deepseek-r1:1.5b

# add your legal PDFs to data/raw/
# then build the vector database
python ingest.py

# launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## What the ingestion pipeline does

Running `ingest.py` does three things:

- Loads every PDF from `data/raw/`
- Splits the text into 800-token chunks with 100-token overlap (so context
  is not lost at chunk boundaries)
- Converts each chunk into a 384-dimensional embedding and stores it in ChromaDB

This only needs to run once, or again whenever you add new documents.

---

## Limitations

- Answers are only as good as the documents you provide
- DeepSeek 1.5B is a small model — responses can be slow on CPU (20-40 seconds)
- Currently tested on Income Tax Act and Patent Act PDFs only

---

## Future work

- Upload PDFs directly through the UI instead of dropping them in a folder
- Add RAGAS evaluation for automated answer quality scoring
- Support multiple documents with a document selector
- Faster inference with a GPU or a quantized model

---
