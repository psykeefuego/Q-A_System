#  Q&A SYSTEM — Intelligent Document Q&A with Memory (RAG)

Welcome to **QA_SYSTEM**, an **AI-powered Retrieval-Augmented Generation (RAG)** system built for **intelligent document question answering** with **long-term memory** and **learning from user feedback**.

This project demonstrates a **production-ready architecture** using **Google Gemini**, vector search (FAISS, easily swappable for Pinecone or Chroma), and Streamlit for an interactive UI.

---
Link: https://implementqnda.streamlit.app/ (The cloud version may be slow due to limited free resources. Local runs are faster and recommended for testing. Must be optimised)
## Features

### Multi-format Document Ingestion
- Supports PDF, DOCX, TXT, HTML, and Markdown files  
- Extracts text and splits content into overlapping chunks to preserve context

### Embeddings and Vector Storage
- Uses Google Gemini Text Embedding (`text-embedding-004`)  
- Stores vector embeddings in a local FAISS index with metadata (source, timestamp)

### Retrieval-Augmented Generation (RAG)
- Performs semantic similarity search to retrieve relevant chunks  
- Supports context-aware multi-turn Q&A with recent conversation memory  
- Uses Google Gemini Pro (`gemini-2.5-pro`) for answer generation

### User Feedback and Evaluation
- Collects explicit feedback: thumbs up/down, corrections, and ratings  
- Stores feedback in `feedback.json` for later analysis  
- Includes basic precision, recall, and F1 evaluation using a SQuAD sample

---

## Deliverables Implemented
- Document upload and processing for multiple formats  
- Vector store setup using FAISS with Google Gemini embeddings  
- Conversational Q&A with session-based memory  
- Feedback logging with corrections and answer ratings  
- Benchmark testing using SQuAD data for quality checks

---

## Tech Stack

| Component       | Technology                             |
|-----------------|-----------------------------------------|
| LLM             | Google Gemini Pro (`gemini-2.5-pro`)    |
| Embeddings      | Google Gemini Text Embedding (`text-embedding-004`) |
| Vector Store    | FAISS                                   |
| Backend         | Python, LangChain                       |
| Frontend        | Streamlit                               |
| Deployment      | Streamlit Cloud                         |

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=YOUR_API_KEY" > .env

# Start Streamlit
streamlit run app.py
```
## Notes
- FAISS is used for local vector storage; it can be swapped with Pinecone or ChromaDB for production use

- Feedback is saved locally in feedback.json

- Logs and performance data can be expanded into an admin dashboard

- This project demonstrates the core RAG pipeline and is ready for further development
