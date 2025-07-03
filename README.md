# 📄 Q&A SYSTEM — Intelligent Document Q&A with Memory (RAG)

Welcome to **QA_SYSTEM**, an **AI-powered Retrieval-Augmented Generation (RAG)** system built for **intelligent document question answering** with **long-term memory** and **learning from user feedback**.

This project demonstrates a **production-ready architecture** using **Google Gemini**, vector search (FAISS, easily swappable for Pinecone or Chroma), and Streamlit for an interactive UI.
link: https://implementqnda.streamlit.app/ (should be optimised, works better on local system)
---

## 🚀 Features

✅ **Multi-format Document Ingestion**  
- Supports **PDF**, **DOCX**, **TXT**, **HTML**, and **Markdown** files  
- Semantic chunking with overlapping windows for context preservation

✅ **Embedding & Vector Store**  
- Uses **Google Gemini Embeddings** (`text-embedding-004`)  
- Hierarchical chunk-level embeddings with metadata: source, timestamp  
- Stored in a local **FAISS index** (can migrate to ChromaDB, Pinecone, or Weaviate)

✅ **Retrieval-Augmented Generation (RAG)**  
- Hybrid search: semantic + keyword matching  
- Context-aware query expansion with conversation history  
- Multi-turn conversational support with dynamic context window

✅ **Memory & Learning**  
- **Short-term memory**: last 20 turns in session  
- **Long-term memory**: feedback, corrections, episodic logs  
- **User feedback system**: thumbs up/down, corrections, ratings  
- Answer quality auto-evaluation using **precision, recall, F1**

✅ **Admin Dashboard & Visualization**  
- View performance metrics, logs, and user feedback  
- Visualize memory growth and learning impact over time

✅ **Demo-Ready**  
- Easily processes **10+ diverse documents**  
- Runs **SQuAD 2.0 samples** for benchmark testing  
- Can be extended to COQA & Natural Questions datasets

---

## 🧩 Deliverables

### ✅ **Working System Components**
- 📂 **Document Upload & Processing** — Upload any supported file and generate embeddings automatically
- 💬 **Q&A Chat Interface** — Interactive question answering with conversation history
- 📊 **Admin Dashboard** — Displays feedback, logs, and learning metrics
- 🧠 **Memory Visualization** — Shows how system knowledge grows over time

---

### ✅ **Demonstration Requirements**
- Process 10+ different files
- Support multi-turn queries in a single session
- Accept and learn from explicit corrections
- Show measurable answer improvements (tracked with F1, P, R scores)

---

### ✅ **Technical Documentation**
- 📌 **System Architecture Diagram** — (add your `architecture.png` here)
- 🗂️ **API Endpoints** —  
  - `/upload`: Upload and process files  
  - `/ask`: Query the system with context  
  - `/feedback`: Submit feedback/corrections  
- 🧠 **Memory Schema** — Documents, session chat logs, Q&A pairs, feedback
- ⚡ **Performance Benchmarks** — Aims for <2s response time, F1 > 0.7, handles 100+ docs

---

## ⚙️ Tech Stack

- **LLM:** Google Gemini Pro (`gemini-2.5-pro`)
- **Embeddings:** Google Gemini Text Embedding (`text-embedding-004`)
- **Vector Store:** FAISS (swap to Pinecone/Chroma for production)
- **Backend:** Python, LangChain
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 🚦 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=YOUR_API_KEY" > .env

# Start Streamlit
streamlit run app.py
