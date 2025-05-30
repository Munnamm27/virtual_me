# 🧠 Virtual ME – Chat With Virtual Me

**Virtual Me** is a personal RAG-based assistant that enables you (or recruiters) to query about me, just like chatting with me, Munna.

---

## 🎯 Objective

Create an intelligent, interactive system to query my personal documents using Retrieval-Augmented Generation (RAG). The goal is to provide smart, contextual answers with source references via a simple Streamlit UI and API.

---

## 🧩 Features

- 🔍 **RAG-based QA**: Ask any question and get relevant answers grounded in source documents  
- 🧠 **Virtual Munna**: Acts like a chat interface to talk to "Munna" with full context  
- 📚 **Document Retrieval**: Vector-based document retrieval using FAISS or Chroma  
- 🔗 **Source Highlighting**: All answers include citations to the original documents  
- 🌐 **API + UI**: Access via REST API (FastAPI) or interactive frontend (Streamlit)

---

## 🧰 Tech Stack

| Component    | Tech Used                               |
|--------------|-----------------------------------------|
| Embeddings   | OpenAI  |
| RAG Framework| LangChain                  |
| Vector DB    | FAISS                        |
| Backend API  | FastAPI                                 |
| Frontend     | Streamlit                               |
| Language     | Python                                  |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/virtual-munna.git
cd virtual-munna
