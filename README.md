#  RAG Chatbot — LangGraph + LangChain + Streamlit

A production-deployed, agentic RAG (Retrieval-Augmented Generation) chatbot with multi-turn memory, document Q&A, and web search — built with LangGraph StateGraph and deployed on Streamlit Cloud.

🔗 **Live Demo:** [https://faizananalyst-rag-chatbot.streamlit.app](https://faizananalyst-rag-chatbot.streamlit.app)

---

## 📌 What This Project Does

This chatbot lets you:
- **Chat** with an LLM (Groq `llama-3.3-70b-versatile`) with full conversation memory
- **Upload documents** (PDF, DOCX, TXT) and ask questions directly from their content
- **Switch between conversation threads** — each thread has its own isolated memory and document store
- **Search the web** automatically when the LLM needs current information (DuckDuckGo tool)
- **Persist chat history** across sessions via SQLite — conversations survive app restarts

---

## 🏗️ Architecture

```
User Input
    │
    ▼
Streamlit Frontend (app.py)
    │
    ▼
LangGraph StateGraph
    ├── chat_mod node  ←── RAG context injected here (TF-IDF retrieval)
    │       │
    │       ▼
    │   Groq LLM (llama-3.3-70b-versatile)
    │       │
    │       ▼
    │   Tool call? ──YES──► tools node (DuckDuckGo Search)
    │       │                    │
    │       │◄───────────────────┘
    │       │
    └── END (return response)
    │
    ▼
SQLite Checkpointer (persistent memory per thread_id)
```

### Key Components

| Component | Technology | Purpose |
|---|---|---|
| Agent Graph | LangGraph `StateGraph` | Conditional routing, tool calling, iterative loops |
| LLM | Groq `llama-3.3-70b-versatile` | Fast inference, free API |
| RAG Retrieval | TF-IDF (`scikit-learn`) | Lightweight, no GPU needed |
| Memory | SQLite + `SqliteSaver` | Persistent multi-turn conversation history |
| Web Search | DuckDuckGo via LangChain | Real-time information retrieval |
| Frontend | Streamlit | Interactive chat UI with file upload |
| Deployment | Streamlit Cloud | Free hosting |

---

## 🗄️ Database & Structured Output

### SQLite Conversation Database (`chat_bot_rag.db`)

LangGraph's `SqliteSaver` checkpointer stores all conversation state in SQLite. The database schema contains:

| Table | What It Stores |
|---|---|
| `checkpoints` | Full serialized graph state per `(thread_id, checkpoint_id)` |
| `checkpoint_blobs` | Binary blobs for message history and agent state |
| `checkpoint_writes` | Pending writes between graph steps |

**Querying conversation history directly from SQLite:**

```python
import sqlite3
import json

conn = sqlite3.connect("chat_bot_rag.db")

# List all thread IDs (conversation sessions)
threads = conn.execute("""
    SELECT DISTINCT thread_id FROM checkpoints
""").fetchall()
print([t[0] for t in threads])

# Get full message history for a specific thread
rows = conn.execute("""
    SELECT checkpoint FROM checkpoints
    WHERE thread_id = ?
    ORDER BY checkpoint_id DESC
    LIMIT 1
""", ("YOUR_THREAD_ID",)).fetchone()

state = json.loads(rows[0])
messages = state.get("channel_values", {}).get("messages", [])
for msg in messages:
    print(f"{msg['type']}: {msg['content']}")
```

**Retrieving structured output via LangGraph API:**

```python
from langgraph_rag_backend import rag_work

# Get full state for a thread
state = rag_work.get_state(
    config={"configurable": {"thread_id": "YOUR_THREAD_ID"}}
)

# Access messages
messages = state.values["messages"]
for msg in messages:
    print(f"Role: {type(msg).__name__}")
    print(f"Content: {msg.content}")
    print("---")

# Access thread_id from state
thread_id = state.values.get("thread_id")
```

**List all saved threads:**

```python
from langgraph_rag_backend import list_thread

all_threads = list_thread()
print(f"Total conversations: {len(all_threads)}")
for thread in all_threads:
    print(thread)
```

---

## 📄 RAG Pipeline

Documents are processed using a lightweight TF-IDF pipeline — no GPU, no heavy ML models required:

```
Document Upload (PDF / DOCX / TXT)
        │
        ▼
Document Loader (PyPDF / Docx2txt / TextLoader)
        │
        ▼
RecursiveCharacterTextSplitter
  chunk_size=500, chunk_overlap=80
        │
        ▼
TF-IDF Vectorizer (scikit-learn)
  fit_transform on all chunks
        │
        ▼
In-memory TFIDFStore (per thread_id)
        │
        ▼
Cosine Similarity Search (top-4 chunks)
        │
        ▼
Injected into LLM SystemMessage as context
```

**Why TF-IDF instead of embeddings?**
- No `torch` or `sentence-transformers` dependency → fits within free tier RAM limits
- Instant startup — no model download on cold start
- Sufficient for document Q&A on structured text (reports, papers, notes)

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo 'GROQ_API_KEY=gsk_your_key_here' > .env

# Run
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                      # Streamlit frontend
├── langgraph_rag_backend.py    # LangGraph agent + RAG logic
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml             # Streamlit server config
└── README.md
```

---

## ⚙️ Configuration

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` or Streamlit Secrets | Groq API key for LLM inference |

For Streamlit Cloud deployment, add the key in **App Settings → Secrets**:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph 1.1.8 |
| LLM Provider | Groq (llama-3.3-70b-versatile) |
| LLM Orchestration | LangChain 1.2.15 |
| Document Loading | LangChain Community loaders |
| Text Splitting | LangChain Text Splitters |
| RAG Retrieval | scikit-learn TF-IDF |
| Memory/Persistence | LangGraph SQLite Checkpointer |
| Web Search Tool | DuckDuckGo Search |
| Frontend | Streamlit 1.45.0 |
| PDF Parsing | pypdf |
| DOCX Parsing | docx2txt |

---

## 🔮 Future Improvements

- [ ] Swap TF-IDF for semantic embeddings (when RAM allows)
- [ ] Add FastAPI backend for REST API access to conversation history
- [ ] Multi-user authentication
- [ ] Export conversation history to PDF/CSV
- [ ] Support for image documents (OCR)

---

## 👨‍💻 Author

**Faizan** — Final Year Computer Engineering Student, Mumbai University  
Building AI/ML projects in AgriTech, LegalTech, and GenAI.

---

## 📄 License

MIT License — free to use and modify.
