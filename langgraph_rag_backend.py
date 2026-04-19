# langgraph_rag_backend.py
import sys
import os
import sqlite3
from dotenv import load_dotenv

print("backend.py starting...", flush=True)

load_dotenv()

# ── API Key check ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. Add it in Streamlit Cloud → Settings → Secrets.")

print("GROQ_API_KEY found.", flush=True)

# ── Imports ────────────────────────────────────────────────────
from typing import TypedDict, Annotated, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

print("All imports done.", flush=True)

# ── LLM ───────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY,
)
tools = [DuckDuckGoSearchRun(region="us-en")]
llm_with_tools = llm.bind_tools(tools)

print("LLM ready.", flush=True)

# ── TF-IDF RAG Store ──────────────────────────────────────────
class TFIDFStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self._fitted = False

    def add_texts(self, texts: list[str]):
        self.chunks.extend(texts)
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self._fitted = True

    def search(self, query: str, k: int = 4) -> list[str]:
        if not self._fitted or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_k if scores[i] > 0]

_stores: dict[str, TFIDFStore] = {}

# ── Document helpers ──────────────────────────────────────────
def _load_file(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def ingest_document(thread_id: str, file_path: str) -> str:
    docs = _load_file(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return "⚠️ No text could be extracted from the document."
    texts = [c.page_content for c in chunks]
    if thread_id not in _stores:
        _stores[thread_id] = TFIDFStore()
    _stores[thread_id].add_texts(texts)
    return f"✅ Ingested {len(texts)} chunks from '{os.path.basename(file_path)}'."


def retrieve_context(thread_id: str, query: str, k: int = 4) -> str:
    if thread_id not in _stores:
        return ""
    results = _stores[thread_id].search(query, k=k)
    return "\n\n---\n\n".join(results)


def clear_documents(thread_id: str):
    _stores.pop(thread_id, None)


# ── State schema ──────────────────────────────────────────────
class chatbot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: Optional[str]


# ── Chat node ─────────────────────────────────────────────────
def chat_mod(state: chatbot):
    msgs = state["messages"]
    thread_id = state.get("thread_id", "")

    query = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            query = m.content
            break

    context = retrieve_context(thread_id, query) if thread_id else ""

    if context:
        rag_block = (
            "The user has uploaded document(s). Use the excerpts below to answer "
            "if relevant. If not relevant, answer from your own knowledge.\n\n"
            f"=== Document Excerpts ===\n{context}\n========================"
        )
    else:
        rag_block = (
            "No documents uploaded. Answer from your own knowledge "
            "or use the search tool if needed."
        )

    system_msg = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n"
            f"{rag_block}\n\n"
            "Only call tools when absolutely necessary. "
            "For normal questions, respond directly. "
            "If tool input is unclear, DO NOT call the tool."
        )
    )

    res = llm_with_tools.invoke([system_msg] + msgs)
    return {"messages": [res]}


# ── Graph ─────────────────────────────────────────────────────
tool_node = ToolNode(tools)

DB_PATH = "/data/chat_bot_rag.db" if os.path.exists("/data") else "chat_bot_rag.db"
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(chatbot)
graph.add_node("chat_mod", chat_mod)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_mod")


def route_tools(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


graph.add_conditional_edges("chat_mod", route_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "chat_mod")

rag_work = graph.compile(checkpointer=checkpointer)

print("Graph compiled. Backend ready.", flush=True)


def list_thread():
    seen = set()
    for item in checkpointer.list(None):
        seen.add(item.config["configurable"]["thread_id"])
    return list(seen)
