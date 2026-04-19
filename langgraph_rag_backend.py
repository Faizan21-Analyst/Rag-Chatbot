from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from typing import TypedDict, Annotated, Optional
import sqlite3
import os
import tempfile



llm = ChatGroq(
    model="openai/gpt-oss-120b",groq_api_key=os.getenv("GROQ_API_KEY", "")
)

tools = [DuckDuckGoSearchRun(region="us-en")]
llm_with_tools = llm.bind_tools(tools)


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

_vector_stores: dict[str, FAISS] = {}

def _load_file(file_path: str):
    """Load a file into LangChain Documents based on extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def ingest_document(thread_id: str, file_path: str) -> str:
    """
    Chunk a document, embed it, and store in the thread's vector store.
    Returns a short status message.
    """
    docs = _load_file(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        return "No text could be extracted from the document."

    if thread_id in _vector_stores:
        # Append to existing store for this thread
        _vector_stores[thread_id].add_documents(chunks)
    else:
        _vector_stores[thread_id] = FAISS.from_documents(chunks, embeddings)

    return f"Ingested {len(chunks)} chunks from '{os.path.basename(file_path)}'."


def retrieve_context(thread_id: str, query: str, k: int = 4) -> str:
    """Return the top-k relevant chunks as a single string, or empty string."""
    if thread_id not in _vector_stores:
        return ""
    results = _vector_stores[thread_id].similarity_search(query, k=k)
    if not results:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in results)


def clear_documents(thread_id: str):
    """Remove the vector store for a thread (useful on 'New Chat')."""
    _vector_stores.pop(thread_id, None)



class chatbot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: Optional[str]          



def chat_mod(state: chatbot):
    msgs = state["messages"]
    thread_id = state.get("thread_id", "")

    
    query = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            query = m.content
            break

    # Pull relevant doc chunks (empty string if no docs uploaded)
    context = retrieve_context(thread_id, query) if thread_id else ""

    if context:
        rag_block = (
            "The user has uploaded document(s). Use the excerpts below to answer "
            "if they are relevant. If not relevant, answer from your own knowledge.\n\n"
            f"=== Document Excerpts ===\n{context}\n========================"
        )
    else:
        rag_block = (
            "No documents have been uploaded for this session. "
            "Answer from your own knowledge or use the search tool if needed."
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





tool_node = ToolNode(tools)
import os
os.makedirs("/data", exist_ok=True)
conn = sqlite3.connect(database="/data/chat_bot_rag.db", check_same_thread=False)
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





def list_thread():
    seen = set()
    for item in checkpointer.list(None):
        seen.add(item.config["configurable"]["thread_id"])
    return list(seen)
