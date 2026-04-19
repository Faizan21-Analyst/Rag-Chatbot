import sys
print(f"Python version: {sys.version}", flush=True)
print("app.py starting...", flush=True)

import streamlit as st
import traceback
import sys

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

# ── Try importing the backend and catch ANY error ──────────────
try:
    from langgraph_rag_backend import rag_work, list_thread, ingest_document, clear_documents
    BACKEND_OK = True
    BACKEND_ERROR = None
except Exception as e:
    BACKEND_OK = False
    BACKEND_ERROR = traceback.format_exc()

# ── If backend failed, show the error on screen ───────────────
if not BACKEND_OK:
    st.error("❌ Backend failed to load. See error below:")
    st.code(BACKEND_ERROR, language="python")
    st.stop()

# ── Normal imports only reached if backend loaded fine ────────
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid
import tempfile
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["uploaded_files_info"] = []
    add_thread(thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = rag_work.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", []) if state else []


def get_ai_response(user_input: str, thread_id: str) -> str:
    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    result = rag_work.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "thread_id": thread_id,
        },
        config=CONFIG,
    )
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


# ─────────────────────────────────────────────────────────────
# Session-state bootstrap
# ─────────────────────────────────────────────────────────────
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list_thread()

if "uploaded_files_info" not in st.session_state:
    st.session_state["uploaded_files_info"] = []

add_thread(st.session_state["thread_id"])

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Chatbot")
    st.caption("Upload documents · Chat · Switch threads")
    st.divider()

    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx", "doc"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            key = (st.session_state["thread_id"], uf.name)
            if key not in st.session_state["uploaded_files_info"]:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uf.name)[-1],
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                status = ingest_document(st.session_state["thread_id"], tmp_path)
                os.unlink(tmp_path)
                st.session_state["uploaded_files_info"].append(key)
                st.success(status)

    ingested = [
        fname
        for (tid, fname) in st.session_state["uploaded_files_info"]
        if tid == st.session_state["thread_id"]
    ]
    if ingested:
        st.caption("**Ingested for this thread:**")
        for f in ingested:
            st.caption(f"• {f}")

    st.divider()
    st.button("➕ New Chat", on_click=reset_chat, use_container_width=True)

    st.subheader("💬 Conversations")
    for thread_id in st.session_state["chat_threads"]:
        state = rag_work.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        msgs = state.values.get("messages", []) if state else []
        label = msgs[-1].content[:28] + "…" if msgs else "New Chat"

        if st.sidebar.button(f"🗂 {label}", key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            raw_msgs = load_conversation(thread_id)
            history = []
            for m in raw_msgs:
                role = "user" if isinstance(m, HumanMessage) else "AI"
                if m.content:
                    history.append({"role": role, "content": m.content})
            st.session_state["message_history"] = history
            st.rerun()

    st.divider()
    st.caption(f"Thread: `{str(st.session_state['thread_id'])[:8]}…`")

# ─────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────
st.header("🤖 RAG Chat Assistant")

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything — or ask about your documents…")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        with st.spinner("Thinking…"):
            ai_reply = get_ai_response(user_input, st.session_state["thread_id"])
        st.markdown(ai_reply)

    st.session_state["message_history"].append(
        {"role": "AI", "content": ai_reply}
    )
