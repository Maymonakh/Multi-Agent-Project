import os
import streamlit as st
import retriever_bridge
from orchestrator import ask, clear_cache

DOCS_DIR = "documents"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="Chat with your documents")
st.title("Chat with your documents")

if retriever_bridge.USING_FAKE:
    st.warning("Using FAKE retriever data. Person A's retriever is not connected yet.")

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)
    if files and st.button("Process documents"):
        for f in files:
            path = os.path.join(DOCS_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            st.write(retriever_bridge.ingest(path))
        clear_cache()   # old answers may be wrong for new documents

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask(question, st.session_state.messages)
        st.markdown(reply.answer)
        if reply.cached:
            st.caption("From semantic cache")
        with st.expander("Analysis details"):
            st.json(reply.analysis)
    st.session_state.messages.append({"role": "assistant", "content": reply.answer})
