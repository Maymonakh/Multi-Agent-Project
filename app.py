import os
from pathlib import Path
import streamlit as st
import retriever_bridge
from orchestrator import ask, clear_cache

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Chat with your documents", page_icon="📚")
st.title("Chat with your documents")
st.caption("Upload PDF, DOCX, or TXT files and ask questions grounded in their content.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process documents", type="primary"):
        successful = 0

        try:
            retriever_bridge.initialize()
        except Exception as error:
            st.error(f"Could not initialize the document store: {error}")
            st.stop()

        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            destination = DOCS_DIR / safe_name

            try:
                destination.write_bytes(uploaded_file.getbuffer())
                st.write(retriever_bridge.ingest(str(destination)))
                successful += 1
            except Exception as error:
                st.error(f"Could not process {safe_name}: {error}")

        if successful:
            clear_cache()
            st.success(f"Processed {successful} document(s).")

    st.divider()
    st.caption("Documents are stored in the local documents folder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            try:
                reply = ask(question, st.session_state.messages)
                st.markdown(reply.answer)

                if reply.cached:
                    st.caption("Answer retrieved from semantic cache.")

                if reply.analysis:
                    with st.expander("Analysis details"):
                        st.json(reply.analysis)

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.answer}
                )
            except Exception as error:
                st.error(f"Something went wrong: {error}")
