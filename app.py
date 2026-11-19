from pathlib import Path

import streamlit as st

from orchestrator import ask, clear_cache, get_pipeline

STEP_LABELS = {
    "rewrite": "Understanding the question",
    "cache_lookup": "Checking the semantic cache",
    "retrieve": "Retriever Agent: searching the documents",
    "collect": "Collecting evidence",
    "analyst": "Analyst Agent: checking the evidence",
    "followup": "Analyst asked the Retriever for more evidence",
    "finalize": "Analyst tools: tables, comparison, calculations",
    "answer": "Answer Agent: writing the answer",
    "cache_store": "Saving to cache",
}

st.set_page_config(page_title="Chat with your documents", page_icon="📚")
st.title("Chat with your documents")
st.caption("Upload PDF, DOCX, or TXT files and ask questions grounded in their content.")

store = get_pipeline().store

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process documents", type="primary"):
        successful = 0

        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            incoming = store.incoming_path(safe_name)

            try:
                incoming.write_bytes(uploaded_file.getbuffer())
                st.write(store.ingest(incoming))
                successful += 1
            except Exception as error:
                incoming.unlink(missing_ok=True)
                st.error(f"Could not process {safe_name}: {error}")

        if successful:
            clear_cache()
            st.success(f"Processed {successful} document(s).")

    st.divider()
    names = store.document_names()
    if names:
        st.caption("Indexed documents")
        for name in names:
            st.write(f"• {name}")
    else:
        st.caption("No documents indexed yet.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("Working...", expanded=False) as status:
                def on_step(node: str, update: dict):
                    status.update(label=STEP_LABELS.get(node, node))

                reply = ask(question, history, on_step=on_step)
                status.update(label="Done", state="complete")

            st.markdown(reply.answer)

            if reply.cached:
                st.caption("Answer retrieved from semantic cache.")

            if reply.analysis:
                with st.expander("Analysis details"):
                    st.json(reply.analysis)

            st.session_state.messages.append({"role": "assistant", "content": reply.answer})
        except Exception as error:
            st.error(f"Something went wrong: {error}")
