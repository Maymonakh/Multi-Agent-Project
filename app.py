from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui_client import ApiClient, ApiClientError

load_dotenv()

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]

st.set_page_config(page_title="Chat with your documents", page_icon="📚")
st.title("Chat with your documents")
st.caption("Upload PDF, DOCX, TXT or images (OCR) and ask questions grounded in their content.")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


client = get_client()


def show_reply(reply: dict):
    st.markdown(reply["answer"])
    if reply.get("cached"):
        st.caption("Answer retrieved from semantic cache.")
    if reply.get("ocr_text"):
        with st.expander("Text read from the image"):
            st.text(reply["ocr_text"])
    if reply.get("analysis"):
        with st.expander("Analysis details"):
            st.json(reply["analysis"])
    st.caption(f"{reply['elapsed_ms'] / 1000:.1f}s")


with st.sidebar:
    st.header("Documents")

    try:
        health = client.health()
        st.caption(f"API connected · OCR: {health['ocr_engine']}")
    except ApiClientError as error:
        st.error(error.detail)
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or an image",
        type=["pdf", "docx", "txt"] + IMAGE_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process documents", type="primary"):
        successful = 0
        for uploaded_file in uploaded_files:
            name = Path(uploaded_file.name).name
            is_image = Path(name).suffix.lower().lstrip(".") in IMAGE_TYPES
            try:
                if is_image:
                    result = client.upload_image(name, uploaded_file.getvalue())
                    with st.expander(f"Text read from {name}"):
                        st.text(result["ocr_text"])
                else:
                    result = client.upload_document(name, uploaded_file.getvalue())
                st.write(result["message"])
                successful += 1
            except ApiClientError as error:
                st.error(f"Could not process {name}: {error.detail}")
        if successful:
            st.success(f"Processed {successful} document(s).")

    st.divider()
    try:
        names = client.documents()
    except ApiClientError as error:
        names = []
        st.error(error.detail)

    if names:
        st.caption("Indexed documents")
        for name in names:
            left, right = st.columns([5, 1])
            left.write(f"• {name}")
            if right.button("🗑", key=f"delete_{name}", help=f"Delete {name}"):
                try:
                    client.delete_document(name)
                    st.rerun()
                except ApiClientError as error:
                    st.error(error.detail)
    else:
        st.caption("No documents indexed yet.")

    if st.button("Clear answer cache"):
        try:
            client.clear_cache()
            st.success("Cache cleared.")
        except ApiClientError as error:
            st.error(error.detail)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def ask_and_show(display_text: str, call):
    """`call(history)` performs the API request; history = previous turns only."""
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": display_text})

    with st.chat_message("user"):
        st.markdown(display_text)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and preparing an answer..."):
                reply = call(history)
            show_reply(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply["answer"]})
        except ApiClientError as error:
            st.error(error.detail)


with st.expander("Ask with an image"):
    image_file = st.file_uploader("Image with text", type=IMAGE_TYPES, key="ask_image")
    image_question = st.text_input("Question (optional: empty = the text in the image is the question)")
    if image_file and st.button("Ask about this image"):
        ask_and_show(
            f"📷 {image_question or image_file.name}",
            lambda history: client.ask_image(image_file.name, image_file.getvalue(), image_question, history),
        )

question = st.chat_input("Ask a question about your documents")
if question:
    ask_and_show(question, lambda history: client.ask(question, history))
