# Multi-Agent RAG: Chat with Your Documents

Upload PDF, Word, or text files and ask questions by text, voice, or image. Three AI agents work as a team and answer with sources (document name and page).

## How it works

```
Question → Retriever → Analyst → Answer → Response with citations
               ↑          |
               └──────────┘  (asks for more evidence if needed)
```

| Agent | Job |
|-------|-----|
| **Retriever** | Finds the relevant text in the documents. Does not answer. |
| **Analyst** | Reads the evidence, compares, calculates, and asks the Retriever for more if something is missing. |
| **Answer** | Writes the final answer and adds citations. |

An **Orchestrator** (`orchestrator.py`) connects the three agents.

## Features

- **FastAPI backend**: API endpoints for documents, questions, voice, OCR, and reports.
- **Voice questions**: speak into the microphone, the speech is turned into text and answered.
- **Image OCR**: upload an image with text and use it as a question or as a document.
- **PDF reports**: an agent builds a styled PDF report from the answers.
- **Terraform**: infrastructure code to deploy the app on AWS later (see `terraform/README.md`).

## Team tasks

| Part | Owner |
|------|-------|
| Analyst Agent (`analyst/`) | **Dalah** |
| Retriever Agent (`retriever_agent/`) | **Maymuna** |
| Voice query (`voice/`) and Report generator (`report/`) | **Dalah** |
| FastAPI (`api/`) and Image OCR (`ocr/`) | **Maymuna** |
| Answer Agent, Orchestrator, Streamlit app, testing | **Together** |
| Terraform (`terraform/`) | **Together** |

## Project structure

```
app.py                # Streamlit app
api/                  # FastAPI backend
orchestrator.py       # connects the agents
retriever_agent/      # Retriever + its tools
analyst/              # Analyst + its tools
answer/               # Answer Agent + formatters
voice/                # Speech to text
ocr/                  # Image to text
report/               # PDF report generator
terraform/            # Infrastructure as code (AWS)
shared/               # LLM config and shared schemas
```

## Run it

1. Install:
   ```
   pip install -r requirements.txt
   ```
   OCR also needs [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) installed.
2. Copy `.env.example` to `.env` and add your OpenRouter key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
3. Start the API (terminal 1):
   ```
   uvicorn api.main:app --port 8000
   ```
4. Start the app (terminal 2):
   ```
   streamlit run app.py
   ```
5. Upload a document and ask a question.

## Notes

- The free OpenRouter models have a daily request limit. If you see `free-models-per-day`, wait for the reset or add credits.
- The first voice question downloads the Whisper model, so it can be slow once.
- Test with a small document first.
