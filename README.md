# Multi-Agent RAG: Chat with Your Documents

Upload PDF, Word, or text files and ask questions. Three AI agents work as a team and answer with sources (document name and page).

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

## Team tasks

| Part | Owner |
|------|-------|
| Analyst Agent (`analyst/`) | **Dalah** |
| Retriever Agent (`retriever_agent/`) | **Maymuna** |
| Answer Agent, Orchestrator, Streamlit app, testing | **Together** |

## Project structure

```
app.py                # Streamlit app (run this)
orchestrator.py       # connects the agents
retriever_agent/      # Retriever + its tools
analyst/              # Analyst + its tools
answer/               # Answer Agent + formatters
shared/               # LLM config and shared schemas
```

## Run it

1. Install:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your OpenRouter key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
3. Start:
   ```
   streamlit run app.py
   ```
4. Upload a document and ask a question.

## Notes

- The free OpenRouter models have a daily request limit. If you see `free-models-per-day`, wait for the reset or add credits.
- Test with a small document first.
