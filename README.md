# Person B: Analyst Agent, Answer Agent, Orchestrator, Streamlit, Semantic Cache

Flow: question -> semantic cache -> Retriever (Person A) -> Analyst (loops back to Retriever if evidence is missing) -> Answer Agent -> user

## Run with Docker
1. Install Docker Desktop and start it.
2. Copy `.env.example` to `.env` and add your OPENROUTER_API_KEY.
3. `docker compose up --build`  then open http://localhost:8501
4. Tests: `docker compose run --rm app python test_all.py`
5. Stop: `docker compose down`

## Connect Person A
Edit only `retriever_bridge.py`: replace `retrieve()` and `ingest()` with her code, set `USING_FAKE = False`,
and merge her requirements into `requirements.txt`.
