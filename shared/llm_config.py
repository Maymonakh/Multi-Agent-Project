import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    """Create the configured OpenRouter chat model only when an LLM is needed."""
    from langchain_openrouter import ChatOpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your local .env file.")
    return ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        api_key=api_key,
        temperature=0,
    )
