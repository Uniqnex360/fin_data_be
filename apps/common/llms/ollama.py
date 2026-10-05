from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from django.conf import settings


@lru_cache(maxsize=None)
def get_ollama_instance(model: str = "qwen3.5:4b"):

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    return ChatOllama(
        model=model,
        base_url=settings.OLLAMA_URL,
        temperature=0,
        num_ctx=16384,
        reasoning=False,
    )
