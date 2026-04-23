"""
LLM Client (OpenRouter)
=======================
A simple wrapper around OpenRouter using the openai Python client.
OpenRouter is free to sign up and provides access to many open-source
models (Llama 3, Mistral, Gemma, etc.) through an OpenAI-compatible API.

Sign up: https://openrouter.ai
Get your API key: https://openrouter.ai/keys
"""

import os
from openai import OpenAI


# Default free model on OpenRouter. Can be changed via the MODEL env var
# or by passing model= to the LLMClient constructor.
# Full list of free models: https://openrouter.ai/models?q=free
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """
    Sends RAG queries to an LLM via OpenRouter.

    Parameters
    ----------
    api_key : str | None
        OpenRouter API key. Reads from OPENROUTER_API_KEY env var if not provided.
    model : str | None
        OpenRouter model identifier. Reads from OPENROUTER_MODEL env var,
        then falls back to DEFAULT_MODEL.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY "
                "environment variable or pass api_key= to LLMClient()."
            )
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    def answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Answer a query using retrieved context chunks.

        Parameters
        ----------
        query : str
            The user's question.
        context_chunks : list[dict]
            Retrieved chunks from the vector store (output of VectorStore.query()).

        Returns
        -------
        str
            The LLM's answer.
        """
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}"
            for c in context_chunks
        )

        system_prompt = (
            "You are a helpful assistant answering questions about the One Piece anime/manga. "
            "Use only the provided context to answer. If the context does not contain enough "
            "information to answer the question, say so clearly. "
            "Keep your answer concise — 2 to 4 sentences."
        )

        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Low temp for factual retrieval tasks
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()

    def __repr__(self):
        return f"LLMClient(model='{self.model}')"
