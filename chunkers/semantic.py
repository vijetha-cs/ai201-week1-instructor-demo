"""
Semantic Chunker
================
Groups sentences together based on semantic similarity. Sentences that are
closely related in meaning stay in the same chunk; when the topic shifts
significantly, a new chunk begins.

This strategy requires an embedding model to measure similarity between
adjacent sentences. We use sentence-transformers (local, free, no API key).

Demo talking point:
    Show students a chunk from fixed-size that cuts mid-thought, then show
    the same passage chunked semantically — the related sentences stay together.
    Ask: "Why might this matter when answering 'What happened to Ace at Marineford?'"
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticChunker:
    """
    Groups sentences into chunks based on semantic similarity between
    adjacent sentences.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model to use for embeddings.
        Default: "all-MiniLM-L6-v2" — small, fast, runs locally, no API key.
    similarity_threshold : float
        Cosine similarity below which a sentence starts a new chunk.
        Range: 0.0 (always split) to 1.0 (never split).
        Default: 0.5 — a good starting point for narrative text.
    max_chunk_size : int
        Maximum character length for a single chunk. Prevents runaway chunks
        when a long passage has uniformly high similarity throughout.
        Default: 1000.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1000,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self._model = None  # lazy load

    @property
    def model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            print(f"Loading embedding model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            print("Model loaded.")
        return self._model

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Naive sentence splitter using punctuation markers.
        Handles common patterns well enough for wiki-style text.
        """
        import re
        # Split on . ! ? followed by whitespace and a capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter out empty strings and very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def chunk(self, text: str) -> list[dict]:
        """
        Split text into semantically coherent chunks.

        Returns a list of dicts with keys:
            - text: the chunk content
            - index: position of this chunk in the document
            - sentence_count: how many sentences are in this chunk
            - strategy: always "semantic"
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [{"text": text, "index": 0, "sentence_count": 1, "strategy": "semantic"}]

        # Embed all sentences in one batch (efficient)
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        chunks = []
        current_sentences = [sentences[0]]
        current_start = 0
        chunk_index = 0

        for i in range(1, len(sentences)):
            # Compare current sentence to the previous one
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i - 1].reshape(1, -1)
            )[0][0]

            current_text = " ".join(current_sentences)
            would_exceed_max = len(current_text) + len(sentences[i]) > self.max_chunk_size

            if sim < self.similarity_threshold or would_exceed_max:
                # Topic shift detected (or chunk too large) — save current chunk
                chunks.append({
                    "text": current_text,
                    "index": chunk_index,
                    "sentence_count": len(current_sentences),
                    "strategy": "semantic",
                })
                current_sentences = [sentences[i]]
                chunk_index += 1
            else:
                # Similar enough — keep building the current chunk
                current_sentences.append(sentences[i])

        # Don't forget the last chunk
        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "index": chunk_index,
                "sentence_count": len(current_sentences),
                "strategy": "semantic",
            })

        return chunks

    def __repr__(self):
        return (
            f"SemanticChunker(model='{self.model_name}', "
            f"threshold={self.similarity_threshold}, "
            f"max_chunk_size={self.max_chunk_size})"
        )
