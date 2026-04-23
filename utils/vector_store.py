"""
Vector Store Utility
====================
A thin wrapper around ChromaDB that handles creating collections,
adding chunks, and querying — one collection per chunking strategy.

Uses sentence-transformers for embeddings (local, free, no API key).
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# Shared embedding function — same model used by the SemanticChunker
# so embeddings are consistent across the pipeline
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Collection names — one per strategy
COLLECTION_NAMES = {
    "fixed_size": "wiki_fixed_size",
    "semantic": "wiki_semantic",
    "recursive": "wiki_recursive",
}


class VectorStore:
    """
    Manages ChromaDB collections for each chunking strategy.

    Parameters
    ----------
    persist_dir : str | Path
        Directory where ChromaDB stores its data on disk.
        Default: "./chroma_db" (relative to where scripts are run from).
    """

    def __init__(self, persist_dir: str | Path = "./chroma_db"):
        self.persist_dir = str(persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

    def get_or_create_collection(self, strategy: str) -> chromadb.Collection:
        """Get or create the ChromaDB collection for a given strategy."""
        name = COLLECTION_NAMES.get(strategy)
        if not name:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(COLLECTION_NAMES.keys())}"
            )
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
        )

    def reset_collection(self, strategy: str):
        """Delete and recreate a collection — useful for re-indexing."""
        name = COLLECTION_NAMES.get(strategy)
        if not name:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        try:
            self.client.delete_collection(name)
        except Exception:
            pass  # Collection didn't exist yet
        return self.client.create_collection(
            name=name,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, strategy: str, chunks: list[dict], source_file: str):
        """
        Add a list of chunks to the collection for the given strategy.

        Parameters
        ----------
        strategy : str
            One of "fixed_size", "semantic", "recursive".
        chunks : list[dict]
            Output from any chunker's .chunk() method.
        source_file : str
            Name of the source document (used in metadata for display).
        """
        collection = self.get_or_create_collection(strategy)

        documents = [c["text"] for c in chunks]
        ids = [f"{source_file}_{strategy}_{c['index']}" for c in chunks]
        metadatas = [
            {
                "source": source_file,
                "strategy": strategy,
                "index": c["index"],
                "char_count": len(c["text"]),
            }
            for c in chunks
        ]

        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def query(self, strategy: str, query_text: str, n_results: int = 3) -> list[dict]:
        """
        Query a collection and return the top results.

        Parameters
        ----------
        strategy : str
            Which collection to query.
        query_text : str
            The natural language query.
        n_results : int
            Number of results to return. Default: 3.

        Returns
        -------
        list[dict]
            Each dict contains: text, score, source, strategy, index, char_count.
        """
        collection = self.get_or_create_collection(strategy)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns L2 distance; convert to a 0–1 similarity score
            similarity = max(0.0, 1.0 - dist / 2.0)
            output.append({
                "text": doc,
                "score": round(similarity, 4),
                "source": meta.get("source", "unknown"),
                "strategy": meta.get("strategy", strategy),
                "index": meta.get("index", -1),
                "char_count": meta.get("char_count", len(doc)),
            })

        return output

    def collection_stats(self, strategy: str) -> dict:
        """Return basic stats about a collection — useful for the demo summary table."""
        collection = self.get_or_create_collection(strategy)
        count = collection.count()
        if count == 0:
            return {"strategy": strategy, "chunk_count": 0, "avg_char_count": 0}

        # Sample all items to compute average chunk size
        all_items = collection.get(include=["metadatas"])
        char_counts = [m.get("char_count", 0) for m in all_items["metadatas"]]
        avg = sum(char_counts) / len(char_counts) if char_counts else 0

        return {
            "strategy": strategy,
            "chunk_count": count,
            "avg_char_count": round(avg),
        }
