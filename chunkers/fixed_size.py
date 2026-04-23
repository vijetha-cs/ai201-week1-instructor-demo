"""
Fixed-Size Chunker
==================
Splits text into chunks of a fixed character length with optional overlap.
Makes no attempt to respect word or sentence boundaries — it splits wherever
the character count lands. This is the simplest chunking strategy and the
fastest to implement, but it can cut sentences and ideas in half.

Demo talking point:
    Ask students: "What's the tradeoff of a smaller chunk size vs. larger?"
    Smaller = more precise retrieval but may lose context.
    Larger = more context but may pull in irrelevant content.
"""


class FixedSizeChunker:
    """
    Splits text by character count with configurable size and overlap.

    Parameters
    ----------
    chunk_size : int
        Number of characters per chunk. Default: 500.
    overlap : int
        Number of characters to repeat at the start of the next chunk.
        Overlap helps avoid cutting off context at chunk boundaries.
        Default: 50.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[dict]:
        """
        Split text into fixed-size chunks.

        Returns a list of dicts with keys:
            - text: the chunk content
            - index: position of this chunk in the document
            - start_char: character offset where this chunk begins
            - strategy: always "fixed_size"
        """
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "index": index,
                "start_char": start,
                "strategy": "fixed_size",
            })

            # Advance by chunk_size minus overlap so the next chunk
            # starts slightly before the end of this one
            start += self.chunk_size - self.overlap
            index += 1

        return chunks

    def __repr__(self):
        return f"FixedSizeChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
