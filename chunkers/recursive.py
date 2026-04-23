"""
Recursive Chunker
=================
Splits text using a hierarchy of separators, trying each in order until
chunks are small enough. It first tries to split on double newlines
(paragraph breaks), then single newlines, then sentence-ending punctuation,
then spaces. This preserves natural document structure as much as possible.

This is the strategy used by LangChain's RecursiveCharacterTextSplitter and
is generally the best default for unstructured text documents.

Demo talking point:
    Show how this strategy keeps paragraphs intact and only falls back to
    finer splits when a paragraph is too long. Compare a paragraph-level
    chunk here vs. where the same paragraph lands in fixed-size output.
"""


class RecursiveChunker:
    """
    Splits text recursively using a hierarchy of separators.

    Parameters
    ----------
    chunk_size : int
        Target maximum character length for each chunk.
        Default: 600.
    overlap : int
        Number of characters to repeat at chunk boundaries for context
        continuity. Default: 60.
    separators : list[str] | None
        Ordered list of separators to try. Defaults to a standard hierarchy
        suitable for wiki-style documents.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 60,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the first separator that produces
        chunks within the target size. Falls back to the next separator
        if needed.
        """
        if not separators:
            # Base case: no separators left — return as-is
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split as last resort
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        splits = text.split(separator)

        good_splits = []
        current = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len + len(separator) > self.chunk_size and current:
                # Current accumulation is full — finalize it
                good_splits.append(separator.join(current))
                # Keep overlap: retain the last portion of current for context
                overlap_text = separator.join(current)
                current = [overlap_text[-self.overlap:]] if self.overlap > 0 else []
                current_len = len(current[0]) if current else 0

            if split_len > self.chunk_size:
                # This individual split is too large — recurse with next separator
                if current:
                    good_splits.append(separator.join(current))
                    current = []
                    current_len = 0
                sub_splits = self._split_text(split, remaining_separators)
                good_splits.extend(sub_splits)
            else:
                current.append(split)
                current_len += split_len + len(separator)

        if current:
            good_splits.append(separator.join(current))

        return [s for s in good_splits if s.strip()]

    def chunk(self, text: str) -> list[dict]:
        """
        Split text into chunks using recursive separator hierarchy.

        Returns a list of dicts with keys:
            - text: the chunk content
            - index: position of this chunk in the document
            - strategy: always "recursive"
        """
        raw_chunks = self._split_text(text, self.separators)

        return [
            {
                "text": chunk_text.strip(),
                "index": i,
                "strategy": "recursive",
            }
            for i, chunk_text in enumerate(raw_chunks)
            if chunk_text.strip()
        ]

    def __repr__(self):
        return f"RecursiveChunker(chunk_size={self.chunk_size}, overlap={self.overlap})"
