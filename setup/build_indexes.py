"""
Build Indexes
=============
One-time setup script. Run this before the demo to chunk all wiki pages
and load them into ChromaDB.

Usage:
    python setup/build_indexes.py

This will:
    1. Read all .txt files from data/wiki_pages/
    2. Chunk each file using all three strategies
    3. Store the chunks in ChromaDB (persisted to ./chroma_db/)

Run time: ~2-3 minutes on a laptop (mostly the sentence-transformers
embedding step for the semantic chunker).

Re-running this script is safe — it resets all collections first.
"""

import sys
from pathlib import Path

# Make sure imports work from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunkers import FixedSizeChunker, SemanticChunker, RecursiveChunker
from utils.vector_store import VectorStore


DATA_DIR = Path(__file__).parent.parent / "data" / "wiki_pages"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


def load_wiki_pages(data_dir: Path) -> dict[str, str]:
    """Load all .txt files from the wiki_pages directory."""
    pages = {}
    for path in sorted(data_dir.glob("*.txt")):
        pages[path.stem] = path.read_text(encoding="utf-8")
    return pages


def main():
    print("=" * 60)
    print("AI 201 — Week 1 Demo: Building RAG Indexes")
    print("=" * 60)

    # Load wiki pages
    print(f"\nLoading wiki pages from {DATA_DIR}...")
    pages = load_wiki_pages(DATA_DIR)
    if not pages:
        print(f"ERROR: No .txt files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(pages)} pages: {', '.join(pages.keys())}")

    # Initialize chunkers
    fixed_chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    semantic_chunker = SemanticChunker(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.5,
        max_chunk_size=1000,
    )
    recursive_chunker = RecursiveChunker(chunk_size=600, overlap=60)

    chunkers = {
        "fixed_size": fixed_chunker,
        "semantic": semantic_chunker,
        "recursive": recursive_chunker,
    }

    # Initialize vector store and reset all collections
    print(f"\nInitializing ChromaDB at {CHROMA_DIR}...")
    store = VectorStore(persist_dir=CHROMA_DIR)

    print("Resetting existing collections...")
    for strategy in chunkers:
        store.reset_collection(strategy)

    # Process each page with each chunker
    total_chunks = {"fixed_size": 0, "semantic": 0, "recursive": 0}

    for page_name, page_text in pages.items():
        print(f"\nProcessing: {page_name}.txt ({len(page_text):,} chars)")

        for strategy, chunker in chunkers.items():
            print(f"  [{strategy}] chunking...", end=" ", flush=True)
            chunks = chunker.chunk(page_text)
            store.add_chunks(strategy, chunks, source_file=page_name)
            total_chunks[strategy] += len(chunks)
            print(f"{len(chunks)} chunks")

    # Print summary
    print("\n" + "=" * 60)
    print("Index build complete!")
    print("=" * 60)
    print(f"\n{'Strategy':<15} {'Total Chunks':>13}")
    print("-" * 30)
    for strategy, count in total_chunks.items():
        stats = store.collection_stats(strategy)
        print(
            f"{strategy:<15} {count:>13}  "
            f"(avg {stats['avg_char_count']} chars/chunk)"
        )

    print(
        f"\nChromaDB data saved to: {CHROMA_DIR}\n"
        "You're ready to run the demo. Open demo.ipynb to begin.\n"
    )


if __name__ == "__main__":
    main()
