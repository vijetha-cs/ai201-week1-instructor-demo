"""
Verify Setup
============
Run this script the night before or morning of the demo to confirm
everything is working end-to-end: ChromaDB, embeddings, and OpenRouter.

Usage:
    python setup/verify_setup.py

What it checks:
    1. ChromaDB collections exist and have chunks
    2. Embedding model loads and runs a test query
    3. OpenRouter API key is set and the LLM responds
    4. Prints a comparison table for the demo's primary query

If anything fails, the error message will tell you exactly what to fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vector_store import VectorStore
from utils.llm import LLMClient
from utils.evaluation import evaluate_results, print_comparison_table

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# The primary demo query — same one used in the notebook
DEMO_QUERY = "What happened to Ace at Marineford?"

STRATEGIES = ["fixed_size", "semantic", "recursive"]


def check_collections(store: VectorStore) -> bool:
    print("\n[1/3] Checking ChromaDB collections...")
    all_ok = True
    for strategy in STRATEGIES:
        stats = store.collection_stats(strategy)
        count = stats["chunk_count"]
        if count == 0:
            print(f"  ✗ '{strategy}' collection is empty. Run: python setup/build_indexes.py")
            all_ok = False
        else:
            print(
                f"  ✓ '{strategy}': {count} chunks, "
                f"avg {stats['avg_char_count']} chars/chunk"
            )
    return all_ok


def check_retrieval(store: VectorStore) -> dict | None:
    print(f'\n[2/3] Running test query: "{DEMO_QUERY}"')
    results_by_strategy = {}
    try:
        for strategy in STRATEGIES:
            results = store.query(strategy, DEMO_QUERY, n_results=3)
            results_by_strategy[strategy] = results
            top_score = results[0]["score"] if results else 0
            print(f"  ✓ '{strategy}': top score {top_score:.4f}")
    except Exception as e:
        print(f"  ✗ Retrieval failed: {e}")
        return None
    return results_by_strategy


def check_llm(results_by_strategy: dict) -> bool:
    print("\n[3/3] Testing OpenRouter LLM...")
    try:
        llm = LLMClient()
        print(f"  Using model: {llm.model}")
        # Use the best-scoring chunks from the recursive strategy for the test
        test_chunks = results_by_strategy.get("recursive", [])[:2]
        answer = llm.answer(DEMO_QUERY, test_chunks)
        print(f"  ✓ LLM responded successfully")
        print(f"  Sample answer: \"{answer[:120]}...\"" if len(answer) > 120 else f"  Sample answer: \"{answer}\"")
        return True
    except ValueError as e:
        print(f"  ✗ {e}")
        return False
    except Exception as e:
        print(f"  ✗ OpenRouter request failed: {e}")
        print("    Check your OPENROUTER_API_KEY and internet connection.")
        return False


def main():
    print("=" * 60)
    print("AI 201 — Week 1 Demo: Pre-Demo Verification")
    print("=" * 60)

    store = VectorStore(persist_dir=CHROMA_DIR)

    collections_ok = check_collections(store)
    if not collections_ok:
        print("\n✗ Setup incomplete. Fix the errors above before the demo.")
        sys.exit(1)

    results_by_strategy = check_retrieval(store)
    if results_by_strategy is None:
        print("\n✗ Retrieval check failed.")
        sys.exit(1)

    llm_ok = check_llm(results_by_strategy)

    # Print the comparison table regardless of LLM status
    print("\n--- Retrieval Comparison Table ---")
    eval_output = evaluate_results(DEMO_QUERY, results_by_strategy)
    print_comparison_table(eval_output)

    if not llm_ok:
        print("⚠ LLM check failed — retrieval demo will work but LLM answers won't.")
        print("  Make sure OPENROUTER_API_KEY is set in your .env file.\n")
        sys.exit(1)

    print("=" * 60)
    print("✓ All checks passed. You're ready for the demo!")
    print("  Open demo.ipynb to begin.")
    print("=" * 60)


if __name__ == "__main__":
    main()
