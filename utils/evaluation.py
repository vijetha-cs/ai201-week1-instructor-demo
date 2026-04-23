"""
Evaluation Utilities
====================
Simple, dependency-light evaluation for the demo. No LLM-as-judge needed —
we use the retrieval scores already returned by ChromaDB plus a few
descriptive statistics to make the comparison between chunking strategies
immediately visible.

Demo talking point:
    "In a production system you'd use a framework like RAGAS to measure
    context relevance and faithfulness with an LLM judge. For today's demo
    we're using the retrieval scores directly — they tell us how well each
    strategy matched the query. Higher score = more semantically similar."
"""

from typing import Any


def evaluate_results(
    query: str,
    results_by_strategy: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Compute a comparison summary across chunking strategies for a given query.

    Parameters
    ----------
    query : str
        The query that was run.
    results_by_strategy : dict[str, list[dict]]
        Keys are strategy names ("fixed_size", "semantic", "recursive").
        Values are lists of result dicts from VectorStore.query().

    Returns
    -------
    dict with:
        - query: the original query
        - summary: list of per-strategy stats (for display as a table)
        - winner: strategy with the highest average retrieval score
    """
    summary = []

    for strategy, results in results_by_strategy.items():
        if not results:
            summary.append({
                "strategy": strategy,
                "chunks_retrieved": 0,
                "avg_score": 0.0,
                "top_score": 0.0,
                "avg_chunk_length": 0,
            })
            continue

        scores = [r["score"] for r in results]
        lengths = [r["char_count"] for r in results]

        summary.append({
            "strategy": strategy,
            "chunks_retrieved": len(results),
            "avg_score": round(sum(scores) / len(scores), 4),
            "top_score": round(max(scores), 4),
            "avg_chunk_length": round(sum(lengths) / len(lengths)),
        })

    # Pick the strategy with the highest average score
    winner = max(summary, key=lambda x: x["avg_score"])["strategy"] if summary else "n/a"

    return {
        "query": query,
        "summary": summary,
        "winner": winner,
    }


def print_comparison_table(eval_output: dict):
    """
    Print a formatted comparison table to the console.
    Used in setup/verify_setup.py and optionally in the notebook.
    """
    print(f"\nQuery: \"{eval_output['query']}\"\n")
    print(f"{'Strategy':<15} {'Chunks':>8} {'Top Score':>10} {'Avg Score':>10} {'Avg Length':>11}")
    print("-" * 58)

    for row in eval_output["summary"]:
        marker = " ◀ best" if row["strategy"] == eval_output["winner"] else ""
        print(
            f"{row['strategy']:<15} "
            f"{row['chunks_retrieved']:>8} "
            f"{row['top_score']:>10.4f} "
            f"{row['avg_score']:>10.4f} "
            f"{row['avg_chunk_length']:>11}{marker}"
        )

    print()


def format_result_for_display(result: dict, max_text_length: int = 300) -> str:
    """
    Format a single retrieval result for clean display in the notebook.

    Parameters
    ----------
    result : dict
        A single result from VectorStore.query().
    max_text_length : int
        Truncate the chunk text at this length for display. Default: 300.
    """
    text = result["text"]
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    return (
        f"[Score: {result['score']:.4f} | Source: {result['source']} | "
        f"Chars: {result['char_count']}]\n{text}"
    )
