# AI 201 — Week 1 Demo: Chunking Strategy Comparison

## What This Demo Does

You'll run a live lore Q&A system built on One Piece wiki pages using three chunking strategies — fixed-size, semantic, and recursive — and show students how the same query returns different results depending on how the documents were chunked. The demo makes the point that chunking is a design decision with real consequences, not a default setting to leave alone.

**Demo query:** *"What happened to Ace at Marineford?"*

The answer requires connecting information from multiple sections of multiple wiki pages. Each chunking strategy handles this differently, and the comparison is immediately visible.

---

## Setup (Do This Before the Demo)

### Step 1 — Prerequisites

You need Python 3.11+ installed. Verify with:

```bash
python --version
```

### Step 2 — Clone and Install

```bash
git clone <repo-url>
cd ai201-week1-starter-repo
pip install -r requirements.txt
```

The first install will download the `all-MiniLM-L6-v2` sentence-transformers model (~80MB). This only happens once — it caches locally after that.

### Step 3 — Set Up Your OpenRouter Key

Sign up for a free account at [openrouter.ai](https://openrouter.ai). No credit card required. Get your API key at [openrouter.ai/keys](https://openrouter.ai/keys).

Then:

```bash
cp .env.example .env
```

Open `.env` and replace `your_openrouter_api_key_here` with your actual key.

### Step 4 — Build the Indexes

This chunks all five wiki pages using all three strategies and stores everything in ChromaDB. Run it once before the demo. It takes about 2–3 minutes.

```bash
python setup/build_indexes.py
```

You should see output like:

```bash
Processing: marineford_arc.txt (6,482 chars)
  [fixed_size] chunking... 14 chunks
  [semantic] chunking... 9 chunks
  [recursive] chunking... 12 chunks
...
Index build complete!
```

### Step 5 — Verify Everything Works

Run this the night before or morning of the demo:

```bash
python setup/verify_setup.py
```

This confirms ChromaDB is populated, the embedding model loads, OpenRouter responds, and prints a preview of the comparison table. If anything fails, the output tells you exactly what to fix.

### Step 6 — Open the Notebook

```bash
jupyter notebook demo.ipynb
```

Run **Cell 1** (setup cell) to confirm imports load cleanly. Then leave the notebook in that state — ready to demo.

---

## Demo Guide

### Before You Start

- Have `demo.ipynb` open with **Cell 1 already run** (setup cell)
- Know which query you're leading with: *"What happened to Ace at Marineford?"*
- Know the two functions you'll write live (see below — read them before class)

---

### Part 1: Write the Fixed-Size Chunker (~3 min)

**You write this live. Students watch.**

**Cell 2** is blank with instructor comments. Type out `chunk_fixed()` while narrating:

```python
def chunk_fixed(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

Ask students before you start: *"What parameters do we need?"* (chunk_size, overlap). Ask as you write the loop: *"How do we know when to stop?"* (when start reaches the end of the text). Ask after you run it: *"What's the tradeoff of a smaller chunk size vs. a larger one?"*

**Cell 3** shows a boundary between two adjacent fixed-size chunks — point out where it cut mid-sentence. Ask: *"Is this a complete thought? What's missing?"*

---

### Part 2: Compare Pre-Built Strategies (~2 min)

**Run Cells 4–5.** These use the already-built semantic and recursive chunkers.

**Cell 4** loads both chunkers and prints chunk counts and average sizes across all three. Draw attention to the differences — the number of chunks and their sizes tell a story even before you look at content.

**Cell 5** shows how semantic and recursive chunkers handled the same Ace's-death passage. Ask: *"Why did the semantic chunker group those sentences together?"* (Topic similarity — the narrative shifted.) Ask: *"Why did the recursive chunker preserve the paragraph?"* (It splits on paragraph breaks first, only falls back to finer splits when needed.)

**Talking point to land here:** All three strategies are looking at the same document. The chunking decision changes what the retriever has to work with — before a single query is ever run.

---

### Part 3: Run the Query (~2 min)

**Run Cell 6.**

Runs *"What happened to Ace at Marineford?"* against all three pre-built indexes simultaneously. Students watch the scores and top chunks appear.

Ask: *"Look at the top result for each strategy. Which one retrieved the most complete picture?"*

What to expect:

- Fixed-size often captures the moment Ace is struck but not the emotional context — the key scene is split across chunk boundaries
- Semantic tends to pull the event and Ace's final words into the same chunk
- Recursive typically retrieves a clean paragraph covering the core facts

Results may vary slightly run-to-run. That variation is the lesson.

---

### Part 4: Write the RAG Pipeline (~4 min)

**You write this live. Students watch.**

**Cell 7** is blank with instructor comments. Type out `rag_answer()` while narrating:

```python
def rag_answer(query, chunks, llm_client):
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}"
        for c in chunks
    )
    system = (
        "You are a helpful assistant answering questions about One Piece. "
        "Use only the provided context to answer. "
        "If the context doesn't have enough information, say so."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
```

Ask before writing: *"What does the LLM need from us?"* (The question and the context.) Ask as you write the context assembly: *"Why are we joining multiple chunks instead of just sending one?"* (We retrieved 3 — each adds more context.) Ask after the test answer runs: *"What would happen if we sent 20 chunks instead of 3?"*

---

### Part 5: Compare the Answers (~1 min)

**Run Cell 8.**

Calls `rag_answer()` for all three strategies and prints the results side by side. Ask: *"Why are these answers different if we used the exact same model with the same code?"*

The answer is the entire point of the demo: the model can only work with what retrieval gave it. Bad chunks → incomplete context → worse answers. The LLM is not the bottleneck. The pipeline is.

---

### Part 6: Evaluation Table (~1 min)

**Run Cell 9.**

Show the comparison table. Ask: *"This measures retrieval quality. What else would we want to measure in production?"*

Guide toward: faithfulness (did the LLM make things up?), answer relevance (did it address the question?), groundedness (is every claim supported by the retrieved context?). Preview: those are the metrics RAGAS measures.

---

### Live Try It (~1 min)

**Run the final cell** with a query suggested by a student. Good fallback options:

- *"How does Haki counter Logia Devil Fruit powers?"*
- *"Why did Ace turn back to fight Akainu instead of escaping?"*
- *"What happened to Ace's Devil Fruit after he died?"*
- *"What was Whitebeard's reputation before Marineford?"*

---

## Troubleshooting

**"Collection is empty" in verify_setup.py**
You haven't run `build_indexes.py` yet, or it errored partway through. Re-run it and check for errors.

**"OpenRouter API key not found"**
Your `.env` file is missing or the key isn't set. Make sure you copied `.env.example` to `.env` and filled in your actual key.

**"OpenRouter request failed"**
Check your internet connection. OpenRouter may be having an outage — check [status.openrouter.ai](https://status.openrouter.ai). As a fallback, the retrieval portion of the demo (Parts 1–2 and the evaluation table) works without the LLM.

**HTTP 404 — "No endpoints found for ..."**
OpenRouter rotates its `:free` endpoints — the model in `.env` was retired upstream. Pick a different free model from [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free) and set it as `OPENROUTER_MODEL` in your `.env`, then restart the Jupyter kernel.

**HTTP 429 — "temporarily rate-limited upstream"**
OpenRouter's free tier shares a rate-limit pool across all users of a given upstream provider. The Meta Llama free models (`llama-3.x-*:free`) all route through Venice and get throttled aggressively. Swap `OPENROUTER_MODEL` in `.env` to a model on a different upstream provider — `openai/gpt-oss-20b:free`, `google/gemma-4-31b-it:free`, or `deepseek/deepseek-v4-flash:free` — then restart the Jupyter kernel.

**Sentence-transformers model is slow to load**
The model downloads on first use and caches locally. If you're demoing on a slow connection, run `build_indexes.py` at home before coming in — the model will be cached and loads in seconds after that.

**Semantic chunker produces very few chunks**
The `similarity_threshold` may be too high for your data, grouping too many sentences together. Lower it slightly (e.g., from 0.5 to 0.4) in `setup/build_indexes.py` and re-run.

---

## Repo Structure

```plaaintext
ai201-week1-starter-repo/
├── README.md                        # This file
├── requirements.txt
├── .env.example                     # Copy to .env and add your key
├── demo.ipynb                       # The demo notebook
│
├── data/
│   └── wiki_pages/                  # One Piece wiki source documents
│       ├── marineford_arc.txt
│       ├── portgas_d_ace.txt
│       ├── whitebeard.txt
│       ├── devil_fruits.txt
│       └── haki.txt
│
├── chunkers/                        # The three chunking strategies
│   ├── fixed_size.py
│   ├── semantic.py
│   └── recursive.py
│
├── utils/
│   ├── vector_store.py              # ChromaDB wrapper
│   ├── llm.py                       # OpenRouter client
│   └── evaluation.py               # Scoring and display utilities
│
└── setup/
    ├── build_indexes.py             # One-time setup: chunk + embed + store
    └── verify_setup.py             # Pre-demo sanity check
```

The `chroma_db/` directory is created automatically by `build_indexes.py` and is not committed to the repo.
