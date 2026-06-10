# RAG Project Planning

## Domain

I chose student course reviews for computer science courses. This domain is useful because students often want information about course difficulty, workload, topics covered, and recommendations before enrolling.

## Documents

The project uses 10 course review documents:

- cs101.txt
- cs102.txt
- cs201.txt
- cs202.txt
- cs301.txt
- cs302.txt
- cs303.txt
- cs401.txt
- cs402.txt
- cs499.txt

## Chunking Strategy

Documents are split into chunks of 500 characters with an overlap of 100 characters. This allows important information to remain together while creating manageable chunks for retrieval.

## Retrieval Approach

Embeddings are generated using the sentence-transformers model all-MiniLM-L6-v2. Chunks are stored in ChromaDB and semantic similarity search is used to retrieve the most relevant chunks for a user query.

## Evaluation Plan

Test Questions:

1. Which course is best for beginners?
   - Expected Answer: CS101

2. Which course teaches SQL?
   - Expected Answer: CS302

3. Which course is considered the hardest?
   - Expected Answer: CS301

4. Which course involves team projects?
   - Expected Answer: CS402

5. Which course focuses on operating systems?
   - Expected Answer: CS303

## Anticipated Challenges

- Retrieval may return related but incorrect chunks.
- Small datasets can lead to weak retrieval performance.
- Some questions may not match document wording exactly.
