# Course Review RAG System

## Domain and Sources

Domain: Computer Science Course Reviews

The dataset consists of 10 manually created course review documents:

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

The documents simulate student reviews and course descriptions.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using course review documents. Users can ask questions about computer science courses and receive answers generated from retrieved documents.

## Technologies Used

- Python
- sentence-transformers (all-MiniLM-L6-v2)
- ChromaDB
- Groq (Llama 3.3 70B Versatile)

## Pipeline

1. Load course review documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks
6. Generate answers using Groq
7. Display source citations

## Documents

The system uses 10 course review documents covering introductory programming, algorithms, operating systems, databases, artificial intelligence, software engineering, and capstone courses.

## Sample Chunks

### Chunk 1

Source: cs101.txt

CS101 Introduction to Programming

Students describe this course as beginner friendly. The professor explains programming concepts clearly and provides many examples during lectures. Weekly assignments are manageable and help reinforce concepts. Exams focus on basic problem solving and understanding of Python fundamentals. Most students recommend this course for those new to computer science.

---

### Chunk 2

Source: cs102.txt

CS102 Object-Oriented Programming

Students report that this course introduces classes, inheritance, and polymorphism. Programming projects are larger than those in CS101. Homework requires consistent effort throughout the semester. Exams are moderately difficult and often include coding questions. Students who keep up with assignments generally perform well.

---

### Chunk 3

Source: cs201.txt

CS201 Computer Organization

Students learn how software interacts with hardware. Topics include memory, processors, assembly language, and computer architecture. Labs are detailed and require careful attention. Students say the material is difficult at first but becomes easier with practice. Exams focus heavily on understanding system behavior.

---

### Chunk 4

Source: cs202.txt

CS202 Algorithms

Students frequently describe this course as the hardest course in the computer science major. Many students consider Algorithms harder than Data Structures, Operating Systems, and Database Systems. Homework assignments are challenging and exams require strong problem-solving skills.

---

### Chunk 5

Source: cs301.txt

CS301 Operating Systems

Students study processes, threads, synchronization, scheduling, and memory management. Programming projects are complex and often require debugging. The workload is considered heavy. Students report that completing projects significantly improves programming skills. Exams emphasize operating system concepts.

## Running the Project

Install dependencies:

```bash
pip install chromadb sentence-transformers groq python-dotenv
```

Create a .env file:

```text
GROQ_API_KEY=your_api_key_here
```

Run:

```bash
python main.py
```

## Example Questions

- Which course is best for beginners?
- Which course teaches SQL?
- Which course is considered the hardest?
- Which course focuses on operating systems?
- Which course involves team projects?

## Embedding Model

This project uses the sentence-transformers embedding model `all-MiniLM-L6-v2`.

The model converts each document chunk into a numerical vector representation that captures semantic meaning. These vectors are stored in ChromaDB and used for similarity search during retrieval.

I selected `all-MiniLM-L6-v2` because it is lightweight, runs locally, and provides good retrieval performance without requiring a paid API.

### Production Considerations

For a production system, I would consider larger embedding models that may improve retrieval accuracy. However, larger models require more memory, more computation, and may increase latency. The chosen model provides a good balance between speed and retrieval quality for this project.

## Retrieval Test Results

### Query 1

Question:
Which course teaches SQL?

Top Retrieved Chunk:
cs302.txt

Reason:
The chunk directly contains information about SQL and database concepts.

### Query 2

Question:
Which course is best for beginners?

Top Retrieved Chunk:
cs101.txt

Reason:
The chunk explicitly states that CS101 is beginner friendly.

### Query 3

Question:
Which course involves team projects?

Top Retrieved Chunk:
cs402.txt

Reason:
The chunk discusses collaborative software engineering projects.

## Grounded Generation

Retrieved chunks are passed to the Groq-hosted Llama 3.3 model.

The prompt instructs the model to answer only from the provided context and respond with "I don't have enough information" when the answer is unavailable.

## Query Interface

Input:
A user enters a question through the terminal.

Output:
The system displays an answer and the source documents used.

Example:

User:
Which course teaches SQL?

System:
CS302 Database Systems teaches SQL.

Sources:
cs302.txt

## Failure Case

Question:
Which course is hardest?

Initial Result:
I don't have enough information.

Cause:
The retriever did not return cs301.txt because the wording in the document did not strongly match the query.

Resolution:
The document was updated to explicitly describe CS301 as the hardest course.

## Spec Reflection

The planning document helped determine chunk size, overlap, and retrieval strategy before implementation.

One implementation change was modifying cs301.txt after retrieval testing showed that important information was not being retrieved consistently.

## Limitations

- Small document collection.
- Retrieval quality depends on wording.
- No conversational memory.
- Uses only semantic search.

## Future Improvements

- Hybrid search (BM25 + semantic search)
- Larger document collection
- Web interface using Gradio
- Metadata filtering

## AI Usage

ChatGPT was used to assist with debugging Python package installation, ChromaDB setup, and Groq integration.

## Screenshots

The screenshots folder contains example runs of the system answering course-related questions.

Note: The .env file is not included for security reasons. Create a .env file using .env.example and add your own Groq API key.
