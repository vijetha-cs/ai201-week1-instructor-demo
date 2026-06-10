import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


def load_documents():
    documents = []

    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            filepath = os.path.join("data", filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection("course_reviews")

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

docs = load_documents()

chunk_id = 0

for doc in docs:
    chunks = chunk_text(doc["text"])

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()

        collection.add(
            ids=[str(chunk_id)],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": doc["source"]}]
        )

        chunk_id += 1


print("Documents loaded:", len(docs))
print("Chunks stored in ChromaDB:", chunk_id)


while True:
    question = input("\nAsk a question or type quit: ")

    if question.lower() == "quit":
        break

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    context = ""

    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i]["source"]
        chunk = results["documents"][0][i]

        context += f"\nSource: {source}\n"
        context += chunk + "\n"

    prompt = f"""
Answer the question using ONLY the provided context.

If the answer is not found in the context, say:
I don't have enough information.

Include the source filename in your answer.

Question:
{question}

Context:
{context}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    print("\nANSWER")
    print(answer)

    print("\nSOURCES")
    for metadata in results["metadatas"][0]:
        print("-", metadata["source"])