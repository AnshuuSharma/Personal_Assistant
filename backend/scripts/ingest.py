import json
from sentence_transformers import SentenceTransformer
import chromadb

with open("data/chunks/chunks.json","r") as f:
    chunks=json.load(f)

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings=model.encode(chunks)

client=chromadb.PersistentClient(path="../../vectorDB")
collection=client.get_or_create_collection("personal_info")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"{len(chunks)} chunks embedded and saved to ChromaDB")
