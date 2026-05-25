import json
from sentence_transformers import SentenceTransformer
import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks_path = os.path.join(BASE_DIR, "..", "data", "chunks", "chunks.json")
vectordb_path = os.path.join(BASE_DIR, "..", "..", "vectorDB")
with open(chunks_path,"r") as f:
    chunks=json.load(f)

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings=model.encode(chunks)

client=chromadb.PersistentClient(path=vectordb_path)
collection=client.get_or_create_collection("personal_info")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"{len(chunks)} chunks embedded and saved to ChromaDB")
