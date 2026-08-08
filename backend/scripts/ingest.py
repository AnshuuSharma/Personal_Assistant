import json
import os
import time
from google import genai
from google.genai import types
import chromadb
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks_path = os.path.join(BASE_DIR, "..", "data", "chunks", "chunks.json")
vectordb_path = os.path.join(BASE_DIR, "..", "..", "vectorDB")

with open(chunks_path, "r") as f:
    chunks = json.load(f)

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  
def embed_documents_batch(texts, batch_size=20):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client_genai.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM
            )
        )
        all_embeddings.extend([e.values for e in result.embeddings])
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.5)
    return all_embeddings

embeddings = embed_documents_batch(chunks)

client = chromadb.PersistentClient(path=vectordb_path)

try:
    client.delete_collection("personal_info")
    print("Deleted old collection (dimension/model change)")
except Exception:
    print("No existing collection to delete, creating fresh")

collection = client.get_or_create_collection("personal_info")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"{len(chunks)} chunks embedded and saved to ChromaDB")