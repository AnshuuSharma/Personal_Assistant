import chromadb
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
VECTORDB_PATH=os.path.join(BASE_DIR,"..","..","vectorDB")

client=chromadb.PersistentClient(path=VECTORDB_PATH)
collection=client.get_or_create_collection("personal_info")

def retrieve(query_embedding,n_results=3):
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks=results["documents"][0]
    return chunks

def format_context(chunks):
    return "\n\n".join(chunks)