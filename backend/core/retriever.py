import chromadb
import os
import numpy as np
from rank_bm25 import BM25Okapi
from core.embeddings import create_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORDB_PATH = os.path.join(BASE_DIR, "..", "..", "vectorDB")

client = chromadb.PersistentClient(path=VECTORDB_PATH)
collection = client.get_or_create_collection("personal_info")


_all_docs = None
_bm25 = None

def _load_bm25():
    global _all_docs, _bm25, _all_ids
    if _bm25 is not None:
        return

    print("Loading BM25 index...")
    result = collection.get()
    _all_docs = result["documents"]
    _all_ids = result["ids"]  

    tokenized = [doc.lower().split() for doc in _all_docs]
    _bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built with {len(_all_docs)} chunks")


def dense_retrieve(query_embedding, n_results=10):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    docs = results["documents"][0]
    ids = results["ids"][0]
    return docs, ids


def sparse_retrieve(query, n_results=10):
    _load_bm25()
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    
    top_indices = np.argsort(scores)[::-1][:n_results]
    docs = [_all_docs[i] for i in top_indices]
    ids = [_all_ids[i] for i in top_indices]
    return docs, ids


def reciprocal_rank_fusion(dense_docs, dense_ids, sparse_docs, sparse_ids, k=60):
    """
    RRF score = sum(1 / (k + rank)) for each retriever
    Higher score = more relevant
    """
    rrf_scores = {}
    doc_map = {} 

    for rank, (doc, doc_id) in enumerate(zip(dense_docs, dense_ids)):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, (doc, doc_id) in enumerate(zip(sparse_docs, sparse_ids)):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

  
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def retrieve(query_embedding, n_results=5, query_text=None):
    """
    Hybrid retrieval combining dense + sparse via RRF.
    Falls back to dense-only if query_text not provided.
    """
    if query_text is None:
       
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results["documents"][0]

    candidate_count = n_results * 2

    dense_docs, dense_ids = dense_retrieve(query_embedding, candidate_count)
    sparse_docs, sparse_ids = sparse_retrieve(query_text, candidate_count)

    fused = reciprocal_rank_fusion(
        dense_docs, dense_ids,
        sparse_docs, sparse_ids
    )

    return fused[:n_results]


def format_context(chunks):
    return "\n\n".join(chunks)