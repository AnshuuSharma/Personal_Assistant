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
_all_ids = None

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
    rrf_scores = {}
    doc_map = {}

    for rank, (doc, doc_id) in enumerate(zip(dense_docs, dense_ids)):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, (doc, doc_id) in enumerate(zip(sparse_docs, sparse_ids)):
        weight = 1.5 if rank < 3 else 1.0  # boost top BM25 results
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (k + rank + 1)
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


def retrieve_with_rerank(question, eval_llm_fn, n_results=5):
    """
    Evaluation-only function.
    Fetches 10 candidates via hybrid search then uses LLM to rerank.
    """
    embedding = create_embeddings(question)
    candidates = retrieve(embedding, n_results=10, query_text=question)

    rerank_prompt = f"""Given this question: "{question}"

Rank these chunks by relevance. Return indices of top {n_results}
most relevant chunks as JSON only, no explanation:
{{"top_indices": [0, 3, 1, 4, 2]}}

Chunks:
""" + "\n".join([f"[{i}] {c[:200]}" for i, c in enumerate(candidates)])

    try:
        response = eval_llm_fn(rerank_prompt, max_tokens=100)
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        parsed = __import__('json').loads(clean)
        indices = parsed["top_indices"][:n_results]
        return [candidates[i] for i in indices if i < len(candidates)]
    except Exception as e:
        print(f"  Rerank failed ({e}), falling back to hybrid")
        return candidates[:n_results]


def format_context(chunks):
    return "\n\n".join(chunks)

_load_bm25()