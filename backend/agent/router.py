from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context, CLARIFY_INSTRUCTION
import time

def route(user_query):
    t0=time.time()
    query_embedding = create_embeddings(user_query)
    t1=time.time()

    print(f"Embedding:{t1-t0:.3f}s")

    chunks, confidence = retrieve(
        query_embedding,
        n_results=5,
        query_text=user_query
    )
    t2=time.time()

    print(f"Retrieval : {t2-t1:.3f}s (confidence={confidence})")

    if confidence == "out_of_scope":
        return None, confidence  

    context = format_context(chunks)

    if confidence == "low_confidence":
        context += CLARIFY_INSTRUCTION

    t3=time.time()
    print(f"Format : {t3-t2:.3f}s")
    return context, confidence