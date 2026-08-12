from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context, OUT_OF_SCOPE_MESSAGE
import time

def route(user_query):
    t0=time.time()
    query_embedding = create_embeddings(user_query)
    t1=time.time()

    print(f"Embedding:{t1-t0:.3f}s")

    chunks, out_of_scope = retrieve(
        query_embedding,
        n_results=5,
        query_text=user_query
    )
    t2=time.time()

    print(f"Retrieval : {t2-t1:.3f}s")

    if out_of_scope:
        print("Query out of scope -- skipping context/LLM")
        return None  

    context = format_context(chunks)
    t3=time.time()
    print(f"Format : {t3-t2:.3f}s")
    return context