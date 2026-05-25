from backend.core.embeddings import create_embeddings
from backend.core.retriever import retrieve, format_context


def route(user_query):
    query_embedding = create_embeddings(user_query)
    
    chunks = retrieve(query_embedding)
    
    context = format_context(chunks)

    return context
    