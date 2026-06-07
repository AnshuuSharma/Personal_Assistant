from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context

def route(user_query):
    query_embedding = create_embeddings(user_query)
    
    chunks = retrieve(
        query_embedding,
        n_results=5,
        query_text=user_query  
    )
    
    context = format_context(chunks)
    return context