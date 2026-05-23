from sentence_transformers import SentenceTransformer

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_embeddings(user_query):
   embedding=model.encode(user_query)
   return embedding.tolist()