import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_EMBEDDING"))

MODEL = "gemini-embedding-001"
DIM = 768

def create_embeddings(user_query):
    """Use for embedding a search query (retrieval time)."""
    result = client.models.embed_content(
        model=MODEL,
        contents=user_query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIM
        )
    )
    return result.embeddings[0].values

def create_document_embeddings(text):
    """Use for embedding chunks when building/rebuilding your ChromaDB collection."""
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=DIM
        )
    )
    return result.embeddings[0].values