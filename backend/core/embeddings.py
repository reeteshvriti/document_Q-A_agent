import os
import json
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-qa-embeddings"
index = pc.Index(INDEX_NAME)

def embed_and_upsert_chunks(chunks: list):
    """Embed and insert given chunks into Pinecone."""
    vectors = []
    for chunk in chunks:
        vector_id = f"{chunk['filename']}-{chunk['chunk_id']}"
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["content"]
        )
        embedding = response.data[0].embedding
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": chunk
        })

    if vectors:
        index.upsert(vectors)

def delete_document(doc_id: str):
    """Delete all vectors related to a specific document ID."""
    index.delete(delete_all=False, filter={"filename": {"$eq": doc_id}})
