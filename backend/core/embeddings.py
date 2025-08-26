# core/embeddings.py
import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-qa-embeddings"
index = pc.Index(INDEX_NAME)

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chunks")

def embed_and_upsert_chunks():
    files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")]

    for file in files:
        with open(os.path.join(CHUNKS_DIR, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        vectors = []
        for chunk in chunks:
            # Check if already exists to avoid duplicate embedding
            vector_id = f"{chunk['filename']}-{chunk['chunk_id']}"
            # Generate embedding
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
