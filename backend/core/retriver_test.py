import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from config import CHUNKS_DIR

from dotenv import load_dotenv
load_dotenv()

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
INDEX_NAME = "pdf-qa-embeddings"

index = pc.Index(INDEX_NAME)

# --- Step 1: Create an embedding for your query ---
query = "What is end stage renal disease?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small", 
    input=query
).data[0].embedding

# --- Step 2: Query Pinecone for similar vectors ---
results = index.query(
    vector=query_embedding,
    top_k=5,                     # Number of most similar chunks to return
    include_metadata=True        # So you get back your text chunks
)

# --- Step 3: Show results ---
for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"From: {match['metadata']['filename']} (chunk {match['metadata']['chunk_id']})")
    print(f"Text: {match['metadata']['content']}\n")
