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

#  Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # must match OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Load chunks
files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")]

for file in files:
    with open(os.path.join(CHUNKS_DIR, file), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk["content"]

        #  Generate embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",  # or "text-embedding-3-large"
            input=text
        )
        embedding = response.data[0].embedding

        # Prepare vector for batch upsert
        vectors.append({
        "id": f"{chunk['filename']}-{chunk['chunk_id']}",  # unique ID
        "values": embedding,
        "metadata": {
        "filename": chunk["filename"],
        "chunk_id": chunk["chunk_id"],
        "content": chunk["content"]   # keep the raw text for retrieval later
    }
})


    #  Upsert batch of vectors for this file
    if vectors:
        index.upsert(vectors)
        print(f"Inserted {len(vectors)} chunks from {file} into Pinecone.")

print(" All chunks embedded and stored in Pinecone.")


