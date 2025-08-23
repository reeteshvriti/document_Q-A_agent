# rag_answer.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from dataclasses import dataclass

# --- Config ---
INDEX_NAME = "pdf-qa-embeddings"   # must match what you used in embeddings.py
TOP_K = 5                           # how many chunks to retrieve
MAX_CONTEXT_CHARS = 6000            # guardrail to avoid over-long prompts

# --- Init ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@dataclass
class RetrievedChunk:
    filename: str
    chunk_id: int
    content: str
    score: float

def embed_query(text: str) -> List[float]:
    """Create an embedding vector for the user query."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def retrieve_chunks(query: str) -> List[RetrievedChunk]:
    """Search Pinecone and return top-k chunks with metadata."""
    qvec = embed_query(query)
    resp = index.query(
        vector=qvec,
        top_k=TOP_K,
        include_metadata=True
    )
    chunks: List[RetrievedChunk] = []
    for m in resp.matches or []:
        md = m.metadata or {}
        chunks.append(
            RetrievedChunk(
                filename=str(md.get("filename", "unknown")),
                chunk_id=int(md.get("chunk_id", -1)),
                content=str(md.get("content", "")),
                score=float(m.score) if hasattr(m, "score") else 0.0
            )
        )
    return chunks

def build_context(chunks: List[RetrievedChunk]) -> str:
    """Build a bounded context string with per-chunk source tags for citations."""
    # Sort by score desc (optional, Pinecone already returns sorted)
    chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    context_parts = []
    total = 0
    for ch in chunks:
        tag = f"[{ch.filename}#{ch.chunk_id}]"
        block = f"{tag}\n{ch.content}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            # If the next block would overflow, stop.
            break
        context_parts.append(block)
        total += len(block)

    return "\n---\n".join(context_parts)

SYSTEM_PROMPT = (
    "You are a careful assistant for answering questions using provided document context. "
    "Follow the rules strictly:\n"
    "1) Use ONLY the provided context to answer. If the answer is not present, say you don't know.\n"
    "2) Be concise and factual. Do not invent details beyond the context.\n"
    "3) After any specific claim, include inline citations like [filename#chunkId].\n"
    "4) If multiple places support a claim, cite the strongest one or two."
)

def answer_with_context(question: str, context: str) -> str:
    """Call the chat model and produce a grounded answer with citations."""
    if not context.strip():
        return "I couldn't find relevant context for your question in the knowledge base."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer (use citations like [filename#chunkId]):"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def rag_answer(question: str) -> Dict:
    """Full RAG flow: retrieve → build prompt → generate answer."""
    hits = retrieve_chunks(question)
    context = build_context(hits)
    answer = answer_with_context(question, context)
    sources = [{"filename": h.filename, "chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    # Quick manual test
    q = input("Enter your question: ").strip()
    result = rag_answer(q)
    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(s)
