# core/utils.py
import os
import json
from PyPDF2 import PdfReader

def pdf_to_chunks(pdf_path: str, output_dir: str, chunk_size: int = 500) -> str:
    """
    Convert PDF into text chunks and save as JSON file in output_dir.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Split into chunks
    words = text.split()
    chunks = []
    chunk_id = 0
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append({
            "filename": os.path.basename(pdf_path),
            "chunk_id": chunk_id,
            "content": chunk_text
        })
        chunk_id += 1

    # Save chunks as JSON
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{os.path.basename(pdf_path)}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return out_file


def delete_namespace(namespace: str):
    """
    Delete all vectors in a Pinecone namespace.
    """
    from pinecone import Pinecone
    import os

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("pdf-qa-embeddings")
    index.delete(delete_all=True, namespace=namespace)
