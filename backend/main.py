from fastapi import FastAPI, Query, UploadFile, File, HTTPException
import os
import traceback
from pathlib import Path

from core.rag_answer import rag_answer
from core.embeddings import embed_and_upsert_chunks
from api.ingestion import pdf_to_chunks  # we’ll update this to include page numbers

app = FastAPI(title="DOC Q&A agent", version="1.0")

UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")
PROCESSED_DIR = Path("data/processed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"message": "Doc agent API is running "}


# ---------------------------
# Ask endpoint
# ---------------------------
@app.get("/ask")
def ask(question: str = Query(..., description="Your question for the RAG system")):
    """
    Handles user queries by retrieving answers from vector DB.
    Returns answer text + filename + page numbers instead of chunk ids.
    """
    result = rag_answer(question)

    # Modify response: return page numbers if present
    enhanced_sources = []
    for src in result.get("sources", []):
        enhanced_sources.append({
            "filename": src.get("filename"),
            "page_number": src.get("page_number", "N/A"),  # include page number
            "text_preview": src.get("content", "")[:200]   # optional preview
        })

    return {
        "answer": result.get("answer"),
        "sources": enhanced_sources
    }


# ---------------------------
# upload endpoint
# ---------------------------


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Read file bytes from the uploaded file
    file_bytes = await file.read()

    # Call pdf_to_chunks with bytes (not Path)
    chunks = pdf_to_chunks(file_bytes, file.filename, output_dir=PROCESSED_DIR)

    # Optionally upsert embeddings
    embed_and_upsert_chunks(chunks)

    return {"filename": file.filename, "chunks": chunks}



# ---------------------------
# Delete endpoint
# ---------------------------
@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    """
    Deletes uploaded file + chunks + embeddings from vector DB.
    """
    try:
        # Delete uploaded file
        upload_path = UPLOAD_DIR / filename
        if upload_path.exists():
            upload_path.unlink()

        # Delete chunks file
        chunks_path = CHUNKS_DIR / f"{Path(filename).stem}.json"
        if chunks_path.exists():
            chunks_path.unlink()

        # TODO: Delete from Pinecone by vector ID prefix (filename)
        return {"message": f"{filename} deleted successfully"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
