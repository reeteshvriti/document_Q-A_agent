from fastapi import FastAPI, Query
from core.rag_answer import rag_answer

app = FastAPI(title="DOC Q&A agent", version="1.0")

@app.get("/")
def root():
    return {"message": "Doc agent API is running "}

@app.get("/ask")
def ask(question: str = Query(..., description="Your question for the RAG system")):
    result = rag_answer(question)
    return result
