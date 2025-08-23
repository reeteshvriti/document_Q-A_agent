import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import BASE_DIR, CHUNKS_DIR, PROCESSED_DIR


# Ensure chunks directory exists
os.makedirs(CHUNKS_DIR, exist_ok=True)


# ------------------------------
# Load processed JSON files
# ------------------------------
def load_processed_files():
    documents = []
    for file in os.listdir(PROCESSED_DIR):
        if file.endswith(".json"):
            file_path = os.path.join(PROCESSED_DIR, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Flatten all page texts
            if isinstance(data, list):
                content = " ".join([page.get("text", "") for page in data if page.get("text")])
            else:
                content = data.get("text", "")

            documents.append({"filename": file, "content": content})
    return documents



# ------------------------------
# Chunking with Recursive Splitter
# ------------------------------
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunked_data = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "content": chunk
            })

    print(f" Created {len(chunked_data)} chunks total.")
    return chunked_data


# ------------------------------
# Save chunked JSON
# ------------------------------
def save_chunks(chunks):
    """Save each chunked file as JSON in data/chunks."""
    output_file = os.path.join(CHUNKS_DIR, "chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f" Saved chunks to {output_file}")


# ------------------------------
# Orchestrator
# ------------------------------
if __name__ == "__main__":
    print(" Loading processed files...")
    docs = load_processed_files()

    print(" Splitting documents into chunks...")
    chunks = chunk_documents(docs)

    print(" Saving chunks...")
    save_chunks(chunks)

    print(" Chunking complete!")
