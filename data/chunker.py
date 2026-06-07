import os
import json
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "corpus_snapshot.json")

STRATEGY_MAPPING = {
    "recursive-500": os.path.join(DATA_DIR, "chunked_corpus_recursive500.json"),
    "recursive-1000": os.path.join(DATA_DIR, "chunked_corpus_recursive1000.json"),
}

# Load the raw corpus from the file
def load_raw_corpus(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Raw snapshot file not found at {file_path}.")
    print(f"📖 Loading raw snapshot from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f).get("pages", [])

# Save the chunked corpus in a new file
def save_chunked_corpus(file_path: str, chunks: list[dict], strategy: str, source_file: str) -> None:
    from datetime import datetime, UTC
    output = {
        "metadata": {
            "strategy": strategy,
            "source_file": source_file,
            "total_chunks": len(chunks),
            "generated_at": datetime.now(UTC).isoformat()
        },
        "chunks": chunks
    }
    print(f"💾 Saving {len(chunks)} chunks to: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("✅ Successfully saved.")

# Get the splitter based on the chunking strategy
def get_splitter(strategy: str):
    if strategy == "recursive-500":
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )
    elif strategy == "recursive-1000":
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def chunk_documents(raw_documents: list[dict], splitter, strategy: str) -> list[dict]:
    """Splits raw pages into chunks using the provided splitter."""
    chunks = []
    chunk_counter = 0

    for doc in raw_documents:
        text = doc.get("content", "")
        url = doc.get("source_url", "unknown_source")
        title = doc.get("title", "untitled")

        if not text.strip():
            continue

        print(f"Processing: {title} ({url})")

        for chunk_text in splitter.split_text(text):
            chunks.append({
                "chunk_id": f"id_{chunk_counter}",
                "text": chunk_text,
                "metadata": {
                    "source_url": url,
                    "title": title,
                    "strategy": strategy
                }
            })
            chunk_counter += 1

    return chunks

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Removes duplicate and suspiciously short chunks."""
    seen = set()
    deduped = []

    for chunk in chunks:
        if len(chunk["text"]) < 50:
            continue
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            deduped.append(chunk)

    removed = len(chunks) - len(deduped)
    if removed:
        print(f"   Removed {removed} duplicate chunks")

    return deduped

def main():
    parser = argparse.ArgumentParser(description="Production RAG Pipeline - Chunking Stage")
    parser.add_argument(
        "--strategy", 
        choices=list(STRATEGY_MAPPING.keys()), 
        required=True,
        help="The text chunking approach to execute."
    )
    args = parser.parse_args()
    
    output_file = STRATEGY_MAPPING[args.strategy]
    raw_documents = load_raw_corpus(INPUT_FILE)
    splitter = get_splitter(args.strategy)
    
    processed_chunks = []
    chunk_counter = 0
    
    print(f"✂️ Processing corpus using strategy: '{args.strategy}'...")
    chunks = chunk_documents(raw_documents, splitter, args.strategy)
    chunks = deduplicate_chunks(chunks)

    save_chunked_corpus(output_file, chunks, args.strategy, INPUT_FILE)

    print(f"\n✨ Done! {len(chunks)} chunks from {len(raw_documents)} pages using '{args.strategy}'")
    print(f"   Avg chunks per page: {len(chunks) / len(raw_documents):.1f}")

if __name__ == "__main__":
    main()