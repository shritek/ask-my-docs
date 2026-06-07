import os
import json
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "corpus_snapshot.json")

STRATEGY_MAPPING = {
    "recursive-500": os.path.join(DATA_DIR, "chunked_corpus_recursive500.json"),
    "recursive-1000": os.path.join(DATA_DIR, "chunked_corpus_recursive1000.json"),
    "semantic": os.path.join(DATA_DIR, "chunked_corpus_semantic.json"),
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
        
    elif strategy == "semantic":
        print("🧠 Initializing semantic splitter with 2-pass architecture...")
        
        # Pre-split to fit within nomic-embed-text context window (8192 tokens)
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
            length_function=len
        )
        
        # Semantic splitter — splits on embedding similarity shifts
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        
        # Post-split cap — prevents oversized semantic chunks from exceeding
        # vector store index limits
        post_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len
        )
        
        # Returned as a dict — unpacked by chunk_semantic() during processing
        return {
            "pre": pre_splitter,
            "semantic": semantic_splitter,
            "post": post_splitter
        }
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def chunk_semantic(text: str, splitters: dict) -> list[str]:
    """2-pass semantic chunking: pre-split → semantic → post-cap."""
    pre_splitter = splitters["pre"]
    semantic_splitter = splitters["semantic"]
    post_splitter = splitters["post"]

    chunks = []
    for segment in pre_splitter.split_text(text):
        for fragment in semantic_splitter.split_text(segment):
            if len(fragment) > 3000:
                chunks.extend(post_splitter.split_text(fragment))
            else:
                chunks.append(fragment)
    return chunks

def chunk_documents(raw_documents: list[dict], splitter, strategy: str) -> list[dict]:
    """Splits raw pages into chunks using the provided splitter configuration."""
    chunks = []
    chunk_counter = 0

    for doc in raw_documents:
        text = doc.get("content", "")
        url = doc.get("source_url", "unknown_source")
        title = doc.get("title", "untitled")

        if not text.strip():
            continue

        print(f"Processing: {title} ({url})")

        if strategy == "semantic":
            split_texts = chunk_semantic(text, splitter)
        else:
            split_texts = splitter.split_text(text)

        for chunk_text in split_texts:
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