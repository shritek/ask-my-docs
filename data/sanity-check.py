import json

with open("data/chunked_corpus_recursive500.json") as f:
    data = json.load(f)

chunks = data["chunks"]

# Check shortest and longest chunks
lengths = [len(c["text"]) for c in chunks]
print(f"Total chunks: {len(chunks)}")
print(f"Avg length:   {sum(lengths)/len(lengths):.0f} chars")
print(f"Min length:   {min(lengths)} chars")
print(f"Max length:   {max(lengths)} chars")

# Flag suspiciously short chunks
short = [c for c in chunks if len(c["text"]) < 50]
print(f"\nChunks under 50 chars: {len(short)}")
for c in short[:5]:
    print(f"  '{c['text']}'")