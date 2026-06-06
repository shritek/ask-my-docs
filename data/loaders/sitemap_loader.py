import os
import json
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, UTC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Limit to first 5 pages")
args = parser.parse_args()

# Define output path based on your repository structure
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_snapshot.json")
SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"

# Set to True to test with a small batch of pages first
#LIMIT_PAGES = True 
TEST_LIMIT = 5

def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Fetches the XML sitemap and extracts the documentation URLs."""
    print(f"📥 Fetching sitemap: {sitemap_url}")
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        # Sitemaps use this specific XML namespace
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(f"✅ Found {len(urls)} total URLs in sitemap.")
        return urls
    except Exception as e:
        print(f"❌ Failed to parse sitemap: {e}")
        return []

def clean_html_content(html_content: str, url: str) -> dict:
    """Parses HTML and strips out nav bars, headers, and footers."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # MkDocs Material template puts the actual content inside <article>
    main_content = soup.find("article", class_="md-content__inner")
    
    if not main_content:
        main_content = soup.find("main")
        
    if not main_content:
        return None

    # Strip code-copy buttons and scripts so they don't pollute embeddings
    for element in main_content.find_all(["script", "style", "button"]):
        element.decompose()

    # Get structural text and clean up whitespace
    text = main_content.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    
    title_element = soup.find("h1")
    title = title_element.get_text().strip() if title_element else "FastAPI Docs"

    return {
        "title": title,
        "source_url": url,
        "content": cleaned_text
    }

def chunk_documents(corpus: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for doc in corpus:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunks.append({
                "title": doc["title"],
                "source_url": doc["source_url"],
                "chunk_index": i,
                "content": split
            })
    return chunks

def main():
    urls = fetch_sitemap_urls(SITEMAP_URL)
    if not urls:
        return

    # Filter for standard doc tracks (ignoring assets/images if present)
    urls = [u for u in urls if u.endswith("/")]

    if args.test:
        print(f"⚠️ Test Mode Active: args.help")
        urls = urls[:TEST_LIMIT]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    corpus = []

    print("🚀 Scraping page contents...")
    for i, url in enumerate(urls):
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                doc = clean_html_content(res.text, url)
                if doc:
                    corpus.append(doc)
                    print(f" [{i+1}/{len(urls)}] Scraped: {doc['title']}")
        except Exception as e:
            print(f"⚠️ Error scraping {url}: {e}")

    chunks = chunk_documents(corpus)

    output = {
        "metadata": {
            "source": SITEMAP_URL,
            "total_pages": len(corpus),
            "total_chunks": len(chunks),
            "chunk_size": 500,
            "chunk_overlap": 50,
            "generated_at": datetime.now(UTC).isoformat()
        },
        "chunks": chunks
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✨ Done! Saved {len(chunks)} chunks from {len(corpus)} pages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()