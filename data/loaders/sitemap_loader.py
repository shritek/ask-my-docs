import os
import json
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from datetime import datetime, UTC
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Limit to first 5 pages")
args = parser.parse_args()

# Define output path based on your repository structure
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_snapshot.json")
SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"

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

def main():
    urls = fetch_sitemap_urls(SITEMAP_URL)
    if not urls:
        return

    # Filter for standard doc tracks (ignoring assets/images if present)
    urls = [u for u in urls if u.endswith("/")]

    if args.test:
        print(f"⚠️ Test Mode Active: Limiting scrape to first {TEST_LIMIT} pages.")
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
            time.sleep(0.5)  # polite crawling
        except Exception as e:
            print(f"⚠️ Error scraping {url}: {e}")

    output = {
        "metadata": {
            "source": SITEMAP_URL,
            "total_pages": len(corpus),
            "generated_at": datetime.now(UTC).isoformat()
        },
        "pages": corpus
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✨ Done! Saved {len(corpus)} documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()