import argparse
import os
import json
import random
from config.settings import DEFAULT_LLM, LLMModel
from helpers.llm_factory import get_llm_model

INPUT_FILE = "data/corpus_snapshot.json"
OUTPUT_FILE = "evaluation/test_set.json"
TARGET_COUNT = 50

def generate_qa_pair(llm, page):
    content = page.get("content", "")
    title = page.get("title", "Untitled")
    url = page.get("source_url", "Unknown")

    print(f"Generating pair for: {title}...")
    prompt = f"""
    Generate one high-quality Question and Answer pair based on the following documentation page.
    The question should be specific, challenging but answerable using ONLY the provided content.
    The answer should be a comprehensive explanation derived from the text.

    Page Title: {title}
    URL: {url}
    Content:
    {content}

    Return your response in strictly valid JSON format as follows:
    {{
        "question": "The specific question",
        "answer": "The comprehensive answer based on the text"
    }}
    """

    try:
        response = llm.invoke(prompt)
        # Extract JSON from response (handling potential markdown wrappers)
        text = response.content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        pair = json.loads(text.strip())
        print(f"Successfully parsed JSON for {title}")
        return pair
    except Exception as e:
        print(f"Error generating pair for {title}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Production RAG Pipeline - Test Set Generation")
    parser.add_argument(
        "--generator",
        type=str,
        default=DEFAULT_LLM.value,
        choices=[e.value for e in LLMModel],
        help="The model to use for generating the Q&A pairs."
    )
    args = parser.parse_args()

    # Load raw corpus
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if not pages:
        print("No pages found in corpus.")
        return

    # Sample pages to ensure diversity across the documentation
    sample_size = min(len(pages), TARGET_COUNT * 2) # Sample more than needed to filter failures
    sampled_pages = random.sample(pages, sample_size)

    llm = get_llm_model(LLMModel(args.generator))
    test_set = []

    print(f"Generating {TARGET_COUNT} Q&A pairs using model: {args.generator}...")

    for i, page in enumerate(sampled_pages):
        if len(test_set) >= TARGET_COUNT:
            break

        pair = generate_qa_pair(llm, page)
        if pair and "question" in pair and "answer" in pair:
            # Add metadata to the test set for debugging
            pair["source_url"] = page.get("source_url")
            pair["title"] = page.get("title")
            test_set.append(pair)
            print(f"[{len(test_set)}/{TARGET_COUNT}] Generated pair for: {page.get('title')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Successfully created evaluation test set at {OUTPUT_FILE} with {len(test_set)} pairs.")

if __name__ == "__main__":
    main()
