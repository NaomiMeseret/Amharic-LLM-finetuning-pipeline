# scrape.py
# Collects additional Amharic text from news websites
# to supplement the OPUS Bible corpus training data

import requests
from bs4 import BeautifulSoup
import json
import time
import os

SOURCES = [
    "https://www.bbc.com/amharic",
    "https://www.voaamharic.com",
]

def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        texts = [p.get_text(strip=True) for p in paragraphs
                 if len(p.get_text(strip=True)) > 30]
        return texts
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return []

def save_texts(texts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text},
                    ensure_ascii=False) + "\n")
    print(f"Saved {len(texts)} texts to {output_path}")

if __name__ == "__main__":
    all_texts = []
    for url in SOURCES:
        print(f"Scraping {url}...")
        texts = scrape_page(url)
        all_texts.extend(texts)
        time.sleep(2)  # be respectful to servers

    save_texts(all_texts, "data/raw/scraped_amharic.jsonl")
    print(f"Total collected: {len(all_texts)} text samples")