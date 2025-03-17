import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# File paths
JSON_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles.json"
CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"

def clean_text(text):
    """Remove unnecessary characters and clean text."""
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces/newlines
    return text.strip()

def chunk_articles():
    """Load scraped articles, clean them, and split into chunks."""
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
            print("hello");

        if not articles:
            print(" No articles found in JSON file!")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        chunked_data = []
        for article in articles:
            title = clean_text(article.get("title", "No Title"))
            url = article.get("url", "No URL")
            content = clean_text(article.get("content", "No Content"))

            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                chunked_data.append({"title": title, "url": url, "content": chunk})

        with open(CHUNKED_FILE, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=4, ensure_ascii=False)

        print(f" Chunking completed! Data saved in {CHUNKED_FILE}")

    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    chunk_articles()
