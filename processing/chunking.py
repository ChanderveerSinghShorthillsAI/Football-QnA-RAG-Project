import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ArticleChunker:
    def __init__(self, json_file, chunked_file, chunk_size=500, chunk_overlap=50):
        self.json_file = json_file
        self.chunked_file = chunked_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    @staticmethod
    def clean_text(text):
        """Remove unnecessary characters and clean text."""
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces/newlines
        return text.strip()

    def load_articles(self):
        """Load articles from JSON file."""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
            return articles if articles else []
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []

    def chunk_articles(self):
        """Load, clean, split, and save articles into chunks."""
        articles = self.load_articles()
        if not articles:
            print("No articles found in JSON file!")
            return

        chunked_data = []
        for article in articles:
            title = self.clean_text(article.get("title", "No Title"))
            url = article.get("url", "No URL")
            content = self.clean_text(article.get("content", "No Content"))

            chunks = self.text_splitter.split_text(content)
            for chunk in chunks:
                chunked_data.append({"title": title, "url": url, "content": chunk})

        try:
            with open(self.chunked_file, "w", encoding="utf-8") as f:
                json.dump(chunked_data, f, indent=4, ensure_ascii=False)
            print(f"Chunking completed! Data saved in {self.chunked_file}")
        except Exception as e:
            print(f"Error saving chunked data: {e}")

if __name__ == "__main__":
    json_file = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles.json"
    chunked_file = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"
    
    chunker = ArticleChunker(json_file, chunked_file)
    chunker.chunk_articles()


