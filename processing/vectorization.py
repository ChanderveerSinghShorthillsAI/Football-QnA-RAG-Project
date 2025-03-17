import json
import os
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.docstore.document import Document

# File Paths
CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"
VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"

# Choose Embedding Model (OpenAI or Sentence Transformers)
USE_OPENAI = False  # Change to True if using OpenAI

if USE_OPENAI:
    embeddings = OpenAIEmbeddings()
else:
    from sentence_transformers import SentenceTransformer
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"embeddings_model: {embeddings_model}")

def load_chunks():
    """Load chunked articles from JSON."""
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def create_faiss_index():
    """Convert text chunks into embeddings and store in FAISS."""
    articles = load_chunks()
    if not articles:
        print("⚠️ No chunked articles found!")
        return

    texts = [article["content"] for article in articles]
    
    # Generate embeddings
    if USE_OPENAI:
        vectors = embeddings.embed_documents(texts)
    else:
        vectors = embeddings_model.encode(texts)
        print(f"\nvectors: {vectors}")

    # Convert to NumPy array
    vectors = np.array(vectors, dtype="float32")
    print(f"\nvectors after array : {vectors}")
    # Create FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance for similarity search
    print(f"\nindex before adding vectors: {index}")
    index.add(vectors)
    print(f"\nindex after adding vectors: {index}")

    # Save the index
    faiss.write_index(index, VECTOR_DB_PATH)
    print(f" FAISS index saved to {VECTOR_DB_PATH}")

def load_faiss_index():
    """Load the FAISS index."""
    if not os.path.exists(VECTOR_DB_PATH):
        print("⚠️ FAISS index not found! Run vectorization first.")
        return None
    print(f"\nVECTOR_DB_PATH: {VECTOR_DB_PATH}");
    return faiss.read_index(VECTOR_DB_PATH)

if __name__ == "__main__":
    create_faiss_index()
