import json
import os
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class FAISSIndexer:
    """Class to handle FAISS indexing for document embeddings."""
    
    def __init__(self, chunked_file, vector_db_path, use_openai=False):
        self.chunked_file = chunked_file
        self.vector_db_path = vector_db_path
        self.use_openai = use_openai
        
        if self.use_openai:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"Embeddings model loaded: {self.embeddings_model}")

    def load_chunks(self):
        """Load chunked articles from JSON file."""
        try:
            with open(self.chunked_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return []

    def create_faiss_index(self):
        """Convert text chunks into embeddings and store in FAISS."""
        articles = self.load_chunks()
        if not articles:
            print("⚠️ No chunked articles found!")
            return

        texts = [article["content"] for article in articles]
        
        # Generate embeddings
        if self.use_openai:
            vectors = self.embeddings.embed_documents(texts)
        else:
            vectors = self.embeddings_model.encode(texts)
            print(f"\nGenerated vectors: {vectors}")

        # Convert to NumPy array
        vectors = np.array(vectors, dtype="float32")

        # Create FAISS index
        index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance for similarity search
        index.add(vectors)

        # Save the index
        faiss.write_index(index, self.vector_db_path)
        print(f"FAISS index saved to {self.vector_db_path}")

    def load_faiss_index(self):
        """Load the FAISS index from disk."""
        if not os.path.exists(self.vector_db_path):
            print("⚠️ FAISS index not found! Run vectorization first.")
            return None
        return faiss.read_index(self.vector_db_path)

if __name__ == "__main__":
    indexer = FAISSIndexer(
        chunked_file="/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks/football_chunks.json",
        vector_db_path="/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss/faiss_index",
        use_openai=False
    )
    indexer.create_faiss_index()



