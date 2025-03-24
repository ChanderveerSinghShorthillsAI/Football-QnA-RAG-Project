import os
import faiss
import numpy as np
import json
import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub

class FootballQnA:
    VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss/faiss_index"
    CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks/football_chunks.json"
    LOG_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/QnA_logs/qna_logs.json"
    
    def __init__(self):
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.huggingface_api_key,
            model_kwargs={"temperature": 0.7, "max_length": 500}
        )
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()
    
    def load_faiss_index(self):
        """Load the FAISS index."""
        return faiss.read_index(self.VECTOR_DB_PATH)
    
    def load_chunks(self):
        """Load chunked articles from JSON."""
        with open(self.CHUNKED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_relevant_chunks(self, query, top_k=3):
        """Retrieve the top_k most relevant chunks from FAISS based on the query."""
        query_vector = self.embeddings_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        print(f"\nindices: {indices}")
        print(f"\ndistances: {distances}")
        return [self.chunks[i]["content"] for i in indices[0] if i < len(self.chunks)]
    
    def log_interaction(self, question, generated_answer):
        """Log each user query and its generated answer."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "generated_answer": generated_answer
        }
        try:
            with open(self.LOG_FILE, "r+", encoding="utf-8") as f:
                logs = json.load(f)
                logs.append(log_entry)
                f.seek(0)
                json.dump(logs, f, indent=4, ensure_ascii=False)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=4, ensure_ascii=False)
    
    def generate_answer(self, query):
        """Retrieve relevant chunks and generate an answer using Mistral 7B."""
        relevant_texts = self.get_relevant_chunks(query, top_k=5)
        context = "\n\n".join(relevant_texts)
        
        prompt = f"""
### Football Knowledge Assistant

You are an expert in football. Answer the following question **ONLY using the provided articles**.
- If the answer is **not found**, respond with: **"I don't have enough information."**
- Keep your response **concise** and **factual**.
- If multiple players/teams are mentioned, compare them **briefly**.

####  Articles:
{context}

####  Question: {query}

 **Answer:**
"""
        response = self.llm.invoke(prompt).strip()
        self.log_interaction(query, response)
        return response
    
    def run(self):
        while True:
            user_query = input("Ask a football-related question (or type 'exit' to quit): ")
            if user_query.lower() == "exit":
                break
            print(f"\n Answer: {self.generate_answer(user_query)}\n")

if __name__ == "__main__":
    FootballQnA().run()




