import os
import faiss
import numpy as np
import json
import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain.schema import HumanMessage
# import google.generativeai as genai

# File Paths
VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"
CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"

# Load Hugging Face API Key from environment variable
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Embedding Model
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Free model
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    model_kwargs={"temperature": 0.7, "max_length": 500}  # Adjust parameters as needed
)

def load_faiss_index():
    """Load the FAISS index."""
    index = faiss.read_index(VECTOR_DB_PATH)
    return index

def load_chunks():
    """Load chunked articles from JSON."""
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_relevant_chunks(query, top_k=3):
    """Retrieve the top_k most relevant chunks from FAISS based on the query."""
    index = load_faiss_index()
    chunks = load_chunks()

    # Convert query to embedding
    query_vector = embeddings_model.encode([query])
    query_vector = np.array(query_vector, dtype="float32")

    # Search FAISS index
    distances, indices = index.search(query_vector, top_k)
    print(f"\nindices: {indices}")
    print(f"\ndistances: {distances}")
    # Retrieve top_k relevant chunks
    retrieved_chunks = [chunks[i]["content"] for i in indices[0] if i < len(chunks)]
    
    return retrieved_chunks

LOG_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/qna_logs.json"

def log_interaction(question, generated_answer):
    """Log each user query and its generated answer."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "generated_answer": generated_answer
    }

    # Append to log file
    try:
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=4, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=4, ensure_ascii=False)



def generate_answer(query):
    """Retrieve relevant chunks and generate an answer using Mistral 7B."""
    relevant_texts = get_relevant_chunks(query, top_k=5)

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
    response = llm.invoke(prompt)
    
    log_interaction(query, response.strip());
    
    return response.strip()

if __name__ == "__main__":
    while True:
        user_query = input(" Ask a football-related question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        answer = generate_answer(user_query)
        print(f"\n Answer: {answer}\n")
