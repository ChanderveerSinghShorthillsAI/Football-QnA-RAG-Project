import os
import asyncio
import streamlit as st
import faiss
import numpy as np
import json
import datetime
import time
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Fix for "RuntimeError: no running event loop"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

class FootballQABot:
    VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss/faiss_index"
    CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks/football_chunks.json"
    LOG_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/QnA_logs/qna_logs.json"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self):
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()

    def load_faiss_index(self):
        """Load the FAISS index."""
        return faiss.read_index(self.VECTOR_DB_PATH)

    def load_chunks(self):
        """Load chunked articles from JSON."""
        with open(self.CHUNKED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

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

    def get_relevant_chunks(self, query, top_k=1):
        """Retrieve the top_k most relevant chunks from FAISS based on the query."""
        query_vector = np.array(self.embeddings_model.encode([query]), dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)
        return [self.chunks[i]["content"] for i in indices[0] if i < len(self.chunks)]

    def generate_answer(self, query):
        """Retrieve relevant chunks and generate an answer using OpenAI API."""
        relevant_texts = self.get_relevant_chunks(query, top_k=3)
        if not relevant_texts:
            return "I don't have enough information."

        context = "\n\n".join(relevant_texts)
        prompt = f"""
        ### Football Knowledge Assistant
        You are an expert in football. Answer the following question **ONLY using the provided articles**.
        - If the answer is **not found**, respond with: **"I don't have enough information."**
        - Keep your response **concise** and **factual**.
        - If multiple players/teams are mentioned, compare them **briefly**.

        #### Articles:
        {context}

        #### Question: {query}

        **Answer:**
        """
        response = self.client.chat.completions.create(
            model="gpt-4",  # Or use "gpt-3.5-turbo" if preferred
            messages=[
                {"role": "system", "content": "You are a football knowledge assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        generated_answer = response.choices[0].message.content.strip()
        self.log_interaction(query, generated_answer)
        return generated_answer

# Initialize the chatbot
bot = FootballQABot()

# Streamlit UI
st.title("⚽ Football Q&A Chatbot")
st.write("Ask any football-related question and get an AI-generated answer!")

user_query = st.text_input("Type your question here:")
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Searching for the best answer..."):
            start_time = time.time()
            answer = bot.generate_answer(user_query)
            end_time = time.time()
        st.success(f"**Answer:** {answer}")
        st.write(f"Response Time: `{end_time - start_time:.2f} seconds`")
    else:
        st.error("⚠️ Please enter a question.")

# Sidebar Features
st.sidebar.title("📚 Useful Information")
st.sidebar.markdown("**Latest Football News**")
st.sidebar.write("[BBC Football News](https://www.bbc.com/sport/football)")
st.sidebar.write("[ESPN Soccer](https://www.espn.com/soccer/)")

st.sidebar.markdown("**📊 View Past Queries**")
if st.sidebar.button("Show Log"):
    try:
        with open(bot.LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
            for log in logs[-5:]:
                st.sidebar.markdown(f"`{log['question']}` → **{log['generated_answer']}**")
    except (FileNotFoundError, json.JSONDecodeError):
        st.sidebar.warning("⚠️ No logs found!")
