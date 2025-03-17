import os
import asyncio
import streamlit as st
import faiss
import numpy as np
import json
import datetime
import time
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
import google.generativeai as genai

#  Fix for "RuntimeError: no running event loop"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

#  File Paths
VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"
CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"
LOG_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/qna_logs.json"

#  Load API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")



#  Initialize Models
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,
    model_kwargs={"max_length": 500}
)

#  **Custom CSS for Styling**
st.markdown(
    """
    <style>
    body { background-color: #f8f9fa; }
    .stTextInput { border-radius: 10px; }
    .stButton>button { border-radius: 10px; background-color: #007bff; color: white; }
    .stMarkdown { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

#  **Logging User Interactions**
def log_interaction(question, generated_answer):
    """Log user queries and answers in a JSON file."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "generated_answer": generated_answer
    }

    try:
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=4, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=4, ensure_ascii=False)

#  **Load FAISS Index & Data**
def load_faiss_index():
    return faiss.read_index(VECTOR_DB_PATH)

def load_chunks():
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

#  **Retrieve Relevant Text**
def get_relevant_chunks(query, top_k=1):
    index = load_faiss_index()
    chunks = load_chunks()
    query_vector = np.array(embeddings_model.encode([query]), dtype="float32")
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i]["content"] for i in indices[0] if i < len(chunks)]

#  **Generate AI Answer**
def generate_answer(query):
    relevant_texts = get_relevant_chunks(query, top_k=3)

    if not relevant_texts:
        return " I don't have enough information."

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
    log_interaction(query, response.strip())  # Log the interaction
    return response.strip()

#  **Streamlit UI**
st.title(" Football Q&A Chatbot")
st.write(" Ask any football-related question and get an AI-generated answer!")

#  **Input Field**
user_query = st.text_input(" Type your question here:")

if st.button("Get Answer "):
    if user_query.strip():
        with st.spinner(" Searching for the best answer..."):
            start_time = time.time()
            answer = generate_answer(user_query)
            end_time = time.time()

        st.success(f" **Answer:** {answer}")
        st.write(f" Response Time: `{end_time - start_time:.2f} seconds`")
    else:
        st.error("⚠️ Please enter a question.")

#  **Sidebar for Extra Features**
st.sidebar.title(" Useful Information")
st.sidebar.markdown(" **Latest Football News**")
st.sidebar.write(" [BBC Football News](https://www.bbc.com/sport/football)")
st.sidebar.write("⚡ [ESPN Soccer](https://www.espn.com/soccer/)")

# st.sidebar.markdown(" **Popular Questions**")
st.sidebar.write("Which Liverpool goalkeeper had the best performance in a match, saving several crucial shots from opponents, earning them the title 'best in the world'?")
st.sidebar.write("What is the current status of Goutas as Cardiff boss?")
st.sidebar.write("When is Livingston expected to finalize their new investment deal?")

st.sidebar.markdown(" **View Past Queries**")
if st.sidebar.button(" Show Log"):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
            for log in logs[-5:]:  # Show last 5 queries
                st.sidebar.markdown(f" `{log['question']}` → **{log['generated_answer']}**")
    except (FileNotFoundError, json.JSONDecodeError):
        st.sidebar.warning("⚠️ No logs found!")
