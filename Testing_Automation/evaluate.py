import json
import os
import sys
import numpy as np
import faiss
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint


VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"
CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"

#  Load Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

#  Initialize Sentence Transformer Embedding Model
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

#  Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

#  Fix HuggingFaceEndpoint issue: Pass `temperature` explicitly
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,  # Explicitly pass this
    model_kwargs={"max_length": 500}  # Fix warning
)

def load_faiss_index():
    """Load the FAISS index."""
    return faiss.read_index(VECTOR_DB_PATH)

def load_chunks():
    """Load chunked articles from JSON."""
    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_relevant_chunks(query, top_k=1):
    """Retrieve the most relevant chunk from FAISS for the query."""
    index = load_faiss_index()
    chunks = load_chunks()

    # Convert query to embedding
    query_vector = embeddings_model.encode([query])
    query_vector = np.array(query_vector, dtype="float32")

    # Search FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Retrieve the most relevant chunk
    retrieved_chunks = [chunks[i]["content"] for i in indices[0] if i < len(chunks)]
    
    return retrieved_chunks

def generate_answer(query):
    """Retrieve relevant chunk and generate an answer using Mistral 7B."""
    relevant_texts = get_relevant_chunks(query, top_k=1)  # Retrieve **only one** most relevant answer

    if not relevant_texts:
        return "I don't have enough information."

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
    return response.strip()



# File Paths
TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases.json"
EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results.json"

# Load test cases
def load_test_cases():
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate):
    smooth = SmoothingFunction().method1  # Apply smoothing to avoid zero scores
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)



from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  #  Correct ROUGE types
    scores = scorer.score(reference, candidate)
    return {
        "rouge-1": scores["rouge1"].fmeasure,
        "rouge-2": scores["rouge2"].fmeasure,
        "rouge-L": scores["rougeL"].fmeasure
    }


# F1 Score Calculation
def calculate_f1(reference, candidate):
    reference_tokens = set(reference.lower().split())
    candidate_tokens = set(candidate.lower().split())

    common_tokens = reference_tokens.intersection(candidate_tokens)
    precision = len(common_tokens) / len(candidate_tokens) if candidate_tokens else 0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Run Evaluation
def evaluate_test_cases():
    test_cases = load_test_cases()
    results = []

    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        ground_truth = test_case["answer"]

        print(f" Evaluating {i+1}/{len(test_cases)}: {question}")

        generated_answer = generate_answer(question)  # Get the system's response

        bleu_score = calculate_bleu(ground_truth, generated_answer)
        rouge_score = calculate_rouge(ground_truth, generated_answer)
        f1_score = calculate_f1(ground_truth, generated_answer)

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "bleu_score": bleu_score,
            "rouge_score": rouge_score,
            "f1_score": f1_score
        }

        results.append(result)

    # Save results to JSON
    with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n Evaluation completed. Results saved in `{EVALUATION_RESULTS_FILE}`.")

if __name__ == "__main__":
    evaluate_test_cases()


# import json
# import os
# import sys
# import time
# import numpy as np
# import faiss
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer
# from langchain_huggingface import HuggingFaceEndpoint
# from requests.exceptions import RequestException

# # ✅ File Paths
# VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss_index"
# CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks.json"
# TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases.json"
# EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results.json"

# # ✅ Hugging Face API Keys (Switch if exhausted)
# HF_API_KEYS = [
#     os.getenv("HUGGINGFACE_API_KEY"),  # Primary API Key
#     os.getenv("HUGGINGFACE_API_KEY_ALT")  # Secondary API Key (Set this in your environment variables)
# ]
# CURRENT_KEY_INDEX = 0
# REQUEST_COUNT = 0  # Track API requests

# def switch_api_key():
#     """Switch API key after 900 requests."""
#     global CURRENT_KEY_INDEX, REQUEST_COUNT
#     CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(HF_API_KEYS)
#     os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEYS[CURRENT_KEY_INDEX]
#     print(f"🔄 Switching to API Key {CURRENT_KEY_INDEX + 1}")
#     REQUEST_COUNT = 0  # Reset request count

# # ✅ Initialize Sentence Transformer Embedding Model
# embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Initialize Hugging Face LLM
# def initialize_llm():
#     return HuggingFaceEndpoint(
#         endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
#         huggingfacehub_api_token=HF_API_KEYS[CURRENT_KEY_INDEX],
#         temperature=0.7,
#         model_kwargs={"max_length": 500}
#     )

# llm = initialize_llm()

# def load_faiss_index():
#     """Load the FAISS index."""
#     return faiss.read_index(VECTOR_DB_PATH)

# def load_chunks():
#     """Load chunked articles from JSON."""
#     with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
#         return json.load(f)

# def get_relevant_chunks(query, top_k=1):
#     """Retrieve the most relevant chunk from FAISS for the query."""
#     index = load_faiss_index()
#     chunks = load_chunks()

#     query_vector = embeddings_model.encode([query])
#     query_vector = np.array(query_vector, dtype="float32")

#     distances, indices = index.search(query_vector, top_k)
#     retrieved_chunks = [chunks[i]["content"] for i in indices[0] if i < len(chunks)]

#     return retrieved_chunks

# def generate_answer(query, max_retries=3):
#     """Retrieve relevant chunk and generate an answer using Mistral 7B."""
#     global REQUEST_COUNT

#     relevant_texts = get_relevant_chunks(query, top_k=1)
#     if not relevant_texts:
#         return "I don't have enough information."

#     context = "\n\n".join(relevant_texts)

#     prompt = f"""
# ### Football Knowledge Assistant

# You are an expert in football. Answer the following question **ONLY using the provided articles**.
# - If the answer is **not found**, respond with: **"I don't have enough information."**
# - Keep your response **concise** and **factual**.
# - If multiple players/teams are mentioned, compare them **briefly**.

# #### 📜 Articles:
# {context}

# #### ❓ Question: {query}

# 💡 **Answer:**
# """
#     for attempt in range(max_retries):
#         try:
#             if REQUEST_COUNT >= 900:
#                 switch_api_key()  # Switch API after 900 requests
#                 REQUEST_COUNT = 0

#             response = llm.invoke(prompt)
#             REQUEST_COUNT += 1
#             return response.strip()
#         except RequestException as e:
#             print(f"⚠️ API request failed (attempt {attempt+1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(5)
#             else:
#                 return "API error: Unable to generate answer."

# def load_test_cases():
#     """Load test cases from JSON."""
#     with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
#         return json.load(f)

# def calculate_bleu(reference, candidate):
#     """Calculate BLEU score with smoothing."""
#     smooth = SmoothingFunction().method1
#     return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)

# def calculate_rouge(reference, candidate):
#     """Calculate ROUGE scores."""
#     scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#     scores = scorer.score(reference, candidate)
#     return {
#         "rouge-1": scores["rouge1"].fmeasure,
#         "rouge-2": scores["rouge2"].fmeasure,
#         "rouge-L": scores["rougeL"].fmeasure
#     }

# def calculate_f1(reference, candidate):
#     """Calculate F1 score."""
#     reference_tokens = set(reference.lower().split())
#     candidate_tokens = set(candidate.lower().split())

#     common_tokens = reference_tokens.intersection(candidate_tokens)
#     precision = len(common_tokens) / len(candidate_tokens) if candidate_tokens else 0
#     recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0

#     if precision + recall == 0:
#         return 0
#     return 2 * (precision * recall) / (precision + recall)

# def evaluate_test_cases():
#     """Evaluate test cases automatically."""
#     test_cases = load_test_cases()
#     results = []

#     for i, test_case in enumerate(test_cases):
#         question = test_case["question"]
#         ground_truth = test_case["answer"]

#         print(f"📝 Evaluating {i+1}/{len(test_cases)}: {question}")

#         generated_answer = generate_answer(question)

#         bleu_score = calculate_bleu(ground_truth, generated_answer)
#         rouge_score = calculate_rouge(ground_truth, generated_answer)
#         f1_score = calculate_f1(ground_truth, generated_answer)

#         result = {
#             "question": question,
#             "ground_truth": ground_truth,
#             "generated_answer": generated_answer,
#             "bleu_score": bleu_score,
#             "rouge_score": rouge_score,
#             "f1_score": f1_score
#         }

#         results.append(result)

#     with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     print(f"\n✅ Evaluation completed. Results saved in `{EVALUATION_RESULTS_FILE}`.")

# if __name__ == "__main__":
#     evaluate_test_cases()
