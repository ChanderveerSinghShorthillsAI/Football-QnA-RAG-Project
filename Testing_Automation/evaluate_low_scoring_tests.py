import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import datetime
import time
import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint

#  File Paths
TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases.json"
EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results.json"
LOW_SCORE_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/low_scoring_answers.json"

#  Load Model
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.7,
    model_kwargs={"max_length": 500}
)

#  Load Test Cases
def load_test_cases():
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

#  Load Previous Evaluation Results
def load_previous_results():
    try:
        with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

#  Load Low-Scoring Test Cases (BLEU < 0.1)
def load_low_scoring_cases():
    try:
        with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

#  BLEU Score Calculation (With Smoothing)
def calculate_bleu(reference, candidate):
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)

#  ROUGE Score Calculation
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge-1": scores["rouge1"].fmeasure,
        "rouge-2": scores["rouge2"].fmeasure,
        "rouge-L": scores["rougeL"].fmeasure
    }

#  F1 Score Calculation
def calculate_f1(reference, candidate):
    reference_tokens = set(reference.lower().split())
    candidate_tokens = set(candidate.lower().split())

    common_tokens = reference_tokens.intersection(candidate_tokens)
    precision = len(common_tokens) / len(candidate_tokens) if candidate_tokens else 0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

#  Generate Answer (With Updated Prompt)
def generate_answer(query):
    """Retrieve relevant chunk and generate an answer using Mistral 7B."""
    prompt = f"""
### Football Knowledge Assistant

You are an AI football expert. Answer the following question **ONLY using verified football sources**.
- **No assumptions or hallucinations**—stick to facts.
- **If unsure, say**: "I don't have enough information."
- Keep your response **brief, factual, and structured**.

####  Question: {query}

 **Answer:**
"""
    response = llm.invoke(prompt)
    return response.strip()

#  Selectively Re-Evaluate Low-Scoring Cases
def re_evaluate_bad_cases():
    print(" Re-Evaluating Low-Scoring Answers...")
    
    previous_results = load_previous_results()
    low_scoring_cases = load_low_scoring_cases()
    test_cases = load_test_cases()

    #  Convert previous results into a dict (for fast updates)
    previous_results_dict = {r["question"]: r for r in previous_results}

    #  Iterate only over low-scoring cases
    updated_results = []
    for i, test_case in enumerate(low_scoring_cases):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f" Re-Evaluating {i+1}/{len(low_scoring_cases)}: {question}")

        generated_answer = generate_answer(question)  # Get updated AI response

        #  Recalculate Scores
        bleu_score = calculate_bleu(ground_truth, generated_answer)
        rouge_score = calculate_rouge(ground_truth, generated_answer)
        f1_score = calculate_f1(ground_truth, generated_answer)

        updated_result = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "bleu_score": bleu_score,
            "rouge_score": rouge_score,
            "f1_score": f1_score
        }

        #  Update only the low-scoring answers
        previous_results_dict[question] = updated_result
        updated_results.append(updated_result)

    #  Merge Updated Results with Good Answers
    final_results = list(previous_results_dict.values())

    #  Save Updated Results
    with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n Re-Evaluation Completed. Results saved in `{EVALUATION_RESULTS_FILE}`.")

if __name__ == "__main__":
    re_evaluate_bad_cases()
