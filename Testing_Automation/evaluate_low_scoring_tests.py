import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint

class FootballEvaluation:
    TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases/football_test_cases.json"
    EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results/evaluation_results.json"
    LOW_SCORE_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/low_scoring_answers/low_scoring_answers.json"

    def __init__(self):
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.huggingface_api_key,
            temperature=0.7,
            model_kwargs={"max_length": 500}
        )

    def load_test_cases(self):
        with open(self.TEST_CASES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_previous_results(self):
        try:
            with open(self.EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def load_low_scoring_cases(self):
        try:
            with open(self.LOW_SCORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def calculate_bleu(self, reference, candidate):
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)

    def calculate_rouge(self, reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge-1": scores["rouge1"].fmeasure,
            "rouge-2": scores["rouge2"].fmeasure,
            "rouge-L": scores["rougeL"].fmeasure
        }

    def calculate_f1(self, reference, candidate):
        reference_tokens = set(reference.lower().split())
        candidate_tokens = set(candidate.lower().split())
        common_tokens = reference_tokens.intersection(candidate_tokens)
        precision = len(common_tokens) / len(candidate_tokens) if candidate_tokens else 0
        recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def generate_answer(self, query):
        prompt = f"""
### Football Knowledge Assistant

You are an AI football expert. Answer the following question **ONLY using verified football sources**.
- **No assumptions or hallucinations**—stick to facts.
- **If unsure, say**: "I don't have enough information."
- Keep your response **brief, factual, and structured**.

####  Question: {query}

 **Answer:**
"""
        response = self.llm.invoke(prompt)
        return response.strip()

    def re_evaluate_bad_cases(self):
        print(" Re-Evaluating Low-Scoring Answers...")
        previous_results = self.load_previous_results()
        low_scoring_cases = self.load_low_scoring_cases()
        previous_results_dict = {r["question"]: r for r in previous_results}

        updated_results = []
        for i, test_case in enumerate(low_scoring_cases):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            print(f" Re-Evaluating {i+1}/{len(low_scoring_cases)}: {question}")
            generated_answer = self.generate_answer(question)
            bleu_score = self.calculate_bleu(ground_truth, generated_answer)
            rouge_score = self.calculate_rouge(ground_truth, generated_answer)
            f1_score = self.calculate_f1(ground_truth, generated_answer)

            updated_result = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "bleu_score": bleu_score,
                "rouge_score": rouge_score,
                "f1_score": f1_score
            }

            previous_results_dict[question] = updated_result
            updated_results.append(updated_result)

        final_results = list(previous_results_dict.values())
        with open(self.EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f"\n Re-Evaluation Completed. Results saved in `{self.EVALUATION_RESULTS_FILE}`.")

if __name__ == "__main__":
    evaluator = FootballEvaluation()
    evaluator.re_evaluate_bad_cases()



