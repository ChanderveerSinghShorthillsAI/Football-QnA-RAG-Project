import json
import os
import numpy as np
import faiss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint

class FootballAIAssistant:
    VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss/faiss_index"
    CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks/football_chunks.json"
    TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases/football_test_cases.json"
    EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results/evaluation_results.json"
    
    def __init__(self):
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.huggingface_api_key,
            temperature=0.7,
            model_kwargs={"max_length": 500}
        )
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()
    
    def load_faiss_index(self):
        return faiss.read_index(self.VECTOR_DB_PATH)
    
    def load_chunks(self):
        with open(self.CHUNKED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_relevant_chunks(self, query, top_k=1):
        query_vector = self.embeddings_model.encode([query])
        query_vector = np.array(query_vector, dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)
        return [self.chunks[i]["content"] for i in indices[0] if i < len(self.chunks)]
    
    def generate_answer(self, query):
        relevant_texts = self.get_relevant_chunks(query, top_k=1)
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
        return self.llm.invoke(prompt).strip()
    
    def calculate_bleu(self, reference, candidate):
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)
    
    def calculate_rouge(self, reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {key: scores[key].fmeasure for key in scores}
    
    def calculate_f1(self, reference, candidate):
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        common_tokens = ref_tokens.intersection(cand_tokens)
        precision = len(common_tokens) / len(cand_tokens) if cand_tokens else 0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
        return 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    
    def evaluate_test_cases(self):
        with open(self.TEST_CASES_FILE, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"Evaluating {i+1}/{len(test_cases)}: {test_case['question']}")
            answer = self.generate_answer(test_case['question'])
            bleu = self.calculate_bleu(test_case['answer'], answer)
            rouge = self.calculate_rouge(test_case['answer'], answer)
            f1 = self.calculate_f1(test_case['answer'], answer)
            results.append({
                "question": test_case['question'],
                "ground_truth": test_case['answer'],
                "generated_answer": answer,
                "bleu_score": bleu,
                "rouge_score": rouge,
                "f1_score": f1
            })
        
        with open(self.EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\nEvaluation completed. Results saved in `{self.EVALUATION_RESULTS_FILE}`.")

if __name__ == "__main__":
    assistant = FootballAIAssistant()
    assistant.evaluate_test_cases()




