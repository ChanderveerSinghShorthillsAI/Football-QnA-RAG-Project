import json
import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_correctness
from ragas.evaluation import EvaluationDataset, SingleTurnSample


class FootballAIAssistant:
    VECTOR_DB_PATH = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/faiss/faiss_index"
    CHUNKED_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_chunks/football_chunks.json"
    TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases/football_test_cases_ragas.json"
    EVALUATION_RESULTS_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/evaluation_results/evaluation_result_ragas.json"

    def __init__(self):
        """Initialize models and load data"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(" OPENAI_API_KEY environment variable not set.")

        self.client = OpenAI(api_key=self.openai_api_key)
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()

    def load_faiss_index(self):
        """Load FAISS vector database (handle missing index)."""
        if not os.path.exists(self.VECTOR_DB_PATH):
            raise FileNotFoundError(f" FAISS Index Not Found: {self.VECTOR_DB_PATH}")
        return faiss.read_index(self.VECTOR_DB_PATH)

    def load_chunks(self):
        """Load pre-processed document chunks."""
        if not os.path.exists(self.CHUNKED_FILE):
            raise FileNotFoundError(f" Chunks File Not Found: {self.CHUNKED_FILE}")
        with open(self.CHUNKED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_relevant_chunks(self, query, top_k=3):
        """Retrieve top_k most relevant text chunks for a query."""
        query_vector = self.embeddings_model.encode([query])
        query_vector = np.array(query_vector, dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)
        return [self.chunks[i]["content"] for i in indices[0] if i < len(self.chunks)]

    def generate_answer(self, query, max_retries=3):
        """Generate AI-based answer using OpenAI GPT-4-turbo with retry mechanism."""
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

        ####  Articles:
        {context}

        ####  Question: {query}

        **Answer:**
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7,
                    max_tokens=350
                )
                time.sleep(0.5)  # Delay to prevent rate limits
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f" API Error: {e}, Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
        return "API Error: Unable to generate answer."

    def evaluate_test_cases_with_ragas(self, batch_size=1, delay_between_batches=5):
        """Evaluate chatbot responses using RAGAs with error handling and retry."""
        with open(self.TEST_CASES_FILE, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        # Load existing results to avoid duplication
        existing_results = self.load_existing_results()

        processed_questions = {result["question"] for result in existing_results}
        dataset_list = []
        total_cases = len(test_cases)
        print(f" Preparing {total_cases} test cases for RAGAs evaluation...")

        if not test_cases:
            print(" No test cases found! Exiting evaluation.")
            return

        for i, test_case in enumerate(test_cases):
            user_input = test_case["question"]

            # Skip already processed test cases
            if user_input in processed_questions:
                continue

            retrieved_contexts = self.get_relevant_chunks(user_input, top_k=3)
            ground_truth_answer = test_case["answer"]
            model_response = self.generate_answer(user_input)

            # Prepare test case sample
            dataset_list.append(SingleTurnSample(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                response=model_response,
                reference=ground_truth_answer
            ))

            print(f" Processed test case {i + 1}/{total_cases}")

            # Evaluate and save after each batch
            if len(dataset_list) >= batch_size or i == total_cases - 1:
                self._evaluate_and_save_batch(dataset_list, existing_results)
                dataset_list = []  # Clear dataset after saving
                time.sleep(delay_between_batches)  # Delay to avoid rate limits

        print(f"\n Evaluation completed using RAGAs. Results saved in `{self.EVALUATION_RESULTS_FILE}`.")

    def load_existing_results(self):
        """Load existing evaluation results to avoid duplication."""
        if os.path.exists(self.EVALUATION_RESULTS_FILE):
            with open(self.EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _evaluate_and_save_batch(self, dataset_list, existing_results, max_retries=3):
        """Evaluate a batch and append results to file with retry on failure."""
        if not dataset_list:
            return

        dataset = EvaluationDataset(dataset_list)

        for attempt in range(max_retries):
            try:
                eval_results = evaluate(
                    dataset,
                    metrics=[faithfulness, context_precision, answer_correctness],
                )
                break
            except TimeoutError as e:
                print(f"⚠️ TimeoutError: Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
        else:
            print(" Evaluation failed after retries.")
            return

        # Save results to existing results
        for i, sample in enumerate(dataset.samples):
            existing_results.append({
                "question": sample.user_input,
                "ground_truth": sample.reference,
                "generated_answer": sample.response,
                "faithfulness_score": eval_results["faithfulness"][i],
                "context_precision_score": eval_results["context_precision"][i],
                "correctness_score": eval_results["answer_correctness"][i],
            })

        # Save intermediate results incrementally
        self.save_results(existing_results)

        print(f"Saved {len(dataset_list)} test cases to `{self.EVALUATION_RESULTS_FILE}`.")

    def save_results(self, results):
        """Save evaluation results to a file."""
        with open(self.EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    assistant = FootballAIAssistant()
    assistant.evaluate_test_cases_with_ragas(batch_size=1, delay_between_batches=5)
