# Description: This script generates high-quality football test cases using Mistral-7B.

import os
import json
import time
import random
import re
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer, util

class FootballTestCaseGenerator:
    ARTICLES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles/football_articles.json"
    TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases/football_test_cases_ragas.json"
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    TEMPERATURE = 0.7
    MAX_LENGTH = 500
    SIMILARITY_THRESHOLD = 0.78  

    def __init__(self):
        if not self.HUGGINGFACE_API_KEY:
            raise ValueError("⚠️ HUGGINGFACE_API_KEY environment variable not set.")
        self.llm = HuggingFaceEndpoint(
            repo_id=self.MODEL_REPO_ID,
            huggingfacehub_api_token=self.HUGGINGFACE_API_KEY,
            temperature=self.TEMPERATURE,
            model_kwargs={"max_length": self.MAX_LENGTH}
        )
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_articles(self):
        """Load football articles from JSON file."""
        try:
            with open(self.ARTICLES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading articles: {e}")
            return []

    def load_existing_test_cases(self):
        """Load existing test cases to avoid duplicates."""
        if os.path.exists(self.TEST_CASES_FILE):
            try:
                with open(self.TEST_CASES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON in test cases file.")
        return []

    def generate_test_case(self, article, test_case_number):
        """Generate a high-quality football test case using Mistral-7B."""
        question_types = ["factual", "comparative", "fact-checking", "multi-step", "hypothetical"]
        selected_type = random.choice(question_types)

        prompt = f"""
        ### Football Test Case Generator

        You are an AI that generates **high-quality football-related test cases**.
        The questions must **test retrieval abilities** and require deep understanding of articles.

        ## Rules:
        - **Question must require detailed knowledge**, not common trivia.
        - **Answer must be factual, precise, and unique**.
        - **Output must be in valid JSON format**.
        - **Question type:** {selected_type}

        ## Article Title: {article['title']}
        ## Article Content:
        {article['content'][:700]}  #  Using 700 characters for better context

        Example JSON format:
        ```json
        {{
            "question": "Which club has the longest unbeaten streak in Premier League history?",
            "answer": "Arsenal holds the record with 49 unbeaten games from 2003-2004."
        }}
        ```

        Generate a **high-quality test case** based on the article.
        """
        try:
            print(f"🧪 Generating Test Case {test_case_number} → {article['title']}")  #  ADDED LINE
            response = self.llm.invoke(prompt)
            json_match = re.search(r"{\s*\"question\":\s*\".*?\",\s*\"answer\":\s*\".*?\"\s*}", response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            test_case = json.loads(response)
            if "question" in test_case and "answer" in test_case:
                return test_case
            else:
                print(f"⚠️ Invalid JSON format for: {article['title']}")
                return None
        except json.JSONDecodeError as json_err:
            print(f"⚠️ JSON decoding error for '{article['title']}': {json_err}. Response: {response}")
            return None
        except Exception as e:
            print(f"⚠️ Error generating test case for '{article['title']}': {e}")
            return None

    def is_similar(self, new_question, existing_questions, threshold=None):
        """Check if the generated question is too similar to existing ones."""
        if not existing_questions:
            return False

        threshold = threshold or self.SIMILARITY_THRESHOLD  # ✅ Use default or override
        new_embedding = self.similarity_model.encode(new_question, convert_to_tensor=True)
        existing_embeddings = self.similarity_model.encode(list(existing_questions), convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, existing_embeddings)

        return any(score > threshold for score in similarity_scores[0])

    def generate_test_cases(self, num_attempts=100):
        """Generate multiple high-quality test cases while ensuring uniqueness and appending to JSON."""
        articles = self.load_articles()
        existing_cases = self.load_existing_test_cases()
        existing_questions = {case["question"] for case in existing_cases}

        new_test_cases = []
        attempts = 0
        max_attempts = num_attempts * 3  #  Increasing attempts dynamically

        while len(new_test_cases) < num_attempts and attempts < max_attempts:
            attempts += 1
            article = random.choice(articles)
            test_case_number = len(new_test_cases) + 1  #  ADDED TEST CASE NUMBER
            test_case = self.generate_test_case(article, test_case_number)

            if test_case:
                question = test_case["question"]
                if question not in existing_questions and not self.is_similar(question, existing_questions):
                    new_test_cases.append(test_case)
                    existing_questions.add(question)

            time.sleep(0.4)  #  Faster execution

        if not new_test_cases:
            print("⚠️ No new test cases were generated. Try increasing the dataset size or adjusting parameters.")
            return

        #  Append new test cases instead of overwriting
        all_test_cases = existing_cases + new_test_cases

        with open(self.TEST_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_test_cases, f, indent=4, ensure_ascii=False)

        print(f"\n Added {len(new_test_cases)} new test cases. Total test cases now: {len(all_test_cases)}.")

if __name__ == "__main__":
    generator = FootballTestCaseGenerator()
    generator.generate_test_cases(num_attempts=400)












    
    




