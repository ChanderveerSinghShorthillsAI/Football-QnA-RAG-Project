import os
import json
import time
import re
import random
import spacy
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer, util

class FootballTestCaseGenerator:
    ARTICLES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles.json"
    TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases.json"
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    TEMPERATURE = 0.85
    MAX_LENGTH = 300

    def __init__(self):
        if not self.HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
        self.llm = HuggingFaceEndpoint(
            repo_id=self.MODEL_REPO_ID,
            huggingfacehub_api_token=self.HUGGINGFACE_API_KEY,
            temperature=self.TEMPERATURE,
            max_length=self.MAX_LENGTH,
        )
        self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, text):
        doc = self.nlp(text)
        keywords = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]]
        return keywords[:5]

    def load_article_topics(self):
        try:
            with open(self.ARTICLES_FILE, "r", encoding="utf-8") as f:
                articles = json.load(f)
            all_topics = []
            for article in articles:
                keywords = self.extract_keywords(article.get("content", ""))
                all_topics.extend(keywords)
            return list(set(all_topics))
        except Exception as e:
            print(f"⚠️ Error loading article topics: {e}")
            return []

    def generate_test_case(self, topic):
        prompt = f"""
### Football Test Case Generator

You are an AI that generates unique football-related **test cases**.
- The test case must include a **question** about **{topic}**.
- The question **must be unique** and should not be a common trivia question.
- The **answer must be concise, factual, and correct**.
- Format the output as valid JSON.
- **Do not repeat questions** already in the dataset.
- Include details like player stats, club history, or match records.

**Example Format:**
```json
{{
    "question": "Which club had the longest unbeaten streak in the English Premier League?",
    "answer": "Arsenal holds the record with 49 unbeaten games from 2003-2004."
}}

Generate a unique question & answer for: {topic} """
        try:
            print(f" Querying model for topic: {topic}")
            response = self.llm.invoke(prompt)
            json_match = re.search(r"{\s*\"question\":\s*\".*?\",\s*\"answer\":\s*\".*?\"\s*}", response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            test_case = json.loads(response)
            if "question" in test_case and "answer" in test_case:
                print(f" Test case generated for: {topic}")
                return test_case
            else:
                print(f"⚠️ Invalid JSON format for topic: {topic}")
                return None
        except json.JSONDecodeError as json_err:
            print(f"⚠️ JSON decoding error for topic '{topic}': {json_err}. response: {response}")
            return None
        except Exception as e:
            print(f"⚠️ Error generating test case for topic '{topic}': {e}")
            return None

    def is_similar(self, new_question, existing_questions, threshold=0.8):
        new_embedding = self.similarity_model.encode(new_question, convert_to_tensor=True)
        existing_embeddings = self.similarity_model.encode(existing_questions, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, existing_embeddings)
        return any(score > threshold for score in similarity_scores[0])

    def generate_test_cases(self, num_cases=100):
        test_cases = []
        skipped_cases = []
        existing_cases = []
        if os.path.exists(self.TEST_CASES_FILE):
            with open(self.TEST_CASES_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_cases = json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ Warning: Existing test cases file is not valid JSON. Starting with empty list.")
        existing_questions = {case["question"] for case in existing_cases}
        for i in range(num_cases):
            topic = random.choice(self.load_article_topics())
            print(f" Generating test case {i + 1}/{num_cases} on '{topic}'...")
            test_case = self.generate_test_case(topic)
            if test_case:
                question = test_case["question"]
                if question not in existing_questions:
                    test_cases.append(test_case)
                    existing_questions.add(question)
                else:
                    print(f"⚠️ Duplicate or similar question found: '{question}'")
            else:
                skipped_cases.append(topic)
            time.sleep(2)
        all_cases = existing_cases + test_cases
        with open(self.TEST_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, indent=4, ensure_ascii=False)
        print(f"\n Successfully saved {len(test_cases)} new unique test cases.")
        print(f"⚠️ Skipped {len(skipped_cases)} topics due to errors.")
        print(f" Test cases saved to `{self.TEST_CASES_FILE}`")

    def run(self):
        try:
            self.generate_test_cases(num_cases=200)
        except ValueError as ve:
            print(f"Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generator = FootballTestCaseGenerator()
    generator.run()







    
    




