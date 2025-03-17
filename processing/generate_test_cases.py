import os
import json
import time
import re
import random
from langchain_huggingface import HuggingFaceEndpoint
import spacy
from sentence_transformers import SentenceTransformer, util

# Configuration
ARTICLES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles.json"
TEST_CASES_FILE = "/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_test_cases.json"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
TEMPERATURE = 0.85
MAX_LENGTH = 300

def initialize_llm():
    """Initializes the language model."""
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
    return HuggingFaceEndpoint(
        repo_id=MODEL_REPO_ID,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=TEMPERATURE,
        max_length=MAX_LENGTH,
    )

def extract_keywords(text, nlp):
    """Extract key phrases using Named Entity Recognition (NER)."""
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]]
    return keywords[:5]  # Take top 5 keywords

def load_article_topics():
    """Loads diverse topics from football_articles.json."""
    try:
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        nlp = spacy.load("en_core_web_sm")  # Load NLP model for entity recognition
        all_topics = []
        
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            keywords = extract_keywords(content, nlp)
            all_topics.extend(keywords)
            # print(keywords)
        return list(set(all_topics))  # Ensure uniqueness
    except Exception as e:
        print(f"⚠️ Error loading article topics: {e}")
        return []

def generate_test_case(llm, topic):
    """Generates a football-related test case (question and answer)."""
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
        response = llm.invoke(prompt)

        # Extract JSON using regex
        json_match = re.search(r"{\s*\"question\":\s*\".*?\",\s*\"answer\":\s*\".*?\"\s*}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)  # Extract JSON content

        test_case = json.loads(response)  # Parse JSON
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

# Load pre-trained similarity model
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def is_similar(new_question, existing_questions, threshold=0.8):
    """Check if new_question is semantically similar to any existing question."""
    new_embedding = similarity_model.encode(new_question, convert_to_tensor=True)
    existing_embeddings = similarity_model.encode(existing_questions, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(new_embedding, existing_embeddings)

    # If any similarity score exceeds the threshold, consider it a duplicate
    return any(score > threshold for score in similarity_scores[0])

def generate_test_cases(llm, num_cases=100):
    """Generates multiple test cases and appends them to a JSON file, ensuring uniqueness."""
    test_cases = []
    skipped_cases = []

    # Load existing test cases (if any)
    existing_cases = []
    if os.path.exists(TEST_CASES_FILE):
        with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
            try:
                existing_cases = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Existing test cases file is not valid JSON. Starting with empty list.")

    # Create a set of existing questions for quick duplicate checking
    existing_questions = {case["question"] for case in existing_cases}

    for i in range(num_cases):
        topic = random.choice(load_article_topics())  # Load dynamic topics
        print(f" Generating test case {i + 1}/{num_cases} on '{topic}'...")

        test_case = generate_test_case(llm, topic)
        if test_case:
            question = test_case["question"]
            
            if question not in existing_questions :
                test_cases.append(test_case)
                existing_questions.add(question)  # Prevent duplicates
            else:
                print(f"⚠️ Duplicate or similar question found: '{question}'")
        else:
            skipped_cases.append(topic)

        time.sleep(2)  # Reduce delay

    # Save the final unique test cases
    all_cases = existing_cases + test_cases
    with open(TEST_CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=4, ensure_ascii=False)

    print(f"\n Successfully saved {len(test_cases)} new unique test cases.")
    print(f"⚠️ Skipped {len(skipped_cases)} topics due to errors.")
    print(f" Test cases saved to `{TEST_CASES_FILE}`")

def main():
    """Main function to execute the test case generation."""
    try:
        llm = initialize_llm()
        generate_test_cases(llm, num_cases=200)
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    
    




