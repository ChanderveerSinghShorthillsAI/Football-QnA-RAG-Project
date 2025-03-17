Project Overview

The Football Knowledge Q&A Chatbot is an AI-powered chatbot designed to answer football-related questions using Retrieval-Augmented Generation (RAG). This project scrapes football news, processes the data, retrieves relevant articles using FAISS vector search, and generates accurate answers using Mistral 7B.

 Key Features:
 Real-time Football Q&A: Users can ask football-related questions and receive AI-generated responses.
 Data Processing Pipeline: Includes scraping, chunking, vectorization, retrieval, and answer generation.
 Evaluation & Optimization: Utilizes BLEU, ROUGE, and F1-score for benchmarking accuracy.
 Logging & UI: Logs user interactions and provides a sleek Streamlit UI for a better experience.


Project Structure


Python Data Scrapping Project
├─ Output                   # Stores evaluation summaries and performance reports
│  └─ output.txt
├─ README.md
├─ Testing_Automation        # Scripts for evaluation, optimization, and summarization
│  ├─ __init__.py
│  ├─ evaluate.py           # Evaluates chatbot responses using BLEU, ROUGE, and F1-score
│  ├─ evaluate_low_scoring_tests.py     # Re-evaluates incorrect answers to improve accuracy
│  └─ summarize.py            # Summarizes evaluation results before and after optimization
├─ UI
│  ├─ __pycache__
│  │  └─ test_faiss.cpython-310.pyc
│  └─ app.py           # Streamlit-based UI for the chatbot
├─ __init__.py
├─ data        # Stores raw articles, processed chunks, vectors, test cases, logs
│  ├─ evaluation_result_before_enhancement.json   # evaluation result before optimisation in prompt
│  ├─ evaluation_results.json  # evaluation result after optimisation in prompt
│  ├─ faiss_index              # stores the vector embedding of the chunks
│  ├─ faiss_index.py
│  ├─ faiss_vector.json        # visual representation of the embeddings 
│  ├─ football_articles copy.json  
│  ├─ football_articles.csv    
│  ├─ football_articles.json     # Scraped football articles 
│  ├─ football_chunks.json       # Stored the chunks of the data
│  ├─ football_test_cases.json   # generated test cases of the data
│  ├─ low_scoring_answers.json   # low scoring test cases before enhancement
│  ├─ low_scoring_answers_after_enhancement.json  # low scoring test cases after enhancement
│  ├─ qna_logs.json        #logs stored when the user interacts with the model
│  ├─ rough.json 
│  └─ rough.py
├─ processing
│  ├─ __init__.py
│  ├─ __pycache__
│  │  └─ retrieval.cpython-310.pyc
│  ├─ chunking.py     # Splits articles into smaller text chunks
│  ├─ generate_test_cases.py       # Generates test cases for evaluation
│  ├─ retrieval.py     # Retrieves relevant chunks from FAISS for a query
│  └─ vectorization.py    # Converts text into FAISS embeddings
└─ scrapers
   └─ bbc_scraper.py   # Scrapes football news articles from BBC



Data Processing Pipeline
Step 1: Web Scraping
File: bbc_scraper.py

 Scrapes football news from BBC Sport and stores articles in data/football_articles.json.
 Extracts titles, URLs, and article content for further processing.

Step 2: Text Chunking
File: chunking.py

 Splits long articles into smaller, meaningful chunks for better retrieval.
 Uses LangChain’s RecursiveCharacterTextSplitter for chunking.
 Stores chunked data in data/football_chunks.json.

Step 3: Vectorization (FAISS Indexing)
File: vectorization.py

 Converts text chunks into numerical embeddings using Sentence Transformers.
 Uses FAISS (Facebook AI Similarity Search) for fast similarity-based retrieval.
 Stores the FAISS index in data/faiss_index.

Step 4: Chatbot UI (Streamlit App)
File: app.py

streamlit run app.py

 Launches a web-based UI where users can ask football-related questions.
 Retrieval Process:
 User asks a question.
 The system retrieves relevant articles using FAISS vector search.
 The Mistral 7B model generates an answer using retrieved information.
 The response is displayed in the UI, and the query is logged.
 Model Evaluation & Optimization

Step 5: Generate Test Cases
File: generate_test_cases.py

 Creates football-related test cases using AI.
 Ensures test cases are unique, factual, and diverse.
 Stores generated test cases in data/football_test_cases.json.

Step 6: Evaluate Model Performance
File: evaluate.py

 Runs 1000+ test cases and evaluates chatbot responses.
 Calculates BLEU, ROUGE, and F1-scores for benchmarking.
 Stores evaluation results in data/evaluation_results.json.


Step 7: Optimize Low-Scoring Answers
File: evaluate_low_scoring.py


 Re-evaluates incorrect/low-scoring answers to improve accuracy.
 Uses an enhanced prompt to refine responses.
 Saves updated results in data/evaluation_results.json.


Step 8: Summarize Results
File: summarize.py

 Generates before & after optimization performance summaries.
 Stores the final results in output/output.txt.

Evaluation Metrics
Metric	Before Optimization	| After Optimization
--------------------------------------------------
BLEU	    0.0643	        |     0.1033
ROUGE-1	    0.2482	        |     0.3200
ROUGE-2	    0.1291	        |     0.1909
ROUGE-L	    0.2145	        |     0.2799
F1 Score	0.2287	        |     0.3347
Low-Scoring Answers	818	    |     658


Logging System
 User interactions (queries & responses) are logged in data/qna_logs.json.
 Performance evaluations are logged and stored.
 Re-evaluation results are saved for continuous improvement.

Key Learnings

 Retrieval-Augmented Generation (RAG): Combining retrieval and LLMs for factual responses.
 FAISS Vector Search: Efficient similarity search for document retrieval.
 Model Evaluation: Using BLEU, ROUGE, and F1-score for accuracy assessment.
 Web Scraping & Data Processing: Extracting, chunking, and indexing football news.
 Streamlit UI & Logging: Creating an interactive chatbot with persistent logging.
