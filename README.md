
#  Football Knowledge Q&A Chatbot

The **Football Knowledge Q&A Chatbot** is an **AI-powered chatbot** designed to answer **football-related questions** using **Retrieval-Augmented Generation (RAG)**. This project scrapes football news, processes the data, retrieves relevant articles using **FAISS vector search**, and generates **fact-based** answers using **Mistral 7B**.

---

##  Key Features  
 **Real-time Football Q&A** – Ask football-related questions and receive AI-generated responses.  
 **Retrieval-Augmented Generation (RAG)** – Enhances LLM-generated responses with real-world data.  
 **Efficient Data Pipeline** – Includes **scraping, chunking, vectorization, retrieval, and answer generation**.  
 **AI-Powered Answering System** – Uses **FAISS indexing** and **Mistral 7B LLM** for factual responses.  
 **Automated Evaluation & Optimization** – Measures accuracy using **BLEU, ROUGE, and F1-score**.  
 **Interactive Streamlit UI** – Web interface for easy query submission and response retrieval.  
 **Robust Logging System** – Logs user queries and chatbot responses for tracking and improvement.  

---

##  Project Structure  

```
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
│   └─ bbc_scraper.py   # Scrapes football news articles from BBC
├─ requirment.txt
```

---


## RAG Architecture
 [Image Link](https://drive.google.com/file/d/17gaYTr089MBGTnZ8UQSekHz2VQd0pxi1/view?usp=drive_link)

##  Technologies Used  

### **Programming Languages & Frameworks**  
 **Python 3.10** – Core development language  
 **Streamlit** – UI framework for chatbot interaction  
 **BeautifulSoup & Requests** – Web scraping libraries  
 **LangChain** – Advanced text chunking & retrieval  
 **FAISS** – High-speed vector search indexing  
 **Sentence Transformers** – Text embedding generation  
 **Mistral 7B (Hugging Face)** – Large language model for AI-generated answers  

### **Libraries & Tools**  
 **NumPy** – Data processing  
 **NLTK & ROUGE** – Evaluation metrics (BLEU, ROUGE, and F1-score)  
 **Hugging Face API** – LLM-based inference  

---

##  Installation & Setup  

##  Prerequisites
Ensure the following are installed:  
 **Python 3.10+**  
 **pip (Python Package Manager)**  
 **Hugging Face API Key**  
 **FAISS (for vector search)**  
 **Streamlit (for UI)**  

###  Step 1: Clone the Repository  
```bash
git clone <your-repo-url>
cd Python-Data-Scraping-Project
```

### Step 2: Install Dependencies  
```bash
pip install -r requirements.txt
```

###  Step 3: Set Up API Keys   
```bash
export HUGGINGFACE_API_KEY=your_huggingface_api_key
```

---

##  Running the Project  

### **Step 1: Scrape Football Articles**
```bash
python scrapers/bbc_scraper.py
```
**Extracts football news articles** and saves them in:  
 `data/football_articles.json`  

---

### **Step 2: Process & Chunk the Data**
```bash
python processing/chunking.py
```
**Splits long articles into smaller text chunks** and stores them in:  
 `data/football_chunks.json`  

---

### **Step 3: Generate FAISS Vector Embeddings**
```bash
python processing/vectorization.py
```
**Converts chunks into embeddings** and stores them in:  
 `data/faiss_index`  

---

### **Step 4: Run the Chatbot UI**
```bash
streamlit run UI/app.py
```
 **Ask Football Questions** via an interactive UI and get **AI-powered answers**.  

---

##  Model Evaluation & Optimization  

### **Step 5: Generate Test Cases**
```bash
python processing/generate_test_cases.py
```
Creates **test cases** and stores them in:  
 `data/football_test_cases.json`  

---

### **Step 6: Evaluate Model Performance**
```bash
python Testing_Automation/evaluate.py
```
Calculates **BLEU, ROUGE, and F1-scores** and stores results in:  
 `data/evaluation_results.json`  

---

### **Step 7: Optimize Low-Scoring Answers**
```bash
python Testing_Automation/evaluate_low_scoring_tests.py
```
Refines **low-scoring responses** and updates:  
 `data/evaluation_results.json`  

---

### **Step 8: Summarize Evaluation Results**
```bash
python Testing_Automation/summarize.py
```
Generates **performance summaries** stored in:  
 `output/output.txt`  

---

##  Evaluation Metrics & Results  

| Metric                | Before Optimization | After Optimization |
|-----------------------|--------------------|--------------------|
| **BLEU**             | 0.0643             | 0.1033             |
| **ROUGE-1**          | 0.2482             | 0.3200             |
| **ROUGE-2**          | 0.1291             | 0.1909             |
| **ROUGE-L**          | 0.2145             | 0.2799             |
| **F1 Score**         | 0.2287             | 0.3347             |
| **Low-Scoring Answers** | 818              | 658                |

---

##  Logging System  
 **User queries and AI responses** are stored in: `data/qna_logs.json`  
 **Evaluation results** are logged for tracking improvements  
 **Low-scoring answers** are identified and re-evaluated for **better performance**  

---

##  Troubleshooting  

| Issue | Possible Fix |
|-------|-------------|
| **API key errors** | Ensure `HUGGINGFACE_API_KEY` is correctly set |
| **FAISS index not found** | Run `vectorization.py` before querying |
| **Scraping not working** | BBC Sport may have changed its structure |

---

##  Key Learnings  
 **Retrieval-Augmented Generation (RAG)** – Combining retrieval and LLMs for factual responses.  
 **FAISS Vector Search** – Efficient document retrieval.  
 **Model Evaluation** – Using **BLEU, ROUGE, and F1-score**.  
 **Web Scraping & Data Processing** – Extracting, chunking, and indexing football news.  
 **Streamlit UI & Logging** – Creating an interactive chatbot with persistent logging.  

---

