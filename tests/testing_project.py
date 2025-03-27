import pytest
import json
import os
import faiss
import numpy as np
from unittest.mock import patch, MagicMock , mock_open
from ragas.evaluation import SingleTurnSample
import warnings
warnings.filterwarnings("ignore")

#  Importing from your project folders
from scrapers.bbc_scraper import BBCFootballScraper
from processing.chunking import ArticleChunker
from processing.vectorization import FAISSIndexer
from processing.retrieval import FootballQnA

#  Test BBC Scraper
@pytest.fixture
def scraper():
    return BBCFootballScraper(limit=2)

def test_get_article_links(scraper):
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '<a class="ssrcss-sxweo-PromoLink" href="/sport/football/articles/test-article"></a>'
        
        scraper.get_article_links()
        assert len(scraper.article_links) > 0
        assert "https://www.bbc.com/sport/football/articles/test-article" in scraper.article_links

def test_scrape_article(scraper):
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '<h1>Test Title</h1><article><p>Test Content</p></article>'
        
        result = scraper.scrape_article("https://www.bbc.com/sport/football/articles/test-article")
        assert result["title"] == "Test Title"
        assert "Test Content" in result["content"]

#  Test Chunking
@pytest.fixture
def chunker(tmp_path):
    json_file = tmp_path / "articles.json"
    chunked_file = tmp_path / "chunked.json"

    data = [{"title": "Test Article", "content": "This is a long test article." * 100}]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return ArticleChunker(json_file, chunked_file)

def test_chunk_articles(chunker):
    chunker.chunk_articles()
    with open(chunker.chunked_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    assert len(chunks) > 0
    assert "Test Article" in chunks[0]["title"]

#  Test FAISS Indexing
@pytest.fixture
def indexer(tmp_path):
    chunked_file = tmp_path / "chunks.json"
    vector_db_path = tmp_path / "faiss_index"

    data = [{"content": "This is a test chunk."}]
    with open(chunked_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return FAISSIndexer(str(chunked_file), str(vector_db_path), use_openai=False)

def test_create_faiss_index(indexer):
    indexer.create_faiss_index()
    index = faiss.read_index(indexer.vector_db_path)
    
    assert index.ntotal > 0

def test_load_faiss_index(indexer):
    indexer.create_faiss_index()
    index = indexer.load_faiss_index()

    assert index is not None

#  Test Retrieval
@pytest.fixture
def qna(tmp_path):
    chunked_file = tmp_path / "chunks.json"
    vector_db_path = tmp_path / "faiss_index"
    log_file = tmp_path / "logs.json"

    dim = 384
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, str(vector_db_path))

    data = [{"content": "This is a test chunk."}]
    with open(chunked_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Mock the HuggingFaceHub dependency
    with patch("processing.retrieval.HuggingFaceHub") as MockHub:
        mock_llm = MagicMock()
        mock_llm.return_value = "Mocked response"
        MockHub.return_value = mock_llm
        yield FootballQnA()

# def test_get_relevant_chunks(qna):
#     chunks = qna.get_relevant_chunks("test", top_k=1)
#     assert isinstance(chunks, list)

def test_log_interaction(qna, tmp_path):
    qna.LOG_FILE = tmp_path / "logs.json"
    qna.log_interaction("Test question?", "Test answer.")

    with open(qna.LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    assert len(logs) > 0
    assert logs[0]["question"] == "Test question?"


from processing.generate_test_cases import FootballTestCaseGenerator

@pytest.fixture
def generator(tmp_path):
    # Create a generator instance
    generator = FootballTestCaseGenerator()
    generator.ARTICLES_FILE = tmp_path / "football_articles.json"
    generator.TEST_CASES_FILE = tmp_path / "football_test_cases.json"

    # Mock articles data
    articles = [{"title": "Test Article", "content": "This is a test article about football."}]
    with open(generator.ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f)

    return generator

def test_load_articles(generator):
    articles = generator.load_articles()
    assert len(articles) > 0
    assert articles[0]["title"] == "Test Article"

def test_generate_test_case(generator):
    article = {"title": "Test Article", "content": "This is a test article about football."}
    test_case = generator.generate_test_case(article, 1)
    assert "question" in test_case
    assert "answer" in test_case

def test_generate_test_cases(generator):
    generator.generate_test_cases(num_attempts=2)
    with open(generator.TEST_CASES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    assert len(test_cases) > 0



from UI.app import FootballQABot

@pytest.fixture
def bot(tmp_path):
    # Create a bot instance
    bot = FootballQABot()
    bot.VECTOR_DB_PATH = str(tmp_path / "faiss_index")
    bot.CHUNKED_FILE = tmp_path / "football_chunks.json"
    bot.LOG_FILE = tmp_path / "qna_logs.json"

    # Create a mock FAISS index
    dim = 384
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, bot.VECTOR_DB_PATH)

    # Mock chunks file
    chunks = [{"content": "This is a test chunk."}]
    with open(bot.CHUNKED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    return bot

def test_load_faiss_index(bot):
    index = bot.load_faiss_index()
    assert index.is_trained or index.ntotal == 0

def test_load_chunks(bot):
    chunks = bot.load_chunks()
    assert len(chunks) > 0
    assert "content" in chunks[0]

def test_log_interaction(bot, tmp_path):
    bot.LOG_FILE = tmp_path / "qna_logs.json"
    bot.log_interaction("Test question?", "Test answer.")

    with open(bot.LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    assert len(logs) > 0
    assert logs[0]["question"] == "Test question?"




from Testing_Automation.evaluate import FootballAIAssistant


@pytest.fixture
def assistant():
    """Fixture to create a FootballAIAssistant instance with mock dependencies."""
    with patch("Testing_Automation.evaluate.FootballAIAssistant.load_faiss_index"), \
        patch("Testing_Automation.evaluate.FootballAIAssistant.load_chunks"):
        return FootballAIAssistant()



#  Test: Generate Answer

def test_generate_answer(assistant):
    """Test answer generation with successful API response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a generated answer."

    with patch.object(assistant, "get_relevant_chunks", return_value=["Chunk 1", "Chunk 2"]), \
         patch.object(assistant.client.chat.completions, "create", return_value=mock_response):
        
        result = assistant.generate_answer("Test question?")
        assert result == "This is a generated answer."


def test_generate_answer_no_chunks(assistant):
    """Test answer generation when no relevant chunks are found."""
    with patch.object(assistant, "get_relevant_chunks", return_value=[]):
        result = assistant.generate_answer("Unknown question?")
        assert result == "I don't have enough information."



#  Test: Load Existing Results

def test_load_existing_results(assistant):
    """Test loading existing evaluation results."""
    mock_data = [{"question": "Test Q1", "result": "Correct"}]
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
        results = assistant.load_existing_results()
        assert results == mock_data


def test_load_existing_results_empty(assistant):
    """Test loading results when file is empty or invalid."""
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="")), \
         patch("json.load", side_effect=json.JSONDecodeError("Error", "doc", 0)):
        results = assistant.load_existing_results()
        assert results == []



#  Test: Evaluate Test Cases with RAGAs

def test_evaluate_test_cases_with_ragas(assistant):
    """Test evaluation using RAGAs."""
    mock_test_cases = [{"question": "Test Q1", "answer": "Answer 1"}]
    mock_relevant_chunks = ["Chunk 1"]
    mock_generated_answer = "Generated answer"

    with patch.object(assistant, "load_existing_results", return_value=[]), \
         patch.object(assistant, "get_relevant_chunks", return_value=mock_relevant_chunks), \
         patch.object(assistant, "generate_answer", return_value=mock_generated_answer), \
         patch("json.load", return_value=mock_test_cases), \
         patch.object(assistant, "_evaluate_and_save_batch") as mock_save_batch:

        assistant.evaluate_test_cases_with_ragas(batch_size=1)
        assert mock_save_batch.called



#  Test: _Evaluate and Save Batch

def test_evaluate_and_save_batch(assistant):
    """Test batch evaluation and result saving."""
    mock_results = {
        "faithfulness": [0.9],
        "context_precision": [0.8],
        "answer_correctness": [0.85],
    }
    mock_sample = SingleTurnSample(
        user_input="Test question",
        retrieved_contexts=["Chunk 1"],
        response="Generated answer",
        reference="Ground truth"
    )

    mock_dataset_list = [mock_sample]
    with patch("ragas.evaluate", return_value=mock_results), \
         patch.object(assistant, "save_results") as mock_save:
        assistant._evaluate_and_save_batch(mock_dataset_list, [])

    assert mock_save.called



#  Test: Save Results

def test_save_results(assistant):
    """Test saving results to file."""
    mock_results = [{"question": "Test Q1", "result": "Correct"}]

    with patch("builtins.open", mock_open()) as mock_file:
        assistant.save_results(mock_results)

    mock_file.assert_called_once_with(assistant.EVALUATION_RESULTS_FILE, "w", encoding="utf-8")
