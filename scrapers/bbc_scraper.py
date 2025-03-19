import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

class BBCFootballScraper:
    BASE_URL = "https://www.bbc.com/sport/football"
    CATEGORY_PAGES = [
        BASE_URL,  
        "https://www.bbc.com/sport/football/premier-league",
        "https://www.bbc.co.uk/sport/football/womens",
        "https://www.bbc.co.uk/sport/football/womens-super-league",
        "https://www.bbc.co.uk/sport/football/championship",
        "https://www.bbc.co.uk/sport/football/league-one",
        "https://www.bbc.co.uk/sport/football/league-two",
        "https://www.bbc.co.uk/sport/football/national-league",
        "https://www.bbc.co.uk/sport/football/fa-cup",
        "https://www.bbc.co.uk/sport/football/league-cup",
        "https://www.bbc.co.uk/sport/football/scottish",
        "https://www.bbc.co.uk/sport/football/scottish-premiership",
        "https://www.bbc.co.uk/sport/football/scottish-championship",
        "https://www.bbc.co.uk/sport/football/scottish-league-one",
        "https://www.bbc.co.uk/sport/football/scottish-league-two",
        "https://www.bbc.co.uk/sport/football/scottish-cup",
        "https://www.bbc.co.uk/sport/football/scottish-league-cup",
        "https://www.bbc.co.uk/sport/football/scottish-challenge-cup",
        "https://www.bbc.co.uk/sport/football/welsh",
        "https://www.bbc.co.uk/sport/football/irish",
        "https://www.bbc.co.uk/sport/football/european",
        "https://www.bbc.co.uk/sport/football/champions-league",
        "https://www.bbc.co.uk/sport/football/europa-league",
        "https://www.bbc.co.uk/sport/football/world-cup",
        "https://www.bbc.co.uk/sport/football/womens-world-cup",
        "https://www.bbc.co.uk/sport/football/european-championship",
        "https://www.bbc.co.uk/sport/football/womens-european-championship",
        "https://www.bbc.co.uk/sport/africa"
    ]  

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    def __init__(self, limit=None):
        self.limit = limit
        self.article_links = set()
        self.scraped_articles = []

    def get_article_links(self):
        """Fetch all article links from multiple BBC Football category pages."""
        for page in self.CATEGORY_PAGES:
            print(f" Scraping category page: {page}")
            response = requests.get(page, headers=self.HEADERS)
            if response.status_code != 200:
                print(f" Failed to fetch {page}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.select("a.ssrcss-sxweo-PromoLink")

            for article in articles:
                href = article.get("href")
                if href and href.startswith("/sport/football/articles"):
                    self.article_links.add(urljoin("https://www.bbc.com", href))

            time.sleep(2)  # Avoid rate limiting

        print(f" Extracted {len(self.article_links)} unique article links.")

    def scrape_article(self, url):
        """Scrape the article title and content from a given BBC Football article URL."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            if response.status_code != 200:
                print(f" Failed to fetch {url}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title_tag = soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No Title"

            # Extract article body
            paragraphs = soup.select("article p")
            content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

            return {"url": url, "title": title, "content": content}

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching {url}: {e}")
            return None

    def scrape_articles(self):
        """Scrape multiple articles from BBC Football."""
        self.get_article_links()

        for idx, link in enumerate(list(self.article_links)[:self.limit] if self.limit else self.article_links):
            print(f" Scraping article {idx+1}/{len(self.article_links)}: {link}")
            article_data = self.scrape_article(link)
            if article_data:
                self.scraped_articles.append(article_data)
            time.sleep(3)  # Pause to avoid getting blocked

        return self.scraped_articles

    def save_articles(self, filepath):
        """Save scraped articles to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.scraped_articles, f, indent=4, ensure_ascii=False)
        print(f" Scraping completed! {len(self.scraped_articles)} articles saved in `{filepath}`")

if __name__ == "__main__":
    scraper = BBCFootballScraper(limit=None)  # Scrape all articles
    articles = scraper.scrape_articles()
    scraper.save_articles("/home/shtlp_0060/Desktop/Python Data Scrapping Project/data/football_articles.json")



