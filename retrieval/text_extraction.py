# Article text extraction
from newspaper import Article

def extract_article_text(url):
    """
    Downloads and extracts article text from a URL using newspaper3k.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text

if __name__ == "__main__":
    test_url = "https://www.reuters.com/technology/meta-launches-new-ai-models-2025-08-05/"
    print(extract_article_text(test_url)[:500])
