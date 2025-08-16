import trafilatura
import requests
import logging


def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            # Fallback to requests if trafilatura fails
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            downloaded = response.text
        
        text = trafilatura.extract(downloaded)
        return text if text else "Could not extract content from URL"
    
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return f"Error scraping content: {str(e)}"
