"""
Web research module for searching and extracting content from the web.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from loguru import logger
import time
from urllib.parse import quote_plus


class WebResearcher:
    """Web research agent for gathering information."""

    def __init__(
        self,
        max_results: int = 3,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ):
        """
        Initialize web researcher.

        Args:
            max_results: Maximum number of results per query
            timeout: Request timeout in seconds
            user_agent: User agent string
        """
        self.max_results = max_results
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        logger.info("WebResearcher initialized")

    def search_duckduckgo(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Override default max results

        Returns:
            List of result dictionaries with 'title', 'url', 'snippet'
        """
        max_res = max_results or self.max_results

        try:
            # DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Parse results
            for result_div in soup.find_all('div', class_='result')[:max_res]:
                try:
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')

                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                except Exception as e:
                    logger.warning(f"Error parsing result: {e}")
                    continue

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def fetch_page_content(self, url: str, max_length: int = 5000) -> str:
        """
        Fetch and extract main content from a webpage.

        Args:
            url: Page URL
            max_length: Maximum content length

        Returns:
            Extracted text content
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()

            # Get text
            text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "..."

            logger.info(f"Fetched {len(text)} chars from {url}")
            return text

        except Exception as e:
            logger.error(f"Error fetching page content: {e}")
            return ""

    def research_topic(self, query: str, fetch_content: bool = False) -> Dict:
        """
        Research a topic by searching and optionally fetching page content.

        Args:
            query: Search query
            fetch_content: Whether to fetch full page content

        Returns:
            Dictionary with 'query', 'results', and optionally 'content'
        """
        logger.info(f"Researching: {query}")
        start_time = time.time()

        # Search
        results = self.search_duckduckgo(query)

        research_data = {
            'query': query,
            'results': results,
            'timestamp': time.time()
        }

        # Optionally fetch content from top result
        if fetch_content and results:
            top_result = results[0]
            content = self.fetch_page_content(top_result['url'])
            research_data['content'] = content

        research_time = time.time() - start_time
        logger.info(f"Research completed in {research_time:.2f}s")

        return research_data

    def research_multiple(self, queries: List[str], fetch_content: bool = False) -> List[Dict]:
        """
        Research multiple queries.

        Args:
            queries: List of search queries
            fetch_content: Whether to fetch page content

        Returns:
            List of research result dictionaries
        """
        results = []

        for query in queries:
            research_data = self.research_topic(query, fetch_content)
            results.append(research_data)

            # Small delay to be respectful
            time.sleep(1)

        logger.info(f"Completed research for {len(queries)} queries")
        return results

    def summarize_research(self, research_data: Dict) -> str:
        """
        Create a text summary of research data.

        Args:
            research_data: Research data dictionary

        Returns:
            Formatted text summary
        """
        query = research_data.get('query', 'Unknown')
        results = research_data.get('results', [])

        summary = f"Research: {query}\n"
        summary += f"Found {len(results)} results\n\n"

        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   {result['url']}\n"
            if result.get('snippet'):
                summary += f"   {result['snippet']}\n"
            summary += "\n"

        return summary


def test_researcher():
    """Test web researcher."""
    logger.info("Testing web researcher...")

    researcher = WebResearcher(max_results=3)

    # Test queries
    queries = [
        "Python machine learning tutorials",
        "Real-time audio processing",
        "DeepSeek AI model"
    ]

    for query in queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Query: {query}")
        logger.info('='*50)

        research_data = researcher.research_topic(query, fetch_content=False)

        for i, result in enumerate(research_data['results'], 1):
            logger.info(f"\n{i}. {result['title']}")
            logger.info(f"   URL: {result['url']}")
            logger.info(f"   Snippet: {result['snippet'][:100]}...")

        # Test content fetching for first result
        if research_data['results']:
            logger.info(f"\n--- Fetching content from top result ---")
            content = researcher.fetch_page_content(research_data['results'][0]['url'])
            logger.info(f"Content preview: {content[:200]}...")

        time.sleep(2)


if __name__ == "__main__":
    from loguru import logger
    import os
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/research_test.log")
    test_researcher()
