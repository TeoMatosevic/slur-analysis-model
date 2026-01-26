"""
Web scraper for collecting comments from Croatian news portals.
Targets Index.hr, 24sata.hr, and other Croatian news sites.
"""

import json
import time
import logging
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.options import Options
    from webdriver_manager.firefox import GeckoDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not available. Install with: pip install selenium webdriver-manager")

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Comment:
    """Represents a single comment from a news portal."""
    id: str
    text: str
    source: str
    article_url: str
    article_title: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None
    parent_id: Optional[str] = None
    likes: Optional[int] = None
    replies_count: Optional[int] = None
    scraped_at: str = ""

    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()
        if not self.id:
            # Generate ID from content hash
            content = f"{self.source}:{self.article_url}:{self.text}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]


class BaseScraper:
    """Base class for news portal scrapers."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        rate_limit: float = 3.0,
        use_selenium: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _init_selenium(self):
        """Initialize Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available")

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--width=1920')
        options.add_argument('--height=1080')

        service = Service(GeckoDriverManager().install())
        self.driver = webdriver.Firefox(service=service, options=options)
        logger.info("Selenium WebDriver (Firefox) initialized")

    def _close_selenium(self):
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _get_page(self, url: str, use_selenium: bool = False) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        try:
            if use_selenium and self.use_selenium:
                if not self.driver:
                    self._init_selenium()
                self.driver.get(url)
                time.sleep(2)  # Wait for JavaScript to load
                html = self.driver.page_source
            else:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                html = response.text

            return BeautifulSoup(html, 'lxml')

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        time.sleep(self.rate_limit)

    def scrape_comments(self, url: str) -> List[Comment]:
        """Scrape comments from a single article. Override in subclasses."""
        raise NotImplementedError

    def scrape_article_urls(self, section_url: str, max_articles: int = 50) -> List[str]:
        """Get article URLs from a section page. Override in subclasses."""
        raise NotImplementedError

    def save_comments(self, comments: List[Comment], filename: str):
        """Save comments to JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, 'a', encoding='utf-8') as f:
            for comment in comments:
                f.write(json.dumps(asdict(comment), ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(comments)} comments to {filepath}")


class IndexHrScraper(BaseScraper):
    """Scraper for Index.hr portal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://www.index.hr"
        self.source = "index.hr"

    def scrape_article_urls(
        self,
        section: str = "vijesti",
        max_articles: int = 50
    ) -> List[str]:
        """Get article URLs from a section."""
        urls = []
        # Index.hr uses /vijesti for main news, subsections like /vijesti/hrvatska
        section_url = f"{self.base_url}/{section}"

        soup = self._get_page(section_url)
        if not soup:
            return urls

        # Find article links - Index.hr pattern: /vijesti/clanak/title/id.aspx
        article_links = soup.find_all('a', href=True)
        for link in article_links:
            href = link['href']
            # Match Index.hr article pattern
            if '/clanak/' in href and '.aspx' in href:
                full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                if full_url not in urls:
                    urls.append(full_url)
                    if len(urls) >= max_articles:
                        break

        logger.info(f"Found {len(urls)} article URLs from {section}")
        return urls

    def scrape_comments(self, url: str) -> List[Comment]:
        """Scrape comments from an Index.hr article."""
        comments = []

        # First get the article page to extract commentThreadId
        soup = self._get_page(url, use_selenium=False)
        if not soup:
            return comments

        # Get article title
        title_elem = soup.find('h1')
        article_title = title_elem.get_text(strip=True) if title_elem else None

        # Find commentThreadId in the page JavaScript
        import re
        script_tags = soup.find_all('script')
        comment_thread_id = None
        for script in script_tags:
            if script.string and 'commentThreadId' in str(script.string):
                match = re.search(r'commentThreadId=(\d+)', str(script.string))
                if match:
                    comment_thread_id = match.group(1)
                    break

        if not comment_thread_id:
            logger.warning(f"Could not find commentThreadId for {url}")
            return comments

        # Fetch comments from AJAX endpoint
        comments_url = f"{self.base_url}/ajax-noindex/display-article-comments?commentThreadId={comment_thread_id}&isEventLive=False&hideAdUnits=False"

        try:
            response = self.session.get(comments_url, timeout=30)
            response.raise_for_status()
            comments_soup = BeautifulSoup(response.text, 'lxml')
        except Exception as e:
            logger.error(f"Failed to fetch comments for {url}: {e}")
            return comments

        # Parse comments from the AJAX response
        comment_divs = comments_soup.find_all('div', class_=lambda x: x and 'comment' in str(x).lower())

        for div in comment_divs:
            # Get comment text
            text_elem = div.find(['p', 'div', 'span'], class_=lambda x: x and ('text' in str(x).lower() or 'content' in str(x).lower() or 'body' in str(x).lower()))
            if not text_elem:
                # Try getting all text from the comment div
                text = div.get_text(strip=True)
            else:
                text = text_elem.get_text(strip=True)

            # Skip empty or very short comments
            if len(text) < 10:
                continue

            # Get author
            author_elem = div.find(['span', 'a', 'strong'], class_=lambda x: x and ('author' in str(x).lower() or 'user' in str(x).lower() or 'name' in str(x).lower()))
            author = author_elem.get_text(strip=True) if author_elem else None

            comment = Comment(
                id="",
                text=text,
                source=self.source,
                article_url=url,
                article_title=article_title,
                author=author
            )
            comments.append(comment)

        logger.info(f"Scraped {len(comments)} comments from {url}")
        return comments


class Sata24Scraper(BaseScraper):
    """Scraper for 24sata.hr portal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://www.24sata.hr"
        self.source = "24sata.hr"

    def scrape_article_urls(
        self,
        section: str = "news",
        max_articles: int = 50
    ) -> List[str]:
        """Get article URLs from a section."""
        urls = []
        section_url = f"{self.base_url}/{section}"

        soup = self._get_page(section_url)
        if not soup:
            return urls

        # Find article links
        article_links = soup.find_all('a', href=True)
        for link in article_links:
            href = link['href']
            # 24sata article URLs typically contain numeric ID
            if any(x in href for x in ['/vijesti/', '/news/', '/hrvatska/', '/eu/']):
                full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                if full_url not in urls:
                    urls.append(full_url)
                    if len(urls) >= max_articles:
                        break

        logger.info(f"Found {len(urls)} article URLs from {section}")
        return urls

    def scrape_comments(self, url: str) -> List[Comment]:
        """Scrape comments from a 24sata.hr article."""
        comments = []

        soup = self._get_page(url, use_selenium=True)
        if not soup:
            return comments

        # Get article title
        title_elem = soup.find('h1')
        article_title = title_elem.get_text(strip=True) if title_elem else None

        # Try to find comment section
        comment_containers = soup.find_all(['div', 'article'], class_=lambda x: x and 'comment' in str(x).lower())

        for container in comment_containers:
            text = container.get_text(strip=True)
            if len(text) > 10:
                comment = Comment(
                    id="",
                    text=text,
                    source=self.source,
                    article_url=url,
                    article_title=article_title
                )
                comments.append(comment)

        logger.info(f"Scraped {len(comments)} comments from {url}")
        return comments


class CommentScraper:
    """Main scraper class that coordinates multiple portal scrapers."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        rate_limit: float = 3.0,
        use_selenium: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        # Initialize scrapers
        self.scrapers = {
            'index.hr': IndexHrScraper(
                output_dir=output_dir,
                rate_limit=rate_limit,
                use_selenium=use_selenium
            ),
            '24sata.hr': Sata24Scraper(
                output_dir=output_dir,
                rate_limit=rate_limit,
                use_selenium=use_selenium
            )
        }

    def scrape_portal(
        self,
        portal: str,
        sections: List[str],
        max_articles_per_section: int = 50,
        output_filename: Optional[str] = None
    ) -> int:
        """Scrape comments from a specific portal."""
        if portal not in self.scrapers:
            logger.error(f"Unknown portal: {portal}")
            return 0

        scraper = self.scrapers[portal]
        total_comments = 0
        filename = output_filename or f"{portal.replace('.', '_')}_comments.jsonl"

        for section in tqdm(sections, desc=f"Scraping {portal}"):
            logger.info(f"Scraping section: {section}")

            # Get article URLs
            article_urls = scraper.scrape_article_urls(section, max_articles_per_section)

            for url in tqdm(article_urls, desc=f"Articles in {section}", leave=False):
                comments = scraper.scrape_comments(url)
                if comments:
                    scraper.save_comments(comments, filename)
                    total_comments += len(comments)

                scraper._rate_limit_wait()

        logger.info(f"Total comments scraped from {portal}: {total_comments}")
        return total_comments

    def scrape_all(
        self,
        portals: Optional[List[str]] = None,
        sections: Optional[Dict[str, List[str]]] = None,
        max_articles_per_section: int = 50
    ):
        """Scrape comments from multiple portals."""
        if portals is None:
            portals = list(self.scrapers.keys())

        default_sections = {
            'index.hr': ['vijesti', 'hrvatska', 'eu'],
            '24sata.hr': ['news', 'hrvatska', 'eu']
        }

        if sections is None:
            sections = default_sections

        total = 0
        for portal in portals:
            portal_sections = sections.get(portal, default_sections.get(portal, ['news']))
            count = self.scrape_portal(portal, portal_sections, max_articles_per_section)
            total += count

        logger.info(f"Total comments scraped: {total}")
        return total

    def cleanup(self):
        """Clean up resources."""
        for scraper in self.scrapers.values():
            scraper._close_selenium()


def load_comments_from_jsonl(filepath: str) -> Generator[Comment, None, None]:
    """Load comments from a JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield Comment(**data)


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(description="Scrape comments from Croatian news portals")
    parser.add_argument('--portal', type=str, choices=['index.hr', '24sata.hr', 'all'], default='all')
    parser.add_argument('--sections', type=str, nargs='+', default=None)
    parser.add_argument('--max-articles', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='data/raw')
    parser.add_argument('--rate-limit', type=float, default=3.0)
    parser.add_argument('--no-selenium', action='store_true')
    parser.add_argument('--test', action='store_true', help="Run in test mode with minimal scraping")

    args = parser.parse_args()

    if args.test:
        logger.info("Running in test mode")
        print("Test mode: Scraper module loaded successfully")
        print(f"Selenium available: {SELENIUM_AVAILABLE}")
        return

    scraper = CommentScraper(
        output_dir=args.output_dir,
        rate_limit=args.rate_limit,
        use_selenium=not args.no_selenium
    )

    try:
        if args.portal == 'all':
            scraper.scrape_all(max_articles_per_section=args.max_articles)
        else:
            sections = args.sections or ['vijesti', 'hrvatska']
            scraper.scrape_portal(args.portal, sections, args.max_articles)
    finally:
        scraper.cleanup()


if __name__ == "__main__":
    main()
