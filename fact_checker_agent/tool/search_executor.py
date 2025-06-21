# engine/tools/search_executor.py

import time
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from fact_checker_agent.models.search_helper_models import BasePayload, PageContent
from utils import sanitize_text

GOOGLE_RESULT_SELECTORS = [
    "div.g",           # Generic result block
    "div.MjjYud",      # Used in recent versions
    "div.Ww4FFb",      # Another common container
    "div[data-sokoban-container]" # The one from the previous version
]

class SearchExecutor:

    def __init__(self):
        pass

    def extract_search_information(
        self, query: str
    ) -> tuple[list[BasePayload], PageContent]:
        """Orchestrates the search and extraction process."""
        combined_html = self.run_search(query)

        if not combined_html:
            return [], PageContent(full_text="", images=[])

        urls_with_details, page_text = self.extract_from_webpage(combined_html)
        return urls_with_details, PageContent(full_text=page_text, images=[])

    @staticmethod
    def get_driver():
        """Initializes and returns a stealth-configured Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        # For CI/CD environments, uncomment the following lines:
        # options.add_argument("--headless=new")
        # options.add_argument("--no-sandbox")
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver

    def run_search(self, search_term: str, scroll_count: int = 4, scroll_pause: float = 1.0) -> str:
        """
        Performs a Google search, scrolls down, navigates to page 2 using a robust
        multi-selector strategy, and returns the combined HTML source.
        """
        driver = self.get_driver()
        all_html = []
        try:
            # --- Page 1 ---
            url = f"https://www.google.com/search?q={search_term}&hl=en"
            driver.get(url)
            print(f"Navigated to Google Search for: {search_term}")
            time.sleep(1) # Wait for initial load
            print("Scrolling down page 1 to load all results...")
            for i in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
            all_html.append(driver.page_source)
            print("Captured HTML from page 1.")

            # --- Page 2 Navigation (with fallback selectors) ---
            navigated_to_page_2 = False
            pagination_selectors = [
                'a[aria-label="Next page"]',  # Primary method
                'a[aria-label="Page 2"]'     # Fallback method, as suggested
            ]

            print("Attempting to navigate to page 2...")
            for i, selector in enumerate(pagination_selectors):
                try:
                    print(f"  - Trying selector #{i+1}: {selector}")
                    next_button = driver.find_element(By.CSS_SELECTOR, selector)
                    # Scroll button into view to ensure it's clickable
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(0.5)
                    next_button.click()
                    print("  - Success! Navigated to next page.")
                    navigated_to_page_2 = True
                    break  # Exit the loop on the first success
                except NoSuchElementException:
                    print("  - Selector not found. Trying next one.")
                    continue  # Try the next selector

            if navigated_to_page_2:
                time.sleep(1.5) # Wait for page 2 to load
                print("Scrolling down page 2 to load all results...")
                for i in range(scroll_count):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause)
                all_html.append(driver.page_source)
                print("Captured HTML from page 2.")
            else:
                print("Could not find any pagination buttons. Only one page of results may exist.")

        except Exception as e:
            print(f"An error occurred during search execution: {e}")
        finally:
            print("Closing the WebDriver.")
            driver.quit()
        return "\n".join(all_html)

    def extract_from_webpage(self, html: str) -> tuple[list[BasePayload], str]:
        """
        Extracts structured search results and full text from Google's result HTML.
        """
        soup = BeautifulSoup(html, 'lxml')
        results_list = []
        seen_urls = set()
        
        search_results_container = None
        for selector in GOOGLE_RESULT_SELECTORS:
            search_results_container = soup.select(selector)
            if search_results_container:
                print(f"Found {len(search_results_container)} results using selector '{selector}'.")
                break
        
        if not search_results_container:
            print("Could not find any search result containers.")
            return [], ""

        for result in search_results_container:
            title_tag = result.find('h3')
            link_tag = result.find('a')
            # Broader search for the snippet
            snippet_tag = result.find('div', {'style': 'display: -webkit-box'}) or result.find(attrs={"data-sncf": "2"})

            if not (title_tag and link_tag and link_tag.get('href')):
                continue

            title = sanitize_text(title_tag.get_text())
            url = link_tag['href']

            if url.startswith('/url?q='):
                url = parse_qs(urlparse(url).query).get('q', [None])[0]

            if not url or not url.startswith('http') or url in seen_urls:
                continue

            seen_urls.add(url)
            description = sanitize_text(snippet_tag.get_text()) if snippet_tag else ""

            results_list.append(
                BasePayload(url=url, title=title, short_description=description, summary=description)
            )

        full_page_text = sanitize_text(soup.get_text())
        return results_list, full_page_text