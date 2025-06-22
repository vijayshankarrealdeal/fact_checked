# fact_checker_agent/tool/search_executor.py

import time
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from fact_checker_agent.models.search_helper_models import BasePayload, PageContent
from utils import sanitize_text

# A list of potential CSS selectors for Google search result containers.
# The script will try these in order.
GOOGLE_RESULT_SELECTORS = [
    "div.g",
    "div.MjjYud",
    "div.Ww4FFb",
    "div[data-sokoban-container]"
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
            return [], PageContent(full_text="")

        urls_with_details, page_text = self.extract_from_webpage(combined_html)
        return urls_with_details, PageContent(full_text=page_text)

    @staticmethod
    def get_driver():
        """Initializes and returns a stealth-configured Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
        options.add_argument("--lang=en-US")
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        
        driver = webdriver.Chrome(options=options)
        return driver

    def run_search(self, search_term: str, scroll_count: int = 3, scroll_pause: float = 0.5) -> str:
        """
        Performs a Google search, scrolls down, and navigates to page 2 using a robust
        multi-selector strategy with explicit waits.
        """
        driver = self.get_driver()
        all_html = []
        try:
            url = f"https://www.google.com/search?q={search_term}&hl=en"
            driver.get(url)
            print(f"Navigated to Google Search for: {search_term}")
            
            # --- Page 1 ---
            # Wait for the first page of results to be present before proceeding
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            print("Scrolling down page 1 to load all results...")
            for _ in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause) # A small sleep is okay for scrolling action
            all_html.append(driver.page_source)
            print("Captured HTML from page 1.")

            # --- Page 2 Navigation ---
            print("Attempting to navigate to page 2...")
            try:
                # FIX: Use `By.LINK_TEXT` to find the "Next" button, as seen in your screenshot.
                # Also, wait for the element to be clickable to avoid race conditions.
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, "Next"))
                )
                
                # Scroll to the button to make sure it's in view
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                next_button.click()
                print("  - Success! Navigated to next page.")
                
                # Wait for the second page of results to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                )
                
                print("Scrolling down page 2 to load all results...")
                for _ in range(scroll_count):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause)
                all_html.append(driver.page_source)
                print("Captured HTML from page 2.")

            except (NoSuchElementException, TimeoutException):
                print("Could not find 'Next' button. Only one page of results may exist.")
                
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