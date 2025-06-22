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
            return [], PageContent(full_text="")

        urls_with_details, page_text = self.extract_from_webpage(combined_html)
        return urls_with_details, PageContent(full_text=page_text)

    @staticmethod
    def get_driver():
        """Initializes and returns a stealth-configured Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        # FIX 1: Use the modern headless mode which is less detectable
        options.add_argument("--headless=new")
        # Standard arguments to make it look more like a real browser
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        # FIX 2: Set a realistic User-Agent string
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
        # FIX 3: Set language to avoid being flagged as a bot
        options.add_argument("--lang=en-US")
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        
        # This specifies the path inside the Docker container
        service = webdriver.ChromeService()
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def run_search(self, search_term: str, scroll_count: int = 3, scroll_pause: float = 0.5) -> str:
        """
        Performs a Google search, scrolls down, and navigates to page 2 using a robust
        multi-selector strategy with explicit waits.
        """
        driver = self.get_driver()
        all_html = []
        try:
            # --- Page 1 ---
            url = f"https://www.google.com/search?q={search_term}&hl=en"
            driver.get(url)
            print(f"Navigated to Google Search for: {search_term}")

            # FIX 4: Use WebDriverWait for more reliable loading
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, GOOGLE_RESULT_SELECTORS[1]))
            )
            
            print("Scrolling down page 1 to load all results...")
            for _ in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
            all_html.append(driver.page_source)
            print("Captured HTML from page 1.")

            # --- Page 2 Navigation (with fallback selectors and explicit wait) ---
            print("Attempting to navigate to page 2...")
            try:
                # Use a more generic selector for the pagination area
                pagination_area = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "pnnext"))
                )
                pagination_area.click()
                print("  - Success! Navigated to next page.")
                
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, GOOGLE_RESULT_SELECTORS[1]))
                )
                print("Scrolling down page 2 to load all results...")
                for _ in range(scroll_count):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause)
                all_html.append(driver.page_source)
                print("Captured HTML from page 2.")

            except (NoSuchElementException, TimeoutException):
                print("Could not find pagination buttons. Only one page of results may exist.")
                # For debugging headless issues, uncomment these lines:
                # driver.save_screenshot('headless_debug.png')
                # with open('headless_debug.html', 'w', encoding='utf-8') as f:
                #     f.write(driver.page_source)

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