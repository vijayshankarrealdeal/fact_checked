# fact_checker_agent/tool/url_executor.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from fact_checker_agent.models.search_helper_models import Payload
from utils import parse_html_content
from fact_checker_agent.logger import get_logger, log_info, log_error, log_success

logger = get_logger(__name__)
thread_local = threading.local()

# --- START: THE FIX ---
# Modified to accept the pre-installed driver path
def get_pooled_driver(driver_path: str):
    """
    Retrieves or creates a WebDriver instance for the current thread.
    Uses a pre-installed driver path to avoid race conditions.
    """
    driver = getattr(thread_local, 'driver', None)
    if driver is None:
        log_info(logger, f"Thread {threading.get_ident()}: Creating new Selenium driver instance.")
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.page_load_strategy = "eager"
        
        # Use the provided path instead of calling install() here
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.set_page_load_timeout(20)
        setattr(thread_local, 'driver', driver)
    return driver

# Modified to accept driver_path and pass it down
def extract_page_info(url_data: Payload, driver_path: str) -> Payload:
    """Worker function for scraping a single page."""
    url_string = url_data.get('link', None)
    if not url_string:
        log_error(logger, f"Invalid URL data: {url_data}. Skipping extraction.")
        return url_data
    
    # Pass the driver_path to the driver factory
    driver = get_pooled_driver(driver_path)
    summary = ""
    try:
        log_info(logger, f"Thread {threading.get_ident()}: Scraping {url_string}")
        driver.get(url_string)
        summary = parse_html_content(driver.page_source)
        log_success(logger, f"Successfully scraped and parsed: {url_string}")
    except Exception as e:
        summary = f"Could not extract content from {url_string}. Reason: {str(e)}"
        log_error(logger, summary)
        # It's good practice to quit and reset the driver on error
        try:
            driver.quit()
        except Exception:
            pass # Ignore errors on quit
        setattr(thread_local, 'driver', None)
    
    url_data['content_summary'] = summary[:1000]
    return url_data
# --- END: THE FIX ---

async def extract_external_links_info(urls: list[Payload]) -> list[Payload]:
    """
    Installs the driver once, then scrapes URLs in parallel.
    """
    if not urls:
        return []

    # --- START: THE FIX ---
    # 1. Install the driver ONCE before any threads are created.
    log_info(logger, "Ensuring ChromeDriver is installed...")
    try:
        driver_path = ChromeDriverManager().install()
        log_success(logger, f"ChromeDriver is ready at path: {driver_path}")
    except Exception as e:
        log_error(logger, f"Failed to download ChromeDriver: {e}. Aborting scrape.")
        # Return urls with error messages
        for url_data in urls:
            url_data['content_summary'] = f"Could not scrape, driver installation failed: {e}"
        return urls
    # --- END: THE FIX ---

    log_info(logger, f"Starting parallel scrape for {len(urls)} URLs.")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 2. Use 'partial' to create a worker function with the driver_path already filled in.
        worker_func = partial(extract_page_info, driver_path=driver_path)
        
        # 3. Map the worker function to the list of URLs.
        futures = [loop.run_in_executor(executor, worker_func, url) for url in urls]
        results = await asyncio.gather(*futures)
    
    # Clean up the single driver for the main thread if it was created (belt-and-suspenders)
    driver = getattr(thread_local, 'driver', None)
    if driver:
        try:
            driver.quit()
        except Exception:
            pass
        setattr(thread_local, 'driver', None)
        log_info(logger, "Cleaned up main thread Selenium driver.")
        
    log_success(logger, f"Finished parallel scraping. Processed {len(results)} URLs.")
    return results