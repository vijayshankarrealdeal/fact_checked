# fact_checker_agent/tool/url_executor.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fact_checker_agent.models.search_helper_models import Payload
from utils import parse_html_content

thread_local = threading.local()

def get_pooled_driver():
    driver = getattr(thread_local, 'driver', None)
    if driver is None:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false") # Disable images
        chrome_options.page_load_strategy = "eager" # Don't wait for all resources
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(20)
        setattr(thread_local, 'driver', driver)
    return driver

def close_all_drivers(pool: dict):
    """Safely quits all drivers in a given pool."""
    for thread_id, driver in pool.items():
        try:
            driver.quit()
            print(f"Closed driver for thread {thread_id}.")
        except Exception as e:
            print(f"Could not close driver for thread {thread_id}: {e}")

def extract_page_info(url_data: Payload) -> Payload:
    # Expect a dictionary (Payload) and extract the url string from it.
    url_string = url_data.get('link',None)
    if not url_string:
        print(f"Invalid URL data: {url_data}. Skipping extraction.")
        return url_data
    driver = get_pooled_driver()
    summary = ""
    try:
        print(f"Scraping: {url_string}")
        driver.get(url_string)
        summary = parse_html_content(driver.page_source)
    except Exception as e:
        summary = f"Could not extract content from {url_string}. Reason: {str(e)}"
        print(summary)
        # Reset driver on error
        driver.quit()
        setattr(thread_local, 'driver', None)
    
    url_data['content_summary'] = summary
    return url_data


async def extract_external_links_info(urls: list[Payload]) -> list[Payload]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [loop.run_in_executor(executor, extract_page_info, url) for url in urls]
        results = await asyncio.gather(*futures)
    
    # Clean up the single driver for the main thread if it was created
    driver = getattr(thread_local, 'driver', None)
    if driver:
        driver.quit()
        setattr(thread_local, 'driver', None)
        
    return results