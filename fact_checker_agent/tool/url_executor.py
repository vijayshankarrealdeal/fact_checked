import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fact_checker_agent.models.search_helper_models import BasePayload
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

def extract_page_info(url_data: dict) -> BasePayload:
    # THIS IS THE KEY FIX: Expect a dictionary and extract the url string from it.
    url_string = url_data.get('url') if isinstance(url_data, dict) else url_data

    # Handle cases where the URL might be invalid or missing
    if not isinstance(url_string, str) or not url_string.startswith('http'):
        error_msg = f"Invalid or missing URL provided: {url_string}"
        print(error_msg)
        return BasePayload(url=str(url_string), title="Invalid URL", summary=error_msg, is_youtube=False)

    driver = get_pooled_driver()
    summary = ""
    title = ""
    try:
        driver.get(url_string)
        title = driver.title
        parsed_data = parse_html_content(driver.page_source)
        summary = parsed_data.get("full_text", "")
        print(f"Successfully extracted content from: {url_string}")
    except Exception as e:
        title = "Error visiting URL"
        summary = f"Could not extract content from {url_string}. Reason: {str(e)}"
        print(summary)
        driver.quit()
        setattr(thread_local, 'driver', None)

    return BasePayload(url=url_string, title=title, summary=summary, is_youtube=False)


async def extract_external_links_info(urls: list) -> list[BasePayload]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        # The list comprehension now passes the entire dictionary `url` to extract_page_info
        futures = [loop.run_in_executor(executor, extract_page_info, url) for url in urls]
        results = await asyncio.gather(*futures)
    return results