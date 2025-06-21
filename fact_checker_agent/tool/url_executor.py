# engine/tools/url_executor.py

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

def extract_page_info(url: str) -> BasePayload:
    driver = get_pooled_driver()
    summary = ""
    title = ""
    try:
        driver.get(url)
        title = driver.title
        # Use our efficient, centralized parser from utils.py
        parsed_data = parse_html_content(driver.page_source)
        summary = parsed_data.get("full_text", "")
        print(f"Successfully extracted content from: {url}")
    except Exception as e:
        title = "Error visiting URL"
        summary = f"Could not extract content from {url}. Reason: {str(e)}"
        print(summary)
        # In case of error, reset the driver for the next task
        driver.quit()
        setattr(thread_local, 'driver', None)

    return BasePayload(url=url, title=title, summary=summary, is_youtube=False)


async def extract_external_links_info(urls: list) -> list[BasePayload]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Create a future for each URL
        futures = [loop.run_in_executor(executor, extract_page_info, url.url) for url in urls]
        results = await asyncio.gather(*futures)
    return results