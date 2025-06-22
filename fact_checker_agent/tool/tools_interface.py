import time  # <-- Import the time module
from typing import Any, List, Dict
from fact_checker_agent.models.search_helper_models import Payload
from fact_checker_agent.tool.llm_calls import generate_bulk_ytd_summary
from fact_checker_agent.tool.search_executor import SearchExecutor
from fact_checker_agent.tool.url_executor import extract_external_links_info
from utils import is_duration_within_limit

# --- START: THE FIX ---
# Drastically reduce the number of items to process to avoid rate limits
MAX_VIDEOS_TO_PROCESS = 1
MAX_WEB_PAGES_TO_PROCESS = 2  # Limit total pages to scrape
WEB_SUMMARY_BATCH_SIZE = 2  # Summarize the 2 pages in one go
# --- END: THE FIX ---


async def summarize_web_pages(urls: List[Payload]) -> Dict[str, Any]:
    """
    Asynchronously scrapes content from web URLs, summarizes the content in batches
    using an LLM, and compiles the results into a single analysis block.
    """
    print(f"--- TOOL: Summarizing a potential of {len(urls)} web pages ---")
    if not urls:
        return {"content": "No web pages to summarize.", "sources": []}

    # Limit the number of URLs to process to avoid long runtimes
    urls_to_process = urls[:MAX_WEB_PAGES_TO_PROCESS]
    print(f"--- Limiting processing to the first {len(urls_to_process)} URLs. ---")

    # 1. Scrape content from all pages concurrently
    scraped_data = await extract_external_links_info(urls_to_process)

    # Filter out pages where scraping failed
    valid_pages = [
        page
        for page in scraped_data
        if page.get("content_summary")
        and not page.get("content_summary").startswith("Could not extract")
    ]
    if not valid_pages:
        return {
            "content": "Could not extract content from any of the provided web pages.",
            "sources": [],
        }

    print(
        f"--- Successfully scraped content from {len(valid_pages)} pages. Now summarizing in batches. ---"
    )
    page_contents = [
        page.model_dump() for page in valid_pages if len(page["content_summary"]) > 1000
    ]
    combined_summary_text = ""
    for i in page_contents:
        i["content_summary"] = i["content_summary"][:1500]
        combined_summary_text += (
            f"--- Summary for {i['title']} ---\n{i['content_summary']}\n"
        )
    return {
        "content": (
            combined_summary_text
            if combined_summary_text
            else "No summaries could be generated."
        ),
    }


def summarize_youtube_videos_in_bulk(query: str, urls: List[Payload]) -> Dict[str, Any]:
    """
    (Orchestrator)
    Filters videos, calls the summarization tool, and formats the output.
    """
    print(f"--- TOOL: Summarizing YouTube videos. Received {len(urls)} URLs. ---")
    if not urls:
        return {"content": "No YouTube videos to summarize.", "sources": []}

    # Filter for valid youtube links with duration
    all_urls = [
        url
        for url in urls
        if isinstance(url, dict)
        and url.get("duration")
        and "youtube" in url.get("link", "")
    ]
    # Filter for videos under the duration limit
    short_urls_payloads = [
        p for p in all_urls if is_duration_within_limit(p["duration"], 6)
    ]

    if not short_urls_payloads:
        print("No videos found within the 6-minute duration limit.")
        return {
            "content": "No suitable YouTube videos found within the 6-minute duration limit.",
            "sources": [],
        }

    # Take up to the max number of videos for our API calls
    urls_to_process = short_urls_payloads[:MAX_VIDEOS_TO_PROCESS]
    links_to_process = [p["link"] for p in urls_to_process]

    # --- START: THE FIX ---
    # The llm_calls function now handles the delay, but we confirm we only send a small number
    print(f"--- Sending {len(links_to_process)} video(s) to summarizer. ---")
    # --- END: THE FIX ---

    # Call the processing function
    summaries_payloads = generate_bulk_ytd_summary(links_to_process)

    print(f"--- Finished. Generated {len(summaries_payloads)} video summaries. ---")

    # Format the output for the FactRankerAgent
    final_analysis_parts = []
    final_source_urls = []

    for summary_payload in summaries_payloads:
        if isinstance(summary_payload, Payload):
            link = summary_payload.link
            title = summary_payload.title
            summary = summary_payload.content_summary
            final_analysis_parts.append(
                f"Source: {title} ({link})\nSummary: {summary}\n"
            )
            final_source_urls.append(link)

    combined_summary_text = "\n".join(final_analysis_parts)

    return {
        "content": (
            combined_summary_text
            if combined_summary_text
            else "No video summaries could be generated."
        ),
        "sources": final_source_urls,
    }


def search_the_web_and_youtube(query: str) -> Dict[str, Any]:
    """
    Searches Google for web articles and YouTube for relevant videos based on a query.

    Args:
        query: The search term.

    Returns:
        A dictionary containing lists of web and YouTube URLs.
    """
    print(f"--- TOOL: Starting search for query: {query} ---")

    search_executor = SearchExecutor()
    web_urls, youtube_urls = search_executor.extract_search_information(query)

    print(
        f"--- Found {len(web_urls)} web pages and {len(youtube_urls)} YouTube videos. ---"
    )
    web_urls = [urls.model_dump() for urls in web_urls]
    youtube_urls = [
        urls.model_dump() for urls in youtube_urls if "youtube" in urls.link
    ]
    return {"gathered_urls": {"web_urls": web_urls, "youtube_urls": youtube_urls}}
