# fact_checker_agent/tool/tools_interface.py

from typing import Any, List, Dict
from fact_checker_agent.models.search_helper_models import Payload
from fact_checker_agent.tool.llm_calls import generate_bulk_ytd_summary
from fact_checker_agent.tool.search_executor import SearchExecutor
from fact_checker_agent.tool.url_executor import extract_external_links_info
from utils import is_duration_within_limit

MAX_VIDEOS_TO_PROCESS = 5


async def summarize_web_pages(urls: List[Payload]) -> Payload:
    """
    Asynchronously scrapes the content from a list of web URLs and compiles them.

    Args:
        urls: A list of Payload objects, each representing a web page.

    Returns:
        A dictionary containing the combined content and source URLs.
    """
    print(f"--- TOOL: Summarizing {len(urls)} web pages ---")
    if not urls:
        return {"content": "No web pages to summarize.", "sources": []}

    # This function now receives a list of Payload objects and passes it along.
    scraped_data = await extract_external_links_info(urls)

    for i in scraped_data:
        content_summary = i.get('content_summary', None)
        if content_summary:
            if len(content_summary) > 4500:
                i['content_summary'] = content_summary[:1500] + "..."

    return scraped_data[:2]


def summarize_youtube_videos_in_bulk(query: str, urls: List[Payload]) -> List[Payload]:
    """
    (Orchestrator)
    Filters videos and calls the bulk summarization tool.
    """
    print(f"--- TOOL: Summarizing YouTube videos. Received {len(urls)} URLs. ---")
    if not urls:
        return []
    all_urls = [url for url in urls if (url.get('duration', None) and 'youtube' in url.get('link'))]
    short_urls = [url for url in all_urls if is_duration_within_limit(url['duration'], 6)]
    if not short_urls:
        print("No videos found within the 6-minute duration limit.")
        return []

    # Take up to the max number of videos for our single API call
    urls_to_process = short_urls[:MAX_VIDEOS_TO_PROCESS]

    # Call the new bulk processing function
    summaries = generate_bulk_ytd_summary(urls_to_process)

    print(
        f"--- Finished. Successfully generated {len(summaries)} summaries from the bulk call. ---"
    )
    return summaries[:2]


def search_the_web_and_youtube(query: str) -> Dict[str, Any]:
    """
    Searches Google for web articles and YouTube for relevant videos based on a query.

    Args:
        query: The search term.

    Returns:
        A dictionary containing lists of web and YouTube URLs.
    """
    print(f"--- TOOL: Starting search for query: {query} ---")

    # Web Search
    search_executor = SearchExecutor()
    web_urls, youtube_urls = search_executor.extract_search_information(query)

    # The tool call from the agent will serialize the BasePayload objects.
    # ADK handles this serialization/deserialization.
    return {"web_urls": web_urls, "youtube_urls": youtube_urls}
