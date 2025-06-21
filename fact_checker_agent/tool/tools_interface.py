from typing import Any, List, Dict

from fact_checker_agent.models.search_helper_models import BasePayload
from fact_checker_agent.tool.llm_calls import generate_ytd_summary
from fact_checker_agent.tool.search_executor import SearchExecutor
from fact_checker_agent.tool.url_executor import extract_external_links_info
from fact_checker_agent.tool.youtube_metadata_executor import search_youtube_urls_by_duration


async def summarize_web_pages(urls: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Asynchronously scrapes the content from a list of web URLs and compiles them.

    Args:
        urls: A list of dictionaries, each with a 'url' key, passed from the LLM.

    Returns:
        A dictionary containing the combined content and source URLs.
    """
    print(f"--- TOOL: Summarizing {len(urls)} web pages ---")
    if not urls:
        return {"content": "No web pages to summarize.", "sources": []}
    
    # This function now receives a list of dicts and passes it along.
    # The 'extract_external_links_info' function will handle the dicts correctly.
    scraped_data = await extract_external_links_info(urls)
    
    # Process the returned list of BasePayload objects
    full_content = "\n\n".join(
        f"--- Content from {item.url} ---\n{item.summary[:4000] if item.summary else 'No summary available.'}"
        for item in scraped_data
    )
    sources = [item.url for item in scraped_data]
    
    return {"content": full_content, "sources": sources}


def summarize_youtube_videos(query: str, urls: List[str]) -> Dict[str, Any]:
    """
    Generates a summary for the first YouTube video in a list.

    Args:
        query: The original search query.
        urls: A list of YouTube video URLs.

    Returns:
        A dictionary containing the summary and source URL of the first video.
    """
    print(f"--- TOOL: Summarizing YouTube videos. Received {len(urls)} URLs. ---")
    if not urls:
        return {"content": "No YouTube videos to summarize.", "sources": []}
    
    # For this implementation, we summarize the first relevant video found.
    first_url = urls[0]
    print(f"--- TOOL: Processing first video: {first_url} ---")
    try:
        summary_payload = generate_ytd_summary(query, first_url)

        if summary_payload and summary_payload.summary_of_video:
            content = f"Title: {summary_payload.video_title}\n\nSummary:\n{summary_payload.summary_of_video}"
            sources = [summary_payload.url]
            return {"content": content, "sources": sources}
        else:
            return {"content": f"Could not generate summary for {first_url}.", "sources": [first_url]}
    except Exception as e:
        print(f"Error calling generate_ytd_summary: {e}")
        return {"content": f"An error occurred while summarizing {first_url}: {e}", "sources": [first_url]}

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
    web_urls, content  = search_executor.extract_search_information(query)
    print(f"--- TOOL: Found {len(web_urls)} web URLs ---")
    youtube_urls = search_youtube_urls_by_duration(query, 5, 120)
    print(f"--- TOOL: Found {len(youtube_urls)} YouTube URLs ---")
   
    # The tool call from the agent will serialize the BasePayload objects.
    # ADK handles this serialization/deserialization.
    return {"web_urls": web_urls, "youtube_urls": youtube_urls}