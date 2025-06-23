# fact_checker_agent/tool/tools_interface.py
from typing import Any, List, Dict
from google import genai
from fact_checker_agent.models.search_helper_models import Payload
from fact_checker_agent.tool.llm_calls import generate_bulk_ytd_summary
from fact_checker_agent.tool.search_executor import SearchExecutor
from fact_checker_agent.tool.url_executor import extract_external_links_info
from utils import is_duration_within_limit
from fact_checker_agent.logger import get_logger, log_tool_call, log_info, log_success, log_warning
from fact_checker_agent.db.llm_version import FLASH_MODEL

logger = get_logger(__name__)


async def summarize_web_pages(urls: List[Payload]) -> Dict[str, Any]:
    """
    Asynchronously scrapes content, then uses an LLM to create a single,
    cohesive summary of all the information found.
    """
    log_tool_call(logger, "summarize_web_pages", f"Summarizing {len(urls)} web pages")
    if not urls:
        log_warning(logger, "TOOL: No web pages to summarize.")
        return {"content": "No web pages to summarize.", "sources": []}

    scraped_data = await extract_external_links_info(urls)

    valid_pages = [
        page for page in scraped_data
        if page.get("content_summary") and not page.get("content_summary").startswith("Could not extract")
    ]
    if not valid_pages:
        log_warning(logger, "TOOL: Could not extract content from any provided web pages.")
        return {
            "content": "Could not extract content from any of the provided web pages.",
            "sources": [],
        }

    # --- START: THE MAIN FIX ---
    # Instead of just concatenating, we build a context block and ask the LLM to summarize it.
    log_info(logger, f"TOOL: Creating combined context from {len(valid_pages)} pages for LLM summarization.")
    
    context_block = ""
    source_links = []
    for page in valid_pages:
        # Truncate each individual article to a reasonable length to avoid overwhelming the context
        truncated_content = page.get("content_summary", "")[:8000]
        context_block += f"--- Source: {page.get('title')} ({page.get('link')}) ---\n"
        context_block += f"{truncated_content}\n\n"
        source_links.append(page.get('link'))

    # Now, call the LLM to create a single, cohesive summary
    try:
        log_info(logger, "TOOL: Sending combined context to LLM for final web summary.")
        model = genai.GenerativeModel(FLASH_MODEL)
        prompt = f"""
        Based on the following articles, please provide a comprehensive, neutral summary of the key points, agreements, and disagreements regarding the user's query.

        ARTICLES:
        {context_block}

        SUMMARY:
        """
        response = await model.generate_content_async(prompt)
        summary_text = response.text
        log_success(logger, "TOOL: Successfully generated cohesive summary from web content.")
    except Exception as e:
        log_warning(logger, f"TOOL: LLM summarization failed: {e}. Falling back to concatenation.")
        summary_text = "Could not generate a cohesive summary. See individual snippets." + context_block[:10000] # Fallback

    return {
        "content": summary_text,
        "sources": source_links
    }
    # --- END: THE MAIN FIX ---


def summarize_youtube_videos_in_bulk(query: str, urls: List[Payload]) -> Dict[str, Any]:
    """
    Filters videos, calls the summarization tool, and formats the output.
    """
    log_tool_call(logger, "summarize_youtube_videos_in_bulk", f"Received {len(urls)} URLs for query '{query}'")
    if not urls:
        log_warning(logger, "TOOL: No YouTube videos to summarize.")
        return {"content": "No YouTube videos to summarize.", "sources": []}

    all_urls = [
        url
        for url in urls
        if url.get("duration")
        and "youtube" in url.get("link", "")
    ]
    short_urls_payloads = [
        p for p in all_urls if is_duration_within_limit(p["duration"], 6)
    ]

    if not short_urls_payloads:
        log_warning(logger, "TOOL: No videos found within the 6-minute duration limit.")
        return {
            "content": "No suitable YouTube videos found within the 6-minute duration limit.",
            "sources": [],
        }
    links_to_process = [p["link"] for p in short_urls_payloads]

    log_info(logger, f"--- Sending {len(links_to_process)} video(s) to summarizer. ---")
    summaries_list = generate_bulk_ytd_summary(links_to_process)

    log_success(logger, f"--- Finished. Generated {len(summaries_list)} video summaries. ---")

    final_analysis_parts = []
    final_source_urls = []

    for i, summary_text in enumerate(summaries_list):
        if i < len(short_urls_payloads):
            link = short_urls_payloads[i]['link']
            final_analysis_parts.append(
                f"Source: ({link})\nSummary: {summary_text}\n"
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
    """
    log_tool_call(logger, "search_the_web_and_youtube", f"query: '{query}'")
    search_executor = SearchExecutor()
    web_urls, youtube_urls = search_executor.extract_search_information(query)

    log_success(
        logger,
        f"--- Found {len(web_urls)} web pages and {len(youtube_urls)} YouTube videos. ---"
    )
    web_urls_dict = [urls.model_dump() for urls in web_urls]
    youtube_urls_dict = [
        urls.model_dump() for urls in youtube_urls if "youtube" in urls.link
    ]
    
    return {"web_urls": web_urls_dict, "youtube_urls": youtube_urls_dict}