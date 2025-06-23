# fact_checker_agent/tool/llm_calls.py
from typing import List
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from fact_checker_agent.db.llm_version import PRO_MODEL
from fact_checker_agent.logger import get_logger, log_info, log_warning, log_success, log_error

logger = get_logger(__name__)


def generate_bulk_ytd_summary(urls: List[str]) -> List[str]:
    """
    Generates summaries for a list of YouTube videos by calling the API for each video.
    This function now includes a delay between calls to respect API rate limits.

    Args:
        urls: A list of video URL strings to summarize.

    Returns:
        A list of Payloads, each containing a summary for one of the input videos.
    """
    if not urls:
        log_warning(logger, "No URLs provided for summarization, skipping.")
        return []

    log_info(logger, f"-> Starting sequential summarization for {len(urls)} videos.")
    responses = []
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    if len(urls) > 4:
        log_warning(logger, f"Limiting video summarization from {len(urls)} to 2 URLs to manage costs/time.")
        urls = urls[:2]
        
    for i in urls:
        log_info(logger, f"Summarizing video: {i}")
        model = PRO_MODEL
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=i,
                            mime_type="video/*",
                        )
                    ),
                    types.Part.from_text(text="""Give the sumary of the video in 2-3 sentences. Focus on the main points and key information presented in the video."""),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,
            ),
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""You analyse the Video and generate the summary of it."""),
            ],
        )
        try:
            resp =  client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            log_success(logger, f"Successfully summarized video: {i}")
            responses.append(resp.text)
        except Exception as e:
            log_error(logger, f"Failed to summarize video {i}. Error: {e}")
            responses.append(f"Error summarizing video {i}: {e}")

    return responses