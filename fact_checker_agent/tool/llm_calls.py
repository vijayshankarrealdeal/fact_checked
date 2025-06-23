# fact_checker_agent/tool/llm_calls.py
import time
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from google.api_core import exceptions as google_exceptions

from fact_checker_agent.db.llm_version import PRO_MODEL
from fact_checker_agent.logger import get_logger, log_info, log_warning, log_success, log_error

logger = get_logger(__name__)

# --- Configuration for Retry Logic ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2


def _summarize_single_video_with_retry(url: str, client: genai.Client) -> str:
    """
    Worker function to summarize a single video with retry logic for rate limiting.

    Args:
        url: The YouTube video URL to summarize.
        client: An initialized genai.Client instance.

    Returns:
        The summary text or an error message string.
    """
    log_info(logger, f"Preparing to summarize video: {url}")
    model = PRO_MODEL
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=url,
                        mime_type="video/*",
                    )
                ),
                types.Part.from_text(text="""Give the summary of the video in 2-3 sentences. Focus on the main points and key information presented in the video."""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text="""You analyse the Video and generate the summary of it.""")],
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            log_success(logger, f"Successfully summarized video: {url}")
            return resp.text
        except google_exceptions.ResourceExhausted as e:
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                backoff_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                log_warning(logger, f"Rate limit hit for {url}. Retrying in {backoff_time:.2f} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff_time)
            else:
                log_error(logger, f"Failed to summarize video {url} after {MAX_RETRIES} attempts due to rate limiting. Error: {e}")
                return f"Error: Rate limit exhausted for video {url} after multiple retries."
        except Exception as e:
            log_error(logger, f"An unexpected error occurred while summarizing video {url}. Error: {e}")
            return f"Error: An unexpected error occurred for video {url}: {e}"

    return f"Error: Failed to summarize video {url} after all retries."


def generate_bulk_ytd_summary(urls: List[str]) -> List[str]:
    """
    Generates summaries for a list of YouTube videos in parallel,
    with retry logic to handle rate limiting.

    Args:
        urls: A list of video URL strings to summarize.

    Returns:
        A list of summary strings, one for each input video.
    """
    if not urls:
        log_warning(logger, "No URLs provided for summarization, skipping.")
        return []

    log_info(logger, f"-> Starting PARALLEL summarization for {len(urls)} videos.")
    if len(urls) > 4:
        log_warning(logger, f"Limiting video summarization from {len(urls)} to 2 URLs to manage costs/time.")
        urls = urls[:2]

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    
    # Use a ThreadPoolExecutor to process URLs in parallel
    # max_workers can be tuned, but starting with a moderate number is safe for API limits.
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Create a lambda to pass the shared client instance to the worker function
        # executor.map will run these tasks concurrently.
        def worker_lambda(url):
            return _summarize_single_video_with_retry(url, client)
        results = list(executor.map(worker_lambda, urls))

    log_success(logger, f"Parallel summarization complete. Processed {len(urls)} URLs.")
    return results