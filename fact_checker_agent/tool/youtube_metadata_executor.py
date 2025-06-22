# engine/tools/youtube_metadata_executor.py

from typing import List, Optional
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from fact_checker_agent.models.search_helper_models import YoutubePayload


def search_youtube_urls_by_duration(
    query: str, max_results: int = 4, max_duration_seconds: int = 300
) -> List[str]:
    """
    Searches YouTube for a query and returns URLs for videos matching the duration criteria.

    This is the most efficient method for the flow: query -> filtered URLs.
    It fetches a large pool of search results with metadata in one call,
    then filters them locally to find videos no longer than `max_duration_seconds`.

    Args:
        query: The search term.
        max_results: The maximum number of video URLs to return.
        max_duration_seconds: The maximum duration of a video in seconds (default is 300s = 5 mins).

    Returns:
        A list of YouTube video URLs that match the duration filter.
    """
    # To find enough matching videos, we search for a larger number initially.
    # A 5x multiplier is a good heuristic.
    search_limit = max_results * 5
    search_query = f"ytsearch{search_limit}:{query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    ydl_opts = {
        "quiet": True,
        "extract_flat": False,
        "geo_bypass": True,
        "skip_download": True,
        "noplaylist": True,
        # Add the headers to the request options
        "http_headers": headers,
        # Force the IPv4 protocol. Sometimes cloud providers have issues with IPv6.
        "source_address": "0.0.0.0",
    }

    filtered_urls = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)

        entries = result.get("entries", [])
        if not entries:
            print("⚠️ No search results found for the query.")
            return []

        for video in entries:
            # Efficiently stop once we have found enough matching videos
            if len(filtered_urls) >= max_results:
                break

            # Ensure the entry is a valid video dictionary
            if not isinstance(video, dict):
                continue

            duration = video.get("duration")
            # The core filtering logic: check if duration is valid and within the limit
            if duration and duration <= max_duration_seconds:
                url = video.get("webpage_url")
                if url:
                    filtered_urls.append(url)

    except Exception as e:
        print(f"❌ Error during filtered YouTube search: {e}")
        return []

    return filtered_urls


# --- OTHER MODULAR FUNCTIONS (Still useful for different flows) ---


def search_youtube_by_query(query: str, max_results: int = 10) -> List[str]:
    """
    Performs a fast, general-purpose search and returns URLs without filtering.
    """
    search_query = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "geo_bypass": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)
        entries = result.get("entries", [])
        if not entries:
            return []
        video_urls = list(
            dict.fromkeys(entry.get("url") for entry in entries if entry.get("url"))
        )
        return video_urls
    except Exception as e:
        print(f"❌ Error during YouTube search query: {e}")
        return []


def get_video_metadata(video_url: str) -> Optional[YoutubePayload]:
    """
    Worker function: Fetches full metadata for a single YouTube video URL.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "geo_bypass": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video = ydl.extract_info(video_url, download=False)
        if not isinstance(video, dict) or video.get("is_live") or video.get("drm"):
            return None
        upload_date = video.get("upload_date")
        formatted_date = (
            f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
            if upload_date and len(upload_date) == 8
            else None
        )
        metadata = {
            "title": video.get("title"),
            "summary": video.get("description"),
            "url": video.get("webpage_url"),
            "result_rank": 0,
            "video_details": {
                "channel": video.get("uploader"),
                "air_time": formatted_date,
                "duration_seconds": video.get("duration", 0),
                "published_at": formatted_date,
            },
            "like_count": video.get("like_count", 0) or 0,
            "view_count": video.get("view_count", 0) or 0,
            "is_youtube": True,
        }
        return YoutubePayload(**metadata)
    except Exception:
        return None


def get_youtube_multiple_metadata(urls: List[str]) -> List[YoutubePayload]:
    """
    Fetches detailed metadata for a list of YouTube video URLs in parallel.
    Useful if you already have a list of URLs from another source.
    """
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(get_video_metadata, urls)
    return [payload for payload in results if payload is not None]