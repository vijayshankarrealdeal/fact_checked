# engine/utils.py

import re
from bs4 import BeautifulSoup
from datetime import timedelta

def sanitize_text(text: str) -> str:
    """Sanitizes text by removing excessive whitespace and non-printable characters."""
    if not text:
        return ""
    # Collapse multiple spaces, newlines, and tabs into a single space
    text = re.sub(r"\s+", " ", text)
    # Remove control characters but keep standard punctuation
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def parse_html_content(html: str) -> str:
    """
    Parses HTML to extract clean text, links, and images in a single pass.
    This is the centralized function for all HTML parsing needs.

    Args:
        html: The raw HTML content as a string.

    Returns:
        full_text: Cleaned, human-readable text from the page.
    """

    soup = BeautifulSoup(html, "lxml")  # Use lxml for performance
    for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
        tag.decompose()
    full_text = soup.get_text(separator=" ", strip=True)
    full_text = sanitize_text(full_text)
    print(f"Extracted text content (length: {len(full_text)} chars).")
    return full_text


def extract_texts(html: str) -> str:
    """Wrapper task to get text from the main parsing function."""
    return parse_html_content(html)



def parse_duration(time_str: str) -> timedelta:
    """
    (Helper function)
    Parses a time string in H:M:S, M:S, or S format into a timedelta object.
    Returns a timedelta object.
    Raises ValueError for invalid formats.
    """
    parts = list(map(int, time_str.split(':')))
    
    hours, minutes, seconds = 0, 0, 0

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        minutes, seconds = parts
    elif len(parts) == 1:
        seconds = parts[0]
    else:
        raise ValueError(f"Invalid time format in string: '{time_str}'")
        
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def is_duration_within_limit(time_string: str, limit_minutes: int) -> bool:
    """
    Checks if a time duration string is less than or equal to a given limit in minutes.

    Args:
        time_string (str): The time duration to check (e.g., "4:37:02", "8:30").
        limit_minutes (int): The maximum allowed duration in minutes.

    Returns:
        bool: True if the duration is within the limit, False otherwise.
    """
    try:
        # 1. Convert the limit in minutes into a timedelta object
        limit_duration = timedelta(minutes=limit_minutes)
        
        # 2. Convert the time string into a timedelta object using the helper
        actual_duration = parse_duration(time_string)
        
        # 3. Perform the comparison and return the boolean result
        return actual_duration <= limit_duration
        
    except (ValueError, TypeError):
        # Catches errors from parsing bad strings (e.g., "abc") or non-string inputs.
        # A duration that can't be parsed is not within the limit.
        return False