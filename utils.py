# engine/utils.py


import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fact_checker_agent.models.search_helper_models import BasePayload, ImagePayload

def sanitize_text(text: str) -> str:
    """Sanitizes text by removing excessive whitespace and non-printable characters."""
    if not text:
        return ""
    # Collapse multiple spaces, newlines, and tabs into a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters but keep standard punctuation
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def parse_html_content(html: str) -> Dict[str, Any]:
    """
    Parses HTML to extract clean text, links, and images in a single pass.
    This is the centralized function for all HTML parsing needs.

    Args:
        html: The raw HTML content as a string.

    Returns:
        A dictionary containing:
        - 'full_text': Cleaned, human-readable text from the page.
        - 'links': A list of BasePayload objects for each discovered link.
        - 'images': A list of ImagePayload objects for each discovered image.
    """

    soup = BeautifulSoup(html, "lxml") # Use lxml for performance
    results = {
        "full_text": "",
        "links": [],
        "images": []
    }

    # --- 1. Extract Links ---
    # Create a copy of the soup to avoid modifying the original during link extraction
    link_soup = BeautifulSoup(str(soup), "lxml")
    for a in link_soup.find_all("a", href=True):
        link = a.get("href")
        if link and link.startswith("https"):
            text = sanitize_text(a.get_text())
            results["links"].append(BasePayload(url=link, title=text))
    print(f"Extracted {len(results['links'])} total links from page.")

    # --- 2. Extract Images ---
    for img in soup.find_all("img", src=True):
        src = img.get("src")
        height = img.get("height")
        try:
            # Only consider images with a reasonable height
            if height and int(height) > 100:
                 results["images"].append(ImagePayload(src=src, height=int(height)))
        except (ValueError, TypeError):
            continue # Ignore images with invalid height attribute
    print(f"Extracted {len(results['images'])} relevant images.")

    # --- 3. Extract and Clean Full Text ---
    # Remove non-content tags before extracting text for better readability
    for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
        tag.decompose()

    full_text = soup.get_text(separator=' ', strip=True)
    results["full_text"] = sanitize_text(full_text)
    print(f"Extracted text content (length: {len(results['full_text'])} chars).")

    return results

# The old functions are no longer needed, as their logic is consolidated above.
# We keep these two as thin wrappers ONLY if they are called directly by Prefect flows
# and you want to maintain the task names. Otherwise, they can be removed.


def extract_url_and_text(html: str) -> List[BasePayload]:
    """Wrapper task to get links from the main parsing function."""
    return parse_html_content(html).get("links", [])


def extract_texts(html: str) -> str:
    """Wrapper task to get text from the main parsing function."""
    return parse_html_content(html).get("full_text", "")


def extract_image_data(html: str) -> List[ImagePayload]:
    """Wrapper task to get images from the main parsing function."""
    return parse_html_content(html).get("images", [])