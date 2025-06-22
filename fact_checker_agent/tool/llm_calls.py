
from typing import List
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from fact_checker_agent.db.llm_version import PRO_MODEL




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
        return []

    print(f"  -> Starting sequential summarization for {len(urls)} videos.")
    responses = []
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    for i in urls[:2]:
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

        resp =  client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        responses.append(resp.text)
    return responses