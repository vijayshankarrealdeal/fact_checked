#fact_checker_agent/tool/llm_calls.py

from typing import List
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from fact_checker_agent.models.search_helper_models import Payload


PRO_MODEL_X = "gemini-2.5-pro"


def generate_bulk_ytd_summary(urls: List[Payload]) -> List[Payload]:
    """
    Generates summaries for a batch of YouTube videos in a SINGLE API call.

    Args:
        urls: A list of video Payloads to summarize.

    Returns:
        A list of Payloads, each containing a summary for one of the input videos.
    """
    if not urls:
        return []

    print(f"  -> Preparing a single API call for {len(urls)} videos.")
    responses = []
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        for url in urls:
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
                        types.Part.from_text(text="""Get text summary of this video"""),
                    ],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"""**Defining Video's Core**
        I've begun to analyze the news analysis video. 
        My focus is on identifying the main arguments and structuring them logically for a concise summary that captures its essence.
        I am paying close attention to the details.
        **Generating Summary**
        My output will be a structured summary that highlights the key points of the video, 
        {Payload.model_json_schema()}
        """
                        ),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                ),
                response_schema=Payload,
                response_mime_type="application/json",
            )

            response = client.models.generate_content(
                model=PRO_MODEL_X,
                contents=contents,
                config=generate_content_config,
            ).parsed
            responses.append(response)
        return responses
    except Exception as e:
        print(f"  [!] ERROR during bulk summarization API call: {e}")
        return []
