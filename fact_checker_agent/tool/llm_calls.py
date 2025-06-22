import json
from typing import List

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from fact_checker_agent.models.search_helper_models import Payload


PRO_MODEL_X = "gemini-2.5-pro"

def generate_bulk_ytd_summary(query: str, urls: List[Payload]) -> List[Payload]:
    """
    Generates summaries for a batch of YouTube videos in a SINGLE API call.

    Args:
        query: The original search query for context.
        urls: A list of video Payloads to summarize.

    Returns:
        A list of Payloads, each containing a summary for one of the input videos.
    """
    if not urls:
        return []

    print(f"  -> Preparing a single API call for {len(urls)} videos.")
    
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))

        # 1. Construct the 'parts' list with multiple video files
        # The first part will be the text prompt, followed by all video parts.
        user_prompt_parts = [
            types.Part.from_text(
                text=f"""Analyze all of the following videos provided. 
For EACH video, provide a summary based on its content.

The original search query for context was: "{query}"

Respond with a JSON array where each object in the array corresponds to one of the videos.
Each object must strictly follow the provided schema and include:
1. a summary of the video.
2. the video's title.
3. the original URL for that video.
4. a relevance score from 0 to 100 indicating how relevant the video is to the original query.
"""
            )
        ]

        # Add each video file to the parts list
        for url in urls:
            user_prompt_parts.append(
                types.Part(file_data=types.FileData(file_uri=url.get('link'), mime_type="video/*"))
            )

        # 2. Define the expected response schema as a LIST of Payloads
        # We need to inform the model that the top-level object is an array.
        generate_content_config = types.GenerateContentConfig(
            response_schema=types.Schema(
                type=types.Type.ARRAY,
                items=Payload.model_json_schema() # Use the schema from your Pydantic model
            ),
            response_mime_type="application/json",
        )

        # 3. Create the final contents object
        contents = [types.Content(role="user", parts=user_prompt_parts)]

        # 4. Make the single API call
        print(f"  -> Sending {len(urls)} videos to Gemini in one request...")
        response = client.models.generate_content(
            model=PRO_MODEL_X,
            contents=contents,
            config=generate_content_config,
        )
        print("  <- Received bulk summary response from Gemini.")

        # The response.text will be a JSON string of a list of objects.
        # We need to parse it and convert it into a list of Payload objects.
        # Note: If Gemini's native response_schema parsing works directly for lists, 
        # `response.parsed` might already be a list of dicts. If so, this is even simpler.
        # Let's write robust code that handles both cases.
        parsed_response = json.loads(response.text)
        if isinstance(parsed_response, list):
             return [Payload(**item) for item in parsed_response]
        else:
             # Handle cases where the model might have wrapped the list in a key
             print("[!] Warning: Model did not return a list directly. Searching for list in response.")
             for key, value in parsed_response.items():
                 if isinstance(value, list):
                     return [Payload(**item) for item in value]
        
        # If we reach here, the response was not in the expected format.
        print("[!] Error: Could not parse the list of summaries from the model's response.")
        return []

    except Exception as e:
        print(f"  [!] ERROR during bulk summarization API call: {e}")
        return []