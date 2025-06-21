from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from fact_checker_agent.models.search_helper_models import YoutubePayload

# Import the PRO_MODEL from the agent definition to maintain consistency
# Assuming agent.py is in the parent directory, or adjust import path as needed
# For simplicity, let's redefine it here or ensure it's imported globally if possible.
# If you prefer to strictly import, make sure the PRO_MODEL is defined in a shared constants file
# For now, I'll hardcode it here for demonstration, but a shared constant is better practice.
PRO_MODEL_FOR_TOOL = "gemini-2.5-pro"


def generate_ytd_summary(query, url: str) -> YoutubePayload:
    """
    Generates a summary of a YouTube video using the Gemini API.
    """

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Use the PRO_MODEL for video summarization, as it's a complex task
    model_to_use = PRO_MODEL_FOR_TOOL
    
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
My output will be a structured summary that highlights the key points of the video, including:
1. **Video Title**: The title of the video.
2. **Summary of Video**: A brief summary that captures the main arguments and points discussed in the video.
3. **Query**: {query} The original query used to related the video.
4. **URL**: The URL of the video.
5. **Relative Percent**: A score indicating how relevant the video is to the query, on a scale from 0 to 100.

"""
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_schema=YoutubePayload,
        response_mime_type="application/json",
    )

    return client.models.generate_content(
        model=model_to_use, # Use the selected model
        contents=contents,
        config=generate_content_config,
    ).parsed