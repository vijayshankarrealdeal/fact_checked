
# fact_checker_agent/agent.py

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from fact_checker_agent.tool.tools_interface import search_the_web_and_youtube, summarize_web_pages, summarize_youtube_videos_in_bulk
from fact_checker_agent.models.agent_output_models import FactCheckResult

# --- Constants ---
PRO_MODEL = "gemini-2.5-pro"
FLASH_MODEL = "gemini-2.0-flash"

query_processor_agent = LlmAgent(
    name="QueryProcessorAgent",
    model=FLASH_MODEL,
    instruction="""
    You are an expert query analyst. Your job is to take a user's request,
    which might be a question, a statement, or contain a URL, and distill it
    into a concise, neutral search query of 5-10 words suitable for Google and YouTube.
    Focus on the core topic. Output ONLY the refined search query.
    """,
    description="Refines user input into an optimal search query.",
    output_key="search_query",
)


# ==============================================================================
# AGENT 2: INFORMATION GATHERER
# ==============================================================================
info_gatherer_agent = LlmAgent(
    name="InfoGathererAgent",
    model=FLASH_MODEL,
    instruction="""
    You are an automated information gathering specialist. Your only task is to use the
    `search_the_web_and_youtube` tool with the provided search query.

    Search Query: {search_query}
    """,
    description="Uses tools to find web and video URLs based on a search query.",
    tools=[search_the_web_and_youtube],
    output_key="gathered_urls",
)


web_summarizer_agent = LlmAgent(
    name="WebSummarizerAgent",
    model=FLASH_MODEL,
    instruction="""
    You are a Web Research Specialist. Your task is to investigate a user's query
    by searching the web for credible news articles and reports.
    You are a web content analyst. Your first step is to get the full content of the provided web URLs.
    1. Call the `summarize_web_pages` tool with the list of web URLs from the `gathered_urls` state.
    2. After getting the content, create a comprehensive, neutral summary of the combined text.
    3. Crucially, you MUST include the list of source URLs provided by the tool at the end of your summary.

    1. Use the `google_search` tool to find at least 3-4 different sources.
    2. Analyze the search results to form a comprehensive understanding of the topic.
    3. Produce a concise, neutral summary of the information you found.
    4. Crucially, you MUST list all the source URLs you used at the end of your summary.
    Web URLs to process:
    {gathered_urls[web_urls]}
    """,
    description="Scrapes, analyzes, and summarizes content from web articles.",
    tools=[summarize_web_pages],
    output_key="web_analysis",
)


video_summarizer_agent = LlmAgent(
    name="VideoSummarizerAgent",
    model=FLASH_MODEL,
    instruction="""
    You are a Video Content Analyst. Your job is to use the available tool
    to find and summarize a YouTube video relevant to the user's query.
    You are a multimedia analyst. Your job is to analyze YouTube videos.
    1. Call the `summarize_youtube_videos` tool with the list of YouTube URLs.
    2. Present the returned summary and source URLs in a clear, readable format.
    YouTube URLs to process:
    {gathered_urls[youtube_urls]}
    """,
    description="Analyzes and summarizes content from YouTube videos.",
    tools=[summarize_youtube_videos_in_bulk],
    output_key="video_analysis",
)

parallel_summarizer = ParallelAgent(
    name="ParallelSummarizer",
    sub_agents=[
        web_summarizer_agent,
        video_summarizer_agent,
    ],
    description="Concurrently summarizes information from web articles and YouTube videos.",
)

fact_ranker_agent = LlmAgent(
    name="FactRankerAgent",
    model=PRO_MODEL,
    instruction="""
    You are a meticulous Fact-Checker and News Analyst. Your role is to synthesize
    information from different media types (web articles and videos) to determine the
    veracity of a claim.
    veracity of a claim based on the user's original query.

    **Original Query:**
    {search_query}

    You will be given a web analysis and a video analysis. Your task is to:
    You have been given a web analysis and a video analysis. Your task is to:
    1.  Review both summaries carefully.
    2.  Compare the information. Do the sources agree or conflict?
    3.  Assess the credibility of the claim based on the combined evidence.
    4.  Formulate a final verdict, choosing from:
        - **Likely True**: Multiple credible sources corroborate the main points.
        - **Likely False**: Multiple credible sources contradict the claim.
        - **Mixed / Misleading**: Some truth, but presented out of context or with false details.
        - **Unverified**: Not enough credible information to make a determination.
    5.  Write a brief, clear explanation for your verdict.
    6.  List all the sources from both the web and video analyses for user reference.
    2.  Compare the information. Identify key points of agreement and conflict.
    3.  Assess the credibility and bias of the sources provided.
    4.  Formulate a final verdict, choosing ONLY from these options:
        - "Likely True"
        - "Likely False"
        - "Mixed / Misleading"
        - "Unverified"
    5.  Write a brief, clear, and neutral **2-3 sentence short summary** of your findings.
    6.  Write a detailed **full explanation** for your verdict.
    7.  List ALL the source URLs from both the web and video analyses.
    8.  **Calculate a `credibility_score` from 0-100.** This score should reflect your confidence in the verdict based on the quality of sources. High-quality, corroborating sources lead to a higher score (e.g., 95). Conflicting or low-quality sources lead to a lower score (e.g., 40).

    Your response MUST be a valid JSON object matching the `FactCheckResult` schema.
    DO NOT include any conversational text outside the JSON.

    **Information to Analyze:**
    {video_analysis}
    """,
    description="Analyzes gathered intelligence, ranks credibility, and provides a final verdict.",
    output_schema=FactCheckResult,
    output_key="final_fact_check_result",
)


root_agent = SequentialAgent(
    name="fact_checker_agent",
    sub_agents=[
        query_processor_agent,
        info_gatherer_agent,
        parallel_summarizer,
        fact_ranker_agent,
    ],
    description="A multi-agent pipeline to verify claims by gathering and synthesizing information from web and video sources, also mentioning the sources used and their credibility in the final analysis.",
)