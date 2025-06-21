# fact_checker_agent/agent.py

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search


# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"


web_summary_agent = LlmAgent(
    name="WebSummarizerAgent",
    model=GEMINI_MODEL,
    instruction="""
    You are a Web Research Specialist. Your task is to investigate a user's query
    by searching the web for credible news articles and reports.

    1. Use the `google_search` tool to find at least 3-4 different sources.
    2. Analyze the search results to form a comprehensive understanding of the topic.
    3. Produce a concise, neutral summary of the information you found.
    4. Crucially, you MUST list all the source URLs you used at the end of your summary.
    """,
    description="Searches the web for articles and summarizes the findings with sources.",
    tools=[google_search],
    output_key="web_analysis",
)


# ==============================================================================
# AGENT 2: VIDEO SUMMARY AGENT (Parallel Task 2)
# ==============================================================================
video_summary_agent = LlmAgent(
    name="VideoSummarizerAgent",
    model=GEMINI_MODEL,
    instruction="""
    You are a Video Content Analyst. Your job is to use the available tool
    to find and summarize a YouTube video relevant to the user's query.

    1. Call the `summarize_youtube_video` tool with the user's original query.
    2. Present the summary and the source URL provided by the tool in a clear format.
    """,
    description="Finds and summarizes a relevant YouTube video.",
    tools=[],
    output_key="video_analysis",
)


# ==============================================================================
# WORKFLOW AGENT 1: INFORMATION GATHERER (Parallel Agent)
# ==============================================================================
information_gatherer = ParallelAgent(
    name="InformationGatherer",
    sub_agents=[
        web_summary_agent,
        video_summary_agent,
    ],
    description="Concurrently gathers and summarizes information from web articles and videos.",
)


# ==============================================================================
# AGENT 3: FACT RANKER & SYNTHESIZER AGENT
# ==============================================================================
fact_checker_agent = LlmAgent(
    name="FactRankerAgent",
    model=GEMINI_MODEL,
    instruction="""
    You are a meticulous Fact-Checker and News Analyst. Your role is to synthesize
    information from different media types (web articles and videos) to determine the
    veracity of a claim.

    You will be given a web analysis and a video analysis. Your task is to:
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

    **Information to Analyze:**

    **Web Article Analysis:**
    {web_analysis}

    **Video Analysis:**
    {video_analysis}
    """,
    description="Analyzes gathered intelligence, ranks credibility, and provides a final verdict.",
)


# ==============================================================================
# ROOT AGENT: FACT CHECKING PIPELINE (Sequential Agent)
# ==============================================================================
root_agent = SequentialAgent(
    name="fact_checker_agent",
    sub_agents=[
        information_gatherer,
        fact_checker_agent,
    ],
    description="A multi-agent pipeline to verify news by gathering and synthesizing information from web and video sources.",
)