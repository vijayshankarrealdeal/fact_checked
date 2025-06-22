# fact_checker_agent/tool/search_executor.py


from serpapi import GoogleSearch
import json
from fact_checker_agent.models.search_helper_models import Payload
from dotenv import load_dotenv
import os
load_dotenv()

class SearchExecutor:

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_SEARCH_APIS_KEY")

    def extract_search_information(self, query: str) -> list[Payload]:

        simple_params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "location": "Austin, Texas, United States",
            "google_domain": "google.com",
            "num": "5",
            "gl": "us",
            "hl": "en",
        }
        news_params = {
            "api_key": self.api_key,
            "engine": "google",
            "filter": "0",
            "num": "10",
            "q": query,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "location": "United States",
            "tbm": "nws",
        }
        video_params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "location": "Austin, Texas, United States",
            "google_domain": "google.com",
            "num": "10",
            "gl": "us",
            "hl": "en",
            "tbm": "vid",
        }
        # search = GoogleSearch(simple_params)
        # results_common = search.get_dict()
        with open("fact_checker_agent/tool/test_common.json", "r") as file:
            results_common = json.load(file)
        organic_results = [Payload(**item) for item in results_common.get("organic_results", [])]
        top_results = [Payload(**item) for item in results_common.get("top_stories", [])]
        organic_results.extend(top_results)

        with open("fact_checker_agent/tool/test_payload_news.json", "r") as file:
            results_news = json.load(file)
        # search = GoogleSearch(news_params)
        # results_news = search.get_dict()
        news_results = [Payload(**item) for item in results_news.get("news_results", [])]

        with open("fact_checker_agent/tool/video_payload.json", "r") as file:
            results_video = json.load(file)
        # search = GoogleSearch(video_params)
        # results_video = search.get_dict()
        video_results_from_file = [Payload(**item) for item in results_video.get("video_results", [])]
        
        web_pages = list()
        video_pages = video_results_from_file

        all_web_results = organic_results + news_results
        
        for result in all_web_results:
            # Simple heuristic to separate video links from web pages
            if "youtube.com" in result.link:
                # Add to video list if not already present
                if not any(v.link == result.link for v in video_pages):
                    video_pages.append(result)
            else:
                # Add to web page list if not already present
                if not any(w.link == result.link for w in web_pages):
                    web_pages.append(result)

        return web_pages, video_pages