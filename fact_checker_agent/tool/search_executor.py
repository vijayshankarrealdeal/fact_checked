from fact_checker_agent.models.search_helper_models import Payload

# from serpapi import GoogleSearch
import json


class SearchExecutor:

    def __init__(self):
        pass

    def extract_search_information(self, query: str) -> list[Payload]:

        simple_params = {
            "api_key": "",
            "engine": "google",
            "q": "Coffee",
            "location": "Austin, Texas, United States",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
        news_params = {
            "api_key": "7b1cd00ef895ee1cf56bdf235b7237431ebb91906aa73ea4a5d2ed178cbdfe94",
            "engine": "google",
            "filter": "0",
            "num": "25",
            "q": "Iran War",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "location": "United States",
            "tbm": "nws",
        }
        video_params = {
            "api_key": "7b1cd00ef895ee1cf56bdf235b7237431ebb91906aa73ea4a5d2ed178cbdfe94",
            "engine": "google",
            "q": "Iran War",
            "location": "Austin, Texas, United States",
            "google_domain": "google.com",
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
        news_results = [Payload(**item) for item in results_news.get("news_results", [])]

        with open("fact_checker_agent/tool/test_payload_video.json", "r") as file:
            results_video = json.load(file)
        video_results = [Payload(**item) for item in results_video.get("video_results", [])]
        
        web_pages = set()

        for page, npage in zip(organic_results, news_results):
            if "youtube" in page.link or "youtube" in npage.link:
                video_results.append(page)
            else:
                web_pages.add(page)
                web_pages.add(npage)
        return list(web_pages) + video_results