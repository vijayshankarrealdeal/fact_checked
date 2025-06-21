# fact_checker_agent/models/search_helper_models.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime


class BasePayload(BaseModel):
    url : str
    title : Optional[str] = Field(None, description="Title of the URL")
    summary : Optional[str] = Field(None, description="Description of the URL")
    short_description : Optional[str] = Field(None, description="Short description of the URL content")
    result_rank : Optional[int] = Field(None, description="Rank of the URL based on relevance")
    rank_reason: Optional[str] = Field(None, description="Reason for the rank")
    is_youtube : Optional[bool] = Field(None, description="Is the URL a YouTube link")




class ImagePayload(BaseModel):
    src: str
    height: int


class PageContent(BaseModel):
    full_text: str

class YoutubePayload(BasePayload):
    query: str
    url: str
    relative_percent: float
    video_title: str
    summary_of_video: str

class VideoDetails(BaseModel):
    channel: str
    duration_seconds: int
    published_at: datetime
    
    @field_validator("published_at", mode="before")
    def parse_published_at(cls, value):
        return datetime.strptime(value, "%Y%m%d")



class URLResponseModel(BaseModel):
    page_content : Optional[PageContent] = Field(None, description="Content of the Main URL")
    urls: Optional[List[Union[BasePayload, YoutubePayload]]] = Field(..., description="List of URLs")