# fact_checker_agent/models/agent_output_models.py

from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any, Union # Added Union

class FactCheckResult(BaseModel):
    query: str
    verdict: Literal["Likely True", "Likely False", "Mixed / Misleading", "Unverified"]
    short_summary: str
    full_explanation: str
    sources: List[str]
    credibility_score: int = Field(ge=0, le=100)

class QuerySubmitResponse(BaseModel):
    message: str
    session_id: str
    status_endpoint: str

class ResultResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[FactCheckResult] = None

class SessionSummaryPair(BaseModel):
    user_query: str
    ai_fact_check_result: FactCheckResult

class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: List[SessionSummaryPair]

# --- START: NEW MODELS FOR FULL EVENT HISTORY ---
class EventHistoryEntry(BaseModel):
    event_id: str
    author: str
    timestamp: Optional[str] = None
    content: Optional[Union[str, List[str]]] = None # Content can be single string or list of strings
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class FullEventHistoryResponse(BaseModel):
    session_id: str
    user_id: str
    history: List[EventHistoryEntry]
# --- END: NEW MODELS ---