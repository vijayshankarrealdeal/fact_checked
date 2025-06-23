# fact_checker_agent/models/agent_output_models.py

from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class FactCheckResult(BaseModel):
    """Schema for the final fact-checking verdict."""
    query: str = Field(
        description="The original query or claim that was fact-checked."
    )
    verdict: Literal["Likely True", "Likely False", "Mixed / Misleading", "Unverified"] = Field(
        description="The final verdict on the claim's veracity."
    )
    short_summary: str = Field(
        description="A concise 2-3 sentence summary of the fact-check result, suitable for quick understanding."
    )
    full_explanation: str = Field(
        description="A detailed explanation for the verdict, citing evidence and comparing information."
    )
    sources: List[str] = Field(
        description="A list of all source URLs (web articles and YouTube videos) used for the fact-check."
    )
    credibility_score: int = Field(
        ...,
        description="An integer score from 0 to 100 representing the confidence in the verdict, based on source quality, consistency, and potential bias.",
        ge=0,
        le=100
    )

# --- START: MODIFIED MODELS FOR STATUS POLLING ---
class ChatMessage(BaseModel):
    user: str
    ai_response: FactCheckResult

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]

class QuerySubmitResponse(BaseModel):
    """Response after submitting a job."""
    message: str
    session_id: str
    status_endpoint: str

class ResultResponse(BaseModel):
    """Response when polling for the job status."""
    session_id: str
    status: str = Field(description="The current status of the fact-checking job.")
    result: Optional[FactCheckResult] = Field(default=None, description="The final result, available only when status is 'COMPLETED'.")
# --- END: MODIFIED MODELS ---


class QueryRequest(BaseModel):
    user_id: str = Field(..., example="flutter_user_123")
    query: str = Field(..., example="Iran and US military conflict")
    session_id: str = Field(..., example="sess_abc123")