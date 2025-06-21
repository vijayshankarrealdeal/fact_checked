# fact_checker_agent/models/agent_output_models.py

from pydantic import BaseModel, Field
from typing import Literal, List

class FactCheckResult(BaseModel):
    """Schema for the final fact-checking verdict."""
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
        ge=0, # Must be greater than or equal to 0
        le=100 # Must be less than or equal to 100
    )