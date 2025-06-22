# main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from contextlib import asynccontextmanager
import datetime

from google.adk.runners import Runner
from google.genai import types

# Import your business logic and data layer
from fact_checker_agent.agent import root_agent
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import FactCheckResult

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Initializing ADK Runner ---")
    runner = Runner(
        agent=root_agent,
        app_name="FactCheckerADK",
        session_service=database.session_service
    )
    app.state.runner = runner
    yield
    print("--- Application Shutting Down ---")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Fact Checked API",
    description="An API for running the Fact Checker agent pipeline.",
    version="1.5.0", # Version bumped to reflect the final fix
    lifespan=lifespan
)

# --- Pydantic Models for API (Unchanged) ---
class QueryRequest(BaseModel):
    user_id: str = Field(..., description="The user's unique identifier.", example="api_user_123")
    query: str = Field(..., description="The claim or topic to fact-check.", example="Iran military conflict")
    session_id: str = Field(..., description="The session ID to use. If it doesn't exist, it will be created.", example="sess_abc123")

class QueryResponse(BaseModel):
    session_id: str
    verdict: str
    credibility_score: Optional[int]
    short_summary: str
    full_explanation: str
    sources: List[str]

class ErrorResponse(BaseModel):
    detail: str

class SessionInfo(BaseModel):
    id: str
    create_time: datetime.datetime

class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]

class ChatMessage(BaseModel):
    role: str
    content: Any

class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]

# --- API Endpoints ---

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run Fact Checker Agent (Creates Session if Needed)",
    description="Takes a user query and a session ID. If the session ID does not exist, it is created automatically before running the query.",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def query_agent(request: Request, payload: QueryRequest):
    """
    This endpoint is the core of the application. It orchestrates the agent
    pipeline. It implements a "create on demand" session logic.
    """
    runner = request.app.state.runner

    try:
        # This call now uses the robust existence check.
        await database.ensure_session_exists_async(
            session_id=payload.session_id,
            user_id=payload.user_id
        )

        # Now, we can safely run the agent, as the session is guaranteed to exist.
        async for event in runner.run_async(
            user_id=payload.user_id,
            session_id=payload.session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=payload.query)]),
        ):
            print(f"--- Event from {event.author} (ID: {event.id}) ---")

            if event.is_final_response() and event.author == "FactRankerAgent":
                if event.content and event.content.parts:
                    final_json_string = event.content.parts[0].text
                    try:
                        result = FactCheckResult.model_validate_json(final_json_string)
                        return QueryResponse(
                            session_id=payload.session_id,
                            verdict=result.verdict,
                            credibility_score=result.credibility_score,
                            short_summary=result.short_summary,
                            full_explanation=result.full_explanation,
                            sources=result.sources
                        )
                    except Exception as e:
                        print(f"Error parsing final agent response: {e}")
                        raise HTTPException(status_code=500, detail=f"Error processing agent's final output: {e}")

        raise HTTPException(status_code=500, detail="Agent pipeline finished without a valid final response.")

    except Exception as e:
        print(f"FATAL ERROR IN AGENT EXECUTION: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions/{user_id}",
    response_model=ListSessionsResponse,
    summary="List User Sessions",
    description="Retrieves a list of all past sessions for a specific user.",
    responses={500: {"model": ErrorResponse}}
)
def list_past_sessions(user_id: str):
    """
    This is a synchronous endpoint, so it's okay for it to use the
    synchronous wrapper from the database module.
    """
    try:
        sessions_response = database.list_sessions_sync(user_id)
        valid_sessions = [s for s in sessions_response.sessions if hasattr(s, 'create_time')]
        return ListSessionsResponse(sessions=valid_sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load sessions: {e}")


@app.get(
    "/session/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get Session Chat History",
    description="Retrieves the full chat history for a specific session.",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
def load_past_session_chats(session_id: str, user_id: str):
    """
    This is a synchronous endpoint, so it's okay for it to use the
    synchronous wrapper from the database module.
    """
    try:
        history = database.get_session_history_sync(session_id, user_id)
        if not history:
             raise HTTPException(status_code=404, detail="Session not found or history is empty.")
        return SessionHistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load session history: {e}")