# main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import datetime

from google.adk.runners import Runner
from google.genai import types

# Import your business logic and data layer
from fact_checker_agent.agent import root_agent
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import FactCheckResult, SessionHistoryResponse
from fact_checker_agent.logger import get_logger, log_info, log_error, log_success, log_warning, log_api_request, log_api_response, BColors

# --- FastAPI Lifespan Management ---
logger = get_logger("FactCheckerAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info(logger, "--- Initializing ADK Runner ---")
    runner = Runner(
        agent=root_agent,
        app_name="FactCheckerADK",
        session_service=database.session_service
    )
    app.state.runner = runner
    yield
    log_info(logger, "--- Application Shutting Down ---")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Fact Checked API",
    description="An API for running the Fact Checker agent pipeline.",
    version="1.7.0", # Version bumped for summarization and stability fix
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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
    pipeline and implements "create on demand" session logic.
    """
    log_api_request(logger, f"Received query for user '{payload.user_id}' on session '{payload.session_id}'. Query: '{payload.query}'")
    runner = request.app.state.runner

    try:
        await database.ensure_session_exists_async(
            session_id=payload.session_id,
            user_id=payload.user_id
        )

        log_info(logger, f"Starting agent execution for session_id: {payload.session_id}")
        async for event in runner.run_async(
            user_id=payload.user_id,
            session_id=payload.session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=payload.query)]),
        ):
            log_info(logger, f"--- Event from {BColors.OKCYAN}{event.author}{BColors.ENDC} (ID: {event.id}) ---")

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        log_warning(logger, f"Agent '{event.author}' is calling tool: {BColors.WARNING}{part.function_call.name}{BColors.ENDC}")
                    if part.text:
                         log_info(logger, f"Agent '{event.author}' produced text part.")

            if event.is_final_response() and event.author == "FactRankerAgent":
                log_success(logger, "FactRankerAgent produced the final response.")
                
                # --- START: THE ROBUSTNESS FIX ---
                # Safely check if the response has content before accessing it.
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_json_string = event.content.parts[0].text
                    try:
                        result = FactCheckResult.model_validate_json(final_json_string)
                        response_data = QueryResponse(
                            session_id=payload.session_id,
                            verdict=result.verdict,
                            credibility_score=result.credibility_score,
                            short_summary=result.short_summary,
                            full_explanation=result.full_explanation,
                            sources=result.sources
                        )
                        log_api_response(logger, f"Sending final response for session '{payload.session_id}'. Verdict: {result.verdict}")
                        return response_data
                    except Exception as e:
                        log_error(logger, f"Error parsing final agent JSON response: {e}. Raw response: {final_json_string}")
                        raise HTTPException(status_code=500, detail=f"Error processing agent's final output: {e}")
                else:
                    # This block now handles the case that caused the crash.
                    log_error(logger, "FactRankerAgent returned a final response but it was empty or malformed.")
                    raise HTTPException(status_code=500, detail="Agent pipeline finished but the final analysis step produced an empty response.")
                # --- END: THE ROBUSTNESS FIX ---

        log_error(logger, "Agent pipeline finished without a valid final response from FactRankerAgent.")
        raise HTTPException(status_code=500, detail="Agent pipeline finished without a valid final response.")

    except Exception as e:
        log_error(logger, f"FATAL ERROR IN AGENT EXECUTION: {e}")
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
    Synchronous endpoint to list user sessions.
    """
    log_api_request(logger, f"Listing sessions for user: {user_id}")
    try:
        sessions_response = database.list_sessions_sync(user_id)
        valid_sessions = [s for s in sessions_response.sessions if hasattr(s, 'create_time')]
        log_api_response(logger, f"Found {len(valid_sessions)} sessions for user: {user_id}")
        return ListSessionsResponse(sessions=valid_sessions)
    except Exception as e:
        log_error(logger, f"Could not load sessions for user {user_id}: {e}")
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
    Synchronous endpoint to retrieve session history.
    """
    log_api_request(logger, f"Loading history for session: {session_id}, user: {user_id}")
    try:
        history = database.get_session_history_sync(session_id, user_id)
        if not history:
             log_warning(logger, f"Session not found or history is empty for session_id: {session_id}")
             raise HTTPException(status_code=404, detail="Session not found or history is empty.")
        log_api_response(logger, f"Successfully loaded history for session: {session_id}")
        return SessionHistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        log_error(logger, f"Could not load session history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load session history: {e}")