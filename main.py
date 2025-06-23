# main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import datetime

from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import DatabaseSessionService

# Import the individual agents we will orchestrate manually
from fact_checker_agent.agent import query_processor_agent, info_gatherer_agent, parallel_summarizer, fact_ranker_agent
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import FactCheckResult, QuerySubmitResponse, ResultResponse
from fact_checker_agent.logger import get_logger, log_info, log_error, log_success

logger = get_logger("FactCheckerAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info(logger, "--- Initializing ADK Session Service ---")
    # The runner is no longer needed at the app level, but the session_service is.
    app.state.session_service = database.session_service
    yield
    log_info(logger, "--- Application Shutting Down ---")

app = FastAPI(
    title="Fact Checked API",
    description="An asynchronous API for running the Fact Checker agent pipeline with status polling.",
    version="3.3.0", # Version bumped for correct orchestration
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Models ---
class QueryRequest(BaseModel):
    user_id: str = Field(..., example="flutter_user_123")
    query: str = Field(..., example="Iran and US military conflict")
    session_id: str = Field(..., example="sess_abc123")

class SessionInfo(BaseModel):
    id: str
    create_time: Optional[datetime.datetime] = None
    last_update_time: Optional[datetime.datetime] = None

class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]

class DeleteResponse(BaseModel):
    message: str
    deleted_count: int

# --- START: THE CORRECTED BACKGROUND ORCHESTRATOR ---
async def run_agent_in_background(session_service: DatabaseSessionService, user_id: str, session_id: str, initial_query: str):
    """
    Runs the agent pipeline step-by-step using a new Runner for each step,
    updating the session status in the DB between each step. This prevents stale session errors.
    """
    log_info(logger, f"BACKGROUND: Starting orchestrated pipeline for session {session_id}")
    state: Dict[str, Any] = {"search_query": initial_query}

    try:
        # Step 1: Refine Query
        await database.update_session_state(session_id, user_id, {"status": "REFINING_QUERY"})
        query_runner = Runner(agent=query_processor_agent, session_service=session_service)
        async for event in query_runner.run_async(session_id=session_id, user_id=user_id, new_message=types.Content(role="user", parts=[types.Part(text=initial_query)])):
            if event.is_final_response():
                state["search_query"] = event.content.parts[0].text
        await database.update_session_state(session_id, user_id, {"search_query": state["search_query"]})
        log_success(logger, f"BACKGROUND [{session_id}]: Query refined to '{state['search_query']}'")

        # Step 2: Gather Sources
        await database.update_session_state(session_id, user_id, {"status": "GATHERING_SOURCES"})
        gatherer_runner = Runner(agent=info_gatherer_agent, session_service=session_service)
        async for event in gatherer_runner.run_async(session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response():
                state.update(event.output)
        await database.update_session_state(session_id, user_id, {"gathered_urls": state["gathered_urls"]})
        log_success(logger, f"BACKGROUND [{session_id}]: Sources gathered.")

        # Step 3: Analyze Content (Web & Video)
        await database.update_session_state(session_id, user_id, {"status": "ANALYZING_CONTENT"})
        summarizer_runner = Runner(agent=parallel_summarizer, session_service=session_service)
        async for event in summarizer_runner.run_async(session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response():
                 state.update(event.output)
        await database.update_session_state(session_id, user_id, {"web_analysis": state.get("web_analysis"), "video_analysis": state.get("video_analysis")})
        log_success(logger, f"BACKGROUND [{session_id}]: Content analysis complete.")

        # Step 4: Generate Final Verdict
        await database.update_session_state(session_id, user_id, {"status": "GENERATING_VERDICT"})
        ranker_runner = Runner(agent=fact_ranker_agent, session_service=session_service)
        async for event in ranker_runner.run_async(session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response():
                result = FactCheckResult.model_validate_json(event.content.parts[0].text)
                state["final_fact_check_result"] = result.model_dump()
        log_success(logger, f"BACKGROUND [{session_id}]: Final verdict generated.")

        # Step 5: Complete
        await database.update_session_state(session_id, user_id, {"status": "COMPLETED", "final_fact_check_result": state["final_fact_check_result"]})
        log_success(logger, f"BACKGROUND: Successfully completed and stored result for session {session_id}.")

    except Exception as e:
        log_error(logger, f"BACKGROUND [{session_id}]: Pipeline failed. Error: {e}", exc_info=True)
        await database.update_session_state(session_id, user_id, {"status": "FAILED", "error_message": str(e)})
# --- END: THE CORRECTED BACKGROUND ORCHESTRATOR ---

# --- API Endpoints ---
@app.post("/query", response_model=QuerySubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(request: Request, payload: QueryRequest, background_tasks: BackgroundTasks):
    log_info(logger, f"API: Job accepted for session '{payload.session_id}'")
    # We now pass the session_service to the background task, not a pre-configured runner.
    session_service = request.app.state.session_service
    await database.ensure_session_exists_async(session_id=payload.session_id, user_id=payload.user_id, query=payload.query)
    background_tasks.add_task(run_agent_in_background, session_service, payload.user_id, payload.session_id, payload.query)
    return QuerySubmitResponse(message="Fact-check job accepted. Poll the status endpoint for results.", session_id=payload.session_id, status_endpoint=f"/query/result/{payload.session_id}?user_id={payload.user_id}")

@app.get("/query/result/{session_id}", response_model=ResultResponse)
async def get_query_result(session_id: str, user_id: str):
    session = await database.get_session(session_id, user_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found.")
    session_status = session.state.get("status", "UNKNOWN")
    final_result = session.state.get("final_fact_check_result")
    log_info(logger, f"API: Status check for session '{session_id}': {session_status}")
    return ResultResponse(session_id=session_id, status=session_status, result=final_result if session_status == "COMPLETED" and final_result else None)

@app.get("/sessions/user/{user_id}", response_model=ListSessionsResponse, summary="List All Sessions for a User")
async def get_all_user_sessions(user_id: str):
    log_info(logger, f"API: Fetching all sessions for user '{user_id}'")
    try:
        sessions = await database.get_all_sessions_for_user_async(user_id)
        session_infos = [
            SessionInfo(
                id=s.id,
                create_time=datetime.datetime.fromtimestamp(s.create_time) if hasattr(s, 'create_time') and s.create_time is not None else None,
                last_update_time=datetime.datetime.fromtimestamp(s.last_update_time) if hasattr(s, 'last_update_time') and s.last_update_time is not None else None
            ) for s in sessions
        ]
        return ListSessionsResponse(sessions=session_infos)
    except Exception as e:
        log_error(logger, f"API: Failed to get sessions for user {user_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions.")

@app.delete("/sessions/user/{user_id}", response_model=DeleteResponse, summary="Delete All Sessions for a User")
async def delete_all_user_sessions(user_id: str):
    log_info(logger, f"API: Deleting all sessions for user '{user_id}'")
    try:
        deleted_count = await database.delete_all_sessions_for_user_async(user_id)
        return DeleteResponse(message=f"Successfully deleted all sessions for user '{user_id}'.", deleted_count=deleted_count)
    except Exception as e:
        log_error(logger, f"API: Failed to delete sessions for user {user_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete sessions.")