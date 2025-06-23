# main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import datetime

from google.adk.runners import Runner
from google.genai import types

from fact_checker_agent.agent import root_agent, query_processor_agent, info_gatherer_agent, parallel_summarizer, fact_ranker_agent
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import FactCheckResult, QuerySubmitResponse, ResultResponse
from fact_checker_agent.logger import get_logger, log_info, log_error, log_success, BColors

logger = get_logger("FactCheckerAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info(logger, "--- Initializing ADK Runner ---")
    app.state.runner = Runner(agent=root_agent, app_name="FactCheckerADK", session_service=database.session_service)
    yield
    log_info(logger, "--- Application Shutting Down ---")

app = FastAPI(
    title="Fact Checked API",
    description="An asynchronous API for running the Fact Checker agent pipeline with status polling.",
    version="3.1.0", # Version bumped for session management endpoints
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Models ---
class QueryRequest(BaseModel):
    user_id: str = Field(..., example="flutter_user_123")
    query: str = Field(..., example="Iran and US military conflict")
    session_id: str = Field(..., example="sess_abc123")

# --- START: NEW MODELS FOR NEW ENDPOINTS ---
class SessionInfo(BaseModel):
    id: str
    create_time: datetime.datetime
    last_update_time: datetime.datetime

class ListSessionsResponse(BaseModel):
    sessions: List[SessionInfo]

class DeleteResponse(BaseModel):
    message: str
    deleted_count: int
# --- END: NEW MODELS ---

# --- Background Task ---
async def run_agent_in_background(runner: Runner, user_id: str, session_id: str, initial_query: str):
    """This function runs the heavy agent logic and updates the DB."""
    # This function is unchanged
    log_info(logger, f"BACKGROUND: Starting pipeline for session {session_id}")
    state: Dict[str, Any] = {"search_query": initial_query}
    try:
        await database.update_session_state(session_id, user_id, {"status": "REFINING_QUERY"})
        async for event in runner.run_async(agent=query_processor_agent, session_id=session_id, user_id=user_id, new_message=types.Content(role="user", parts=[types.Part(text=initial_query)])):
            if event.is_final_response(): state["search_query"] = event.content.parts[0].text
        await database.update_session_state(session_id, user_id, {"search_query": state["search_query"]})
        log_success(logger, f"BACKGROUND [{session_id}]: Query refined to '{state['search_query']}'")
        await database.update_session_state(session_id, user_id, {"status": "GATHERING_SOURCES"})
        async for event in runner.run_async(agent=info_gatherer_agent, session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response(): state.update(event.output)
        await database.update_session_state(session_id, user_id, {"gathered_urls": state["gathered_urls"]})
        log_success(logger, f"BACKGROUND [{session_id}]: Sources gathered.")
        await database.update_session_state(session_id, user_id, {"status": "ANALYZING_CONTENT"})
        async for event in runner.run_async(agent=parallel_summarizer, session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response(): state.update(event.output)
        await database.update_session_state(session_id, user_id, {"web_analysis": state.get("web_analysis"), "video_analysis": state.get("video_analysis")})
        log_success(logger, f"BACKGROUND [{session_id}]: Content analysis complete.")
        await database.update_session_state(session_id, user_id, {"status": "GENERATING_VERDICT"})
        async for event in runner.run_async(agent=fact_ranker_agent, session_id=session_id, user_id=user_id, state=state):
            if event.is_final_response():
                result = FactCheckResult.model_validate_json(event.content.parts[0].text)
                state["final_fact_check_result"] = result.model_dump()
        log_success(logger, f"BACKGROUND [{session_id}]: Final verdict generated.")
        await database.update_session_state(session_id, user_id, {"status": "COMPLETED", "final_fact_check_result": state["final_fact_check_result"]})
        log_success(logger, f"BACKGROUND: Successfully completed and stored result for session {session_id}.")
    except Exception as e:
        log_error(logger, f"BACKGROUND [{session_id}]: Pipeline failed. Error: {e}", exc_info=True)
        await database.update_session_state(session_id, user_id, {"status": "FAILED", "error_message": str(e)})

# --- API Endpoints ---
@app.post("/query", response_model=QuerySubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(request: Request, payload: QueryRequest, background_tasks: BackgroundTasks):
    """Accepts a query and starts the fact-checking process in the background."""
    # This function is unchanged
    log_info(logger, f"API: Job accepted for session '{payload.session_id}'")
    runner = request.app.state.runner
    await database.ensure_session_exists_async(session_id=payload.session_id, user_id=payload.user_id, query=payload.query)
    background_tasks.add_task(run_agent_in_background, runner, payload.user_id, payload.session_id, payload.query)
    return QuerySubmitResponse(message="Fact-check job accepted. Poll the status endpoint for results.", session_id=payload.session_id, status_endpoint=f"/query/result/{payload.session_id}?user_id={payload.user_id}")

@app.get("/query/result/{session_id}", response_model=ResultResponse)
async def get_query_result(session_id: str, user_id: str):
    """Poll this endpoint to get the status and result of a fact-check job."""
    # This function is unchanged
    session = await database.get_session(session_id, user_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found.")
    session_status = session.state.get("status", "UNKNOWN")
    final_result = session.state.get("final_fact_check_result")
    log_info(logger, f"API: Status check for session '{session_id}': {session_status}")
    return ResultResponse(session_id=session_id, status=session_status, result=final_result if session_status == "COMPLETED" and final_result else None)

# --- START: NEW ENDPOINTS ---
@app.get("/sessions/user/{user_id}", response_model=ListSessionsResponse, summary="List All Sessions for a User")
async def get_all_user_sessions(user_id: str):
    """Retrieves a list of all session metadata for a specific user."""
    log_info(logger, f"API: Fetching all sessions for user '{user_id}'")
    try:
        sessions = await database.get_all_sessions_for_user_async(user_id)
        session_infos = [
            SessionInfo(
                id=s.id,
                # Convert Unix timestamp from ADK session to Python datetime
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
    """Deletes all sessions associated with a specific user."""
    log_info(logger, f"API: Deleting all sessions for user '{user_id}'")
    try:
        deleted_count = await database.delete_all_sessions_for_user_async(user_id)
        return DeleteResponse(
            message=f"Successfully deleted all sessions for user '{user_id}'.",
            deleted_count=deleted_count
        )
    except Exception as e:
        log_error(logger, f"API: Failed to delete sessions for user {user_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete sessions.")
# --- END: NEW ENDPOINTS ---