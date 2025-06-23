# main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import datetime
import json  # For safely parsing JSON strings

from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import DatabaseSessionService

from fact_checker_agent.agent import (
    query_processor_agent,
    info_gatherer_agent,
    parallel_summarizer,
    fact_ranker_agent,
)
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import (
    FactCheckResult,
    QuerySubmitResponse,
    ResultResponse,
)
from fact_checker_agent.logger import get_logger, log_info, log_error, log_success, log_warning

logger = get_logger("FactCheckerAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info(logger, "--- Initializing ADK Session Service ---")
    app.state.session_service = database.session_service
    app.state.app_name = database.APP_NAME
    yield
    log_info(logger, "--- Application Shutting Down ---")


app = FastAPI(
    title="Fact Checked API",
    description="An asynchronous API for running the Fact Checker agent pipeline with status polling.",
    version="3.5.1",  # Version bumped for string handling fix
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models --- (Unchanged)
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


# --- End Models ---


async def run_agent_in_background(
    session_service: DatabaseSessionService,
    app_name: str,
    user_id: str,
    session_id: str,
    initial_query: str,
):
    log_info(
        logger, f"BACKGROUND: Starting orchestrated pipeline for session {session_id}"
    )

    try:
        # Step 1: Refine Query
        await database.update_session_state(
            session_id, user_id, {"status": "REFINING_QUERY"}
        )
        query_runner = Runner(
            agent=query_processor_agent,
            session_service=session_service,
            app_name=app_name,
        )
        refined_query = initial_query
        async for event in query_runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=initial_query)]
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                refined_query = event.content.parts[0].text
        await database.update_session_state(
            session_id, user_id, {"search_query": refined_query}
        )
        log_success(
            logger, f"BACKGROUND [{session_id}]: Query refined to '{refined_query}'"
        )

        # Step 2: Gather Sources
        await database.update_session_state(
            session_id, user_id, {"status": "GATHERING_SOURCES"}
        )
        gatherer_runner = Runner(
            agent=info_gatherer_agent,
            session_service=session_service,
            app_name=app_name,
        )
        async for _ in gatherer_runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="trigger gatherer")]
            ),
        ):
            pass

        updated_session = await database.get_session(session_id, user_id)
        gathered_urls_raw = (
            updated_session.state.get("gathered_urls") if updated_session else None
        )

        # --- START: THE FIX ---
        gathered_urls_data = {}
        if isinstance(gathered_urls_raw, dict):
            gathered_urls_data = gathered_urls_raw
        elif isinstance(gathered_urls_raw, str):
            try:
                # Attempt to parse it if it's a JSON string
                parsed_data = json.loads(gathered_urls_raw)
                if isinstance(parsed_data, dict):
                    gathered_urls_data = parsed_data
                    log_warning(
                        logger,
                        f"BACKGROUND [{session_id}]: 'gathered_urls' was a string but successfully parsed as JSON dict.",
                    )
                else:
                    log_warning(
                        logger,
                        f"BACKGROUND [{session_id}]: 'gathered_urls' was a string and parsed, but not into a dict. Value: {gathered_urls_raw}",
                    )
            except json.JSONDecodeError:
                log_warning(
                    logger,
                    f"BACKGROUND [{session_id}]: 'gathered_urls' was a string but could not be parsed as JSON. Value: {gathered_urls_raw}",
                )
        elif gathered_urls_raw is not None:
            log_warning(
                logger,
                f"BACKGROUND [{session_id}]: 'gathered_urls' was an unexpected type: {type(gathered_urls_raw)}. Value: {gathered_urls_raw}",
            )

        await database.update_session_state(
            session_id, user_id, {"gathered_urls": gathered_urls_data}
        )  # Save the potentially corrected dict

        web_urls_count = (
            len(gathered_urls_data.get("web_urls", []))
            if isinstance(gathered_urls_data, dict)
            else 0
        )
        youtube_urls_count = (
            len(gathered_urls_data.get("youtube_urls", []))
            if isinstance(gathered_urls_data, dict)
            else 0
        )
        log_success(
            logger,
            f"BACKGROUND [{session_id}]: Sources gathered: {web_urls_count} web, {youtube_urls_count} video.",
        )
        # --- END: THE FIX ---

        # Step 3: Analyze Content (Web & Video)
        await database.update_session_state(
            session_id, user_id, {"status": "ANALYZING_CONTENT"}
        )
        summarizer_runner = Runner(
            agent=parallel_summarizer,
            session_service=session_service,
            app_name=app_name,
        )
        async for _ in summarizer_runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="trigger summarizer")]
            ),
        ):
            pass

        updated_session_after_summarize = await database.get_session(
            session_id, user_id
        )
        web_analysis_data = (
            updated_session_after_summarize.state.get("web_analysis", "")
            if updated_session_after_summarize
            else ""
        )
        video_analysis_data = (
            updated_session_after_summarize.state.get("video_analysis", "")
            if updated_session_after_summarize
            else ""
        )

        # Ensure web_analysis and video_analysis are strings, as expected by FactRankerAgent
        if not isinstance(web_analysis_data, str):
            log_warning(
                logger,
                f"BACKGROUND [{session_id}]: web_analysis was not a string, converting. Type: {type(web_analysis_data)}",
            )
            web_analysis_data = str(web_analysis_data)
        if not isinstance(video_analysis_data, str):
            log_warning(
                logger,
                f"BACKGROUND [{session_id}]: video_analysis was not a string, converting. Type: {type(video_analysis_data)}",
            )
            video_analysis_data = str(video_analysis_data)

        await database.update_session_state(
            session_id,
            user_id,
            {"web_analysis": web_analysis_data, "video_analysis": video_analysis_data},
        )
        log_success(logger, f"BACKGROUND [{session_id}]: Content analysis complete.")

        # Step 4: Generate Final Verdict
        await database.update_session_state(
            session_id, user_id, {"status": "GENERATING_VERDICT"}
        )
        ranker_runner = Runner(
            agent=fact_ranker_agent, session_service=session_service, app_name=app_name
        )
        final_fact_check_result = None
        async for event in ranker_runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="trigger ranker")]
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                result = FactCheckResult.model_validate_json(
                    event.content.parts[0].text
                )
                final_fact_check_result = result.model_dump()
        log_success(logger, f"BACKGROUND [{session_id}]: Final verdict generated.")

        # Step 5: Complete
        await database.update_session_state(
            session_id,
            user_id,
            {"status": "COMPLETED", "final_fact_check_result": final_fact_check_result},
        )
        log_success(
            logger,
            f"BACKGROUND: Successfully completed and stored result for session {session_id}.",
        )

    except Exception as e:
        log_error(
            logger,
            f"BACKGROUND [{session_id}]: Pipeline failed. Error: {e}",
            exc_info=True,
        )
        await database.update_session_state(
            session_id, user_id, {"status": "FAILED", "error_message": str(e)}
        )


# --- API Endpoints --- (Unchanged)
@app.post(
    "/query", response_model=QuerySubmitResponse, status_code=status.HTTP_202_ACCEPTED
)
async def submit_query(
    request: Request, payload: QueryRequest, background_tasks: BackgroundTasks
):
    log_info(logger, f"API: Job accepted for session '{payload.session_id}'")
    session_service = request.app.state.session_service
    app_name = request.app.state.app_name
    await database.ensure_session_exists_async(
        session_id=payload.session_id, user_id=payload.user_id, query=payload.query
    )
    background_tasks.add_task(
        run_agent_in_background,
        session_service,
        app_name,
        payload.user_id,
        payload.session_id,
        payload.query,
    )
    return QuerySubmitResponse(
        message="Fact-check job accepted. Poll the status endpoint for results.",
        session_id=payload.session_id,
        status_endpoint=f"/query/result/{payload.session_id}?user_id={payload.user_id}",
    )


@app.get("/query/result/{session_id}", response_model=ResultResponse)
async def get_query_result(session_id: str, user_id: str):
    session = await database.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_status = session.state.get("status", "UNKNOWN")
    final_result = session.state.get("final_fact_check_result")
    log_info(logger, f"API: Status check for session '{session_id}': {session_status}")
    return ResultResponse(
        session_id=session_id,
        status=session_status,
        result=final_result if session_status == "COMPLETED" and final_result else None,
    )


@app.get(
    "/sessions/user/{user_id}",
    response_model=ListSessionsResponse,
    summary="List All Sessions for a User",
)
async def get_all_user_sessions(user_id: str):
    log_info(logger, f"API: Fetching all sessions for user '{user_id}'")
    try:
        sessions = await database.get_all_sessions_for_user_async(user_id)
        session_infos = [
            SessionInfo(
                id=s.id,
                create_time=(
                    datetime.datetime.fromtimestamp(s.create_time)
                    if hasattr(s, "create_time") and s.create_time is not None
                    else None
                ),
                last_update_time=(
                    datetime.datetime.fromtimestamp(s.last_update_time)
                    if hasattr(s, "last_update_time") and s.last_update_time is not None
                    else None
                ),
            )
            for s in sessions
        ]
        return ListSessionsResponse(sessions=session_infos)
    except Exception as e:
        log_error(logger, f"API: Failed to get sessions for user {user_id}. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions.")


@app.delete(
    "/sessions/user/{user_id}",
    response_model=DeleteResponse,
    summary="Delete All Sessions for a User",
)
async def delete_all_user_sessions(user_id: str):
    log_info(logger, f"API: Deleting all sessions for user '{user_id}'")
    try:
        deleted_count = await database.delete_all_sessions_for_user_async(user_id)
        return DeleteResponse(
            message=f"Successfully deleted all sessions for user '{user_id}'.",
            deleted_count=deleted_count,
        )
    except Exception as e:
        log_error(
            logger, f"API: Failed to delete sessions for user {user_id}. Error: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to delete sessions.")
