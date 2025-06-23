# main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from contextlib import asynccontextmanager

from google.adk.runners import Runner
from google.genai import types

from fact_checker_agent.agent import (
    root_agent,
    query_processor_agent,
    info_gatherer_agent,
    parallel_summarizer,
    fact_ranker_agent,
)
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import (
    FactCheckResult,
    QueryRequest,
    QuerySubmitResponse,
    ResultResponse,
)
from fact_checker_agent.logger import (
    get_logger,
    log_info,
    log_error,
    log_success,
)

logger = get_logger("FactCheckerAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info(logger, "--- Initializing ADK Runner ---")
    app.state.runner = Runner(
        agent=root_agent,
        app_name="FactCheckerADK",
        session_service=database.session_service,
    )
    yield
    log_info(logger, "--- Application Shutting Down ---")


app = FastAPI(
    title="Fact Checked API",
    description="An asynchronous API for running the Fact Checker agent pipeline with status polling.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# --- Background Task Orchestrator ---
async def run_agent_in_background(
    runner: Runner, user_id: str, session_id: str, initial_query: str
):
    """
    Runs the agent pipeline step-by-step, updating the session status in the DB.
    """
    log_info(logger, f"BACKGROUND: Starting pipeline for session {session_id}")
    state: Dict[str, Any] = {"search_query": initial_query}

    try:
        # Step 1: Refine Query
        await database.update_session_state(
            session_id, user_id, {"status": "REFINING_QUERY"}
        )
        async for event in runner.run_async(
            agent=query_processor_agent,
            session_id=session_id,
            user_id=user_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=initial_query)]
            ),
        ):
            if event.is_final_response():
                state["search_query"] = event.content.parts[0].text
        await database.update_session_state(
            session_id, user_id, {"search_query": state["search_query"]}
        )
        log_success(
            logger,
            f"BACKGROUND [{session_id}]: Query refined to '{state['search_query']}'",
        )

        # Step 2: Gather Sources
        await database.update_session_state(
            session_id, user_id, {"status": "GATHERING_SOURCES"}
        )
        async for event in runner.run_async(
            agent=info_gatherer_agent,
            session_id=session_id,
            user_id=user_id,
            state=state,
        ):
            if event.is_final_response():
                state.update(event.output)  # gets 'gathered_urls'
        await database.update_session_state(
            session_id, user_id, {"gathered_urls": state["gathered_urls"]}
        )
        log_success(logger, f"BACKGROUND [{session_id}]: Sources gathered.")

        # Step 3: Analyze Content (Web & Video)
        await database.update_session_state(
            session_id, user_id, {"status": "ANALYZING_CONTENT"}
        )
        async for event in runner.run_async(
            agent=parallel_summarizer,
            session_id=session_id,
            user_id=user_id,
            state=state,
        ):
            if event.is_final_response():
                state.update(event.output)  # gets 'web_analysis' and 'video_analysis'
        await database.update_session_state(
            session_id,
            user_id,
            {
                "web_analysis": state.get("web_analysis"),
                "video_analysis": state.get("video_analysis"),
            },
        )
        log_success(logger, f"BACKGROUND [{session_id}]: Content analysis complete.")

        # Step 4: Generate Final Verdict
        await database.update_session_state(
            session_id, user_id, {"status": "GENERATING_VERDICT"}
        )
        async for event in runner.run_async(
            agent=fact_ranker_agent, session_id=session_id, user_id=user_id, state=state
        ):
            if event.is_final_response():
                result = FactCheckResult.model_validate_json(
                    event.content.parts[0].text
                )
                state["final_fact_check_result"] = result.model_dump()
        log_success(logger, f"BACKGROUND [{session_id}]: Final verdict generated.")

        # Step 5: Complete
        await database.update_session_state(
            session_id,
            user_id,
            {
                "status": "COMPLETED",
                "final_fact_check_result": state["final_fact_check_result"],
            },
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


# --- API Endpoints ---
@app.post(
    "/query", response_model=QuerySubmitResponse, status_code=status.HTTP_202_ACCEPTED
)
async def submit_query(
    request: Request, payload: QueryRequest, background_tasks: BackgroundTasks
):
    """Accepts a query and starts the fact-checking process in the background."""
    log_info(logger, f"API: Job accepted for session '{payload.session_id}'")
    runner = request.app.state.runner

    await database.ensure_session_exists_async(
        session_id=payload.session_id, user_id=payload.user_id, query=payload.query
    )

    background_tasks.add_task(
        run_agent_in_background,
        runner,
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
    """Poll this endpoint to get the status and result of a fact-check job."""
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
