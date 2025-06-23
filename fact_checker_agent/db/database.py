# fact_checker_agent/db/database.py

import os
import asyncio
from typing import Dict, Any
from google.adk.sessions import DatabaseSessionService, Session
from dotenv import load_dotenv
from fact_checker_agent.logger import get_logger, log_info, log_success, log_warning

load_dotenv()
logger = get_logger(__name__)

DB_URL = os.getenv("DATABASE_URL")
session_service = DatabaseSessionService(db_url=DB_URL)


async def list_sessions(user_id: str):
    log_info(logger, f"DB: Listing sessions for user_id: {user_id}")
    return await session_service.list_sessions(
        app_name="FactCheckerADK", user_id=user_id
    )


async def get_session(session_id: str, user_id: str) -> Session | None:
    log_info(
        logger, f"DB: Attempting to get session_id: {session_id} for user_id: {user_id}"
    )
    return await session_service.get_session(
        app_name="FactCheckerADK", user_id=user_id, session_id=session_id
    )


# --- START: MODIFIED/NEW FUNCTIONS ---
async def update_session_state(
    session_id: str, user_id: str, new_state_values: Dict[str, Any]
):
    """
    Fetches a session, merges new values into its state, and saves it.
    This is used for updating status and saving intermediate/final results.
    """
    status = new_state_values.get("status", "in-progress")
    log_info(
        logger, f"DB: Updating session {session_id} with new state. Status: '{status}'"
    )
    try:
        session = await get_session(session_id, user_id)
        if session:
            # Merge the new values into the existing state
            session.state.update(new_state_values)
            await session_service.update_session(session=session)
            log_success(logger, f"DB: Successfully updated session {session_id}.")
        else:
            log_warning(logger, f"DB: Could not find session {session_id} to update.")
    except Exception as e:
        log_warning(logger, f"DB: Failed to update session {session_id}. Error: {e}")


async def ensure_session_exists_async(session_id: str, user_id: str, query: str):
    """
    Asynchronously and reliably checks if a session exists. If not, it creates it
    with an initial 'ACCEPTED' status.
    """
    log_info(
        logger, f"DB: Ensuring session '{session_id}' exists for user '{user_id}'."
    )
    existing_session = await get_session(session_id, user_id)

    if existing_session is None:
        log_warning(logger, f"DB: Session '{session_id}' not found. Creating it now.")
        initial_state = {
            "user_name": user_id,
            "original_query": query,
            "status": "ACCEPTED",  # Initial status for polling
            "search_query": "",
            "gathered_urls": {},
            "web_analysis": "",
            "video_analysis": "",
            "final_fact_check_result": None,
        }
        await session_service.create_session(
            app_name="FactCheckerADK",
            user_id=user_id,
            state=initial_state,
            session_id=session_id,
        )
        log_success(
            logger, f"DB: Successfully created session with specific ID '{session_id}'."
        )
    else:
        # If session exists, reset it for a new query
        existing_session.state["status"] = "ACCEPTED"
        existing_session.state["original_query"] = query
        existing_session.state["final_fact_check_result"] = None
        await session_service.update_session(session=existing_session)
        log_success(
            logger, f"DB: Found and reset session '{session_id}' for new query."
        )


# --- END: MODIFIED/NEW FUNCTIONS ---


async def get_session_history_async(session_id: str, user_id: str):
    """Asynchronously retrieves chat history."""
    log_info(logger, f"DB: Retrieving session history for session_id: {session_id}")
    session = await get_session(session_id, user_id)
    history = []
    if session and session.state and session.state.get("final_fact_check_result"):
        history.append(
            {
                "user": session.state.get("original_query"),
                "ai_response": session.state.get("final_fact_check_result"),
            }
        )
    log_success(
        logger,
        f"DB: Found {len(history)} entries in history for session_id: {session_id}",
    )
    return history


def get_session_history_sync(session_id: str, user_id: str):
    """Synchronous wrapper to retrieve chat history."""
    return asyncio.run(get_session_history_async(session_id, user_id))


def list_sessions_sync(user_id: str):
    """Synchronous wrapper to list sessions."""
    return asyncio.run(list_sessions(user_id))
