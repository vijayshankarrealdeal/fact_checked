# fact_checker_agent/db/database.py

import os
import asyncio
from typing import Dict, Any, List
from sqlalchemy import delete
from google.adk.sessions import DatabaseSessionService, Session
# --- START: THE FIX ---
# The StorageSession class is located in the database_session_service module, not a 'database' module.
from google.adk.sessions.database_session_service import StorageSession
# --- END: THE FIX ---
from dotenv import load_dotenv
from fact_checker_agent.logger import get_logger, log_info, log_success, log_warning, log_error

load_dotenv()
logger = get_logger(__name__)

DB_URL = os.getenv("DATABASE_URL")
APP_NAME = "FactCheckerADK"
session_service = DatabaseSessionService(db_url=DB_URL)

async def list_sessions(user_id: str):
    log_info(logger, f"DB: Listing sessions for user_id: {user_id}")
    return await session_service.list_sessions(app_name=APP_NAME, user_id=user_id)

async def get_session(session_id: str, user_id: str) -> Session | None:
    log_info(logger, f"DB: Attempting to get session_id: {session_id} for user_id: {user_id}")
    return await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

async def update_session_state(session_id: str, user_id: str, new_state_values: Dict[str, Any]):
    status = new_state_values.get("status", "in-progress")
    log_info(logger, f"DB: Updating session {session_id} with new state. Status: '{status}'")
    try:
        with session_service.database_session_factory() as db_session:
            storage_session = db_session.get(StorageSession, (APP_NAME, user_id, session_id))
            if storage_session:
                storage_session.state.update(new_state_values)
                db_session.commit()
                log_success(logger, f"DB: Successfully updated session {session_id}.")
            else:
                log_warning(logger, f"DB: Could not find session {session_id} to update.")
    except Exception as e:
        log_warning(logger, f"DB: Failed to update session {session_id}. Error: {e}")

async def ensure_session_exists_async(session_id: str, user_id: str, query: str):
    log_info(logger, f"DB: Ensuring session '{session_id}' exists for user '{user_id}'.")
    existing_session = await get_session(session_id, user_id)

    if existing_session is None:
        log_warning(logger, f"DB: Session '{session_id}' not found. Creating it now.")
        initial_state = {
            "user_name": user_id, "original_query": query, "status": "ACCEPTED",
            "search_query": "", "gathered_urls": {}, "web_analysis": "",
            "video_analysis": "", "final_fact_check_result": None,
        }
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, state=initial_state, session_id=session_id
        )
        log_success(logger, f"DB: Successfully created session with specific ID '{session_id}'.")
    else:
        log_info(logger, f"DB: Found session '{session_id}'. Resetting it for new query.")
        try:
            with session_service.database_session_factory() as db_session:
                storage_session = db_session.get(StorageSession, (APP_NAME, user_id, session_id))
                if storage_session:
                    storage_session.state["status"] = "ACCEPTED"
                    storage_session.state["original_query"] = query
                    storage_session.state["final_fact_check_result"] = None
                    storage_session.state["gathered_urls"] = {}
                    storage_session.state["web_analysis"] = ""
                    storage_session.state["video_analysis"] = ""
                    db_session.commit()
                    log_success(logger, f"DB: Successfully reset session '{session_id}'.")
        except Exception as e:
            log_error(logger, f"DB: Failed to reset session {session_id}. Error: {e}")


async def get_all_sessions_for_user_async(user_id: str) -> List[Session]:
    """Fetches all sessions associated with a specific user ID."""
    log_info(logger, f"DB: Fetching all sessions for user_id: {user_id}")
    response = await session_service.list_sessions(app_name=APP_NAME, user_id=user_id)
    log_success(logger, f"DB: Found {len(response.sessions)} sessions for user {user_id}.")
    return response.sessions

async def delete_all_sessions_for_user_async(user_id: str) -> int:
    """Deletes all sessions for a user and returns the count of deleted rows."""
    log_info(logger, f"DB: Deleting all sessions for user_id: {user_id}")
    try:
        with session_service.database_session_factory() as db_session:
            stmt = delete(StorageSession).where(
                StorageSession.app_name == APP_NAME,
                StorageSession.user_id == user_id
            )
            result = db_session.execute(stmt)
            db_session.commit()
            deleted_count = result.rowcount
            log_success(logger, f"DB: Successfully deleted {deleted_count} sessions for user {user_id}.")
            return deleted_count
    except Exception as e:
        log_error(logger, f"DB: Failed to delete sessions for user {user_id}. Error: {e}")
        return 0

async def get_session_history_async(session_id: str, user_id: str):
    log_info(logger, f"DB: Retrieving session history for session_id: {session_id}")
    session = await get_session(session_id, user_id)
    history = []
    if session and session.state and session.state.get("final_fact_check_result"):
         history.append({
            "user": session.state.get('original_query'),
            "ai_response":session.state.get('final_fact_check_result')
        })
    log_success(logger, f"DB: Found {len(history)} entries in history for session_id: {session_id}")
    return history

def get_session_history_sync(session_id: str, user_id: str):
    return asyncio.run(get_session_history_async(session_id, user_id))

def list_sessions_sync(user_id: str):
    return asyncio.run(list_sessions(user_id))