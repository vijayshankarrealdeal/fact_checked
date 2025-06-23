# fact_checker_agent/db/database.py

import os
import asyncio
from typing import Dict, Any, List
from sqlalchemy import delete, asc
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.sessions.database_session_service import StorageSession, StorageEvent
from google.adk.events.event import Event
from google.adk.sessions import _session_util
from dotenv import load_dotenv
from fact_checker_agent.logger import (
    get_logger,
    log_info,
    log_success,
    log_warning,
    log_error,
)

load_dotenv()
logger = get_logger(__name__)

DB_URL = os.getenv("DATABASE_URL")
APP_NAME = "FactCheckerADK"
session_service = DatabaseSessionService(db_url=DB_URL)


# --- Async functions (unchanged from previous correct version) ---
async def list_sessions(user_id: str):
    log_info(logger, f"DB: Listing sessions for user_id: {user_id} (ADK method)")
    return await session_service.list_sessions(app_name=APP_NAME, user_id=user_id)


async def get_session(session_id: str, user_id: str) -> Session | None:
    log_info(
        logger, f"DB: Attempting to get session_id: {session_id} for user_id: {user_id}"
    )
    return await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )


async def update_session_state(
    session_id: str, user_id: str, new_state_values: Dict[str, Any]
):
    status = new_state_values.get("status", "in-progress")
    log_info(
        logger, f"DB: Updating session {session_id} with new state. Status: '{status}'"
    )
    try:
        with session_service.database_session_factory() as db_session:
            storage_session = db_session.get(
                StorageSession, (APP_NAME, user_id, session_id)
            )
            if storage_session:
                storage_session.state.update(new_state_values)
                db_session.commit()
                log_success(logger, f"DB: Successfully updated session {session_id}.")
            else:
                log_warning(
                    logger, f"DB: Could not find session {session_id} to update."
                )
    except Exception as e:
        log_warning(logger, f"DB: Failed to update session {session_id}. Error: {e}")


async def ensure_session_exists_async(session_id: str, user_id: str, query: str):
    log_info(
        logger, f"DB: Ensuring session '{session_id}' exists for user '{user_id}'."
    )
    existing_session = await get_session(session_id, user_id)
    if existing_session is None:
        log_warning(logger, f"DB: Session '{session_id}' not found. Creating it now.")
        initial_state = {
            "user_name": user_id,
            "original_query": query,
            "status": "ACCEPTED",
            "search_query": "",
            "gathered_urls": {},
            "web_analysis": "",
            "video_analysis": "",
            "final_fact_check_result": None,
        }
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            state=initial_state,
            session_id=session_id,
        )
        log_success(
            logger, f"DB: Successfully created session with specific ID '{session_id}'."
        )
    else:
        log_info(
            logger, f"DB: Found session '{session_id}'. Resetting it for new query."
        )
        try:
            with session_service.database_session_factory() as db_session:
                storage_session = db_session.get(
                    StorageSession, (APP_NAME, user_id, session_id)
                )
                if storage_session:
                    current_state = dict(storage_session.state)
                    current_state.update(
                        {
                            "status": "ACCEPTED",
                            "original_query": query,
                            "final_fact_check_result": None,
                            "search_query": "",
                            "gathered_urls": {},
                            "web_analysis": "",
                            "video_analysis": "",
                        }
                    )
                    storage_session.state = current_state
                    db_session.commit()
                    log_success(
                        logger, f"DB: Successfully reset session '{session_id}'."
                    )
        except Exception as e:
            log_error(logger, f"DB: Failed to reset session {session_id}. Error: {e}")


async def get_all_sessions_for_user_async(user_id: str) -> List[Session]:
    log_info(logger, f"DB: Fetching all sessions for user_id: {user_id}")
    response = await session_service.list_sessions(app_name=APP_NAME, user_id=user_id)
    log_success(
        logger, f"DB: Found {len(response.sessions)} sessions for user {user_id}."
    )
    return response.sessions


async def delete_all_sessions_for_user_async(user_id: str) -> int:
    log_info(logger, f"DB: Deleting all sessions for user_id: {user_id}")
    try:
        with session_service.database_session_factory() as db_session:
            stmt = delete(StorageSession).where(
                StorageSession.app_name == APP_NAME, StorageSession.user_id == user_id
            )
            result = db_session.execute(stmt)
            db_session.commit()
            deleted_count = result.rowcount
            log_success(
                logger,
                f"DB: Successfully deleted {deleted_count} sessions for user {user_id}.",
            )
            return deleted_count
    except Exception as e:
        log_error(
            logger, f"DB: Failed to delete sessions for user {user_id}. Error: {e}"
        )
        return 0


async def get_session_summary_async(
    session_id: str, user_id: str
) -> List[Dict[str, Any]]:
    log_info(logger, f"DB: Retrieving summary for session_id: {session_id}")
    session = await get_session(session_id, user_id)
    summary_pair = []
    if session and session.state:
        status = session.state.get("status")
        original_query = session.state.get("original_query")
        final_result = session.state.get("final_fact_check_result")
        if status == "COMPLETED" and original_query and final_result:
            summary_pair.append(
                {"user_query": original_query, "ai_fact_check_result": final_result}
            )
            log_success(
                logger, f"DB: Found completed summary for session_id: {session_id}"
            )
        elif status != "COMPLETED":
            log_warning(
                logger, f"DB: Session {session_id} not completed. Status: {status}"
            )
        else:
            log_warning(
                logger, f"DB: Could not retrieve full summary for {session_id}."
            )
    return summary_pair


# --- START: REMOVE/COMMENT OUT UNNECESSARY SYNC WRAPPER ---
# def get_session_summary_sync(session_id: str, user_id: str) -> List[Dict[str, Any]]:
#     """Synchronous wrapper to retrieve session summary."""
#     # This was causing the error. FastAPI endpoints should await the async version.
#     return asyncio.run(get_session_summary_async(session_id, user_id))
# --- END: REMOVE/COMMENT OUT UNNECESSARY SYNC WRAPPER ---


async def get_full_session_event_history_async(
    session_id: str, user_id: str
) -> List[Dict[str, Any]]:
    log_info(
        logger,
        f"DB: Fetching full event history for session_id: {session_id}, user_id: {user_id}",
    )
    history = []
    try:
        with session_service.database_session_factory() as db_session:
            storage_events = (
                db_session.query(StorageEvent)
                .filter(
                    StorageEvent.app_name == APP_NAME,
                    StorageEvent.user_id == user_id,
                    StorageEvent.session_id == session_id,
                )
                .order_by(asc(StorageEvent.timestamp))
                .all()
            )
            if not storage_events:
                log_warning(logger, f"DB: No events found for session {session_id}")
                return []
            for event_db in storage_events:
                content_simple = None
                if event_db.content:
                    try:
                        decoded_content_obj = _session_util.decode_content(
                            event_db.content
                        )
                        if decoded_content_obj and decoded_content_obj.parts:
                            content_simple = [
                                part.text
                                for part in decoded_content_obj.parts
                                if part.text
                            ]
                            if len(content_simple) == 1:
                                content_simple = content_simple[0]
                            elif not content_simple:
                                content_simple = "[Non-text content]"
                        else:
                            content_simple = "[Empty or non-standard content]"
                    except Exception as e:
                        log_warning(
                            logger,
                            f"DB: Could not decode content for event {event_db.id}: {e}",
                        )
                        content_simple = "[Content decoding error]"
                history_entry = {
                    "event_id": event_db.id,
                    "author": event_db.author,
                    "timestamp": (
                        event_db.timestamp.isoformat() if event_db.timestamp else None
                    ),
                    "content": content_simple,
                    "error_code": event_db.error_code,
                    "error_message": event_db.error_message,
                }
                history.append(history_entry)
            log_success(
                logger, f"DB: Retrieved {len(history)} events for session {session_id}"
            )
    except Exception as e:
        log_error(
            logger, f"DB: Error fetching event history for session {session_id}: {e}"
        )
    return history


def get_full_session_event_history_sync(
    session_id: str, user_id: str
) -> List[Dict[str, Any]]:
    return asyncio.run(get_full_session_event_history_async(session_id, user_id))
