# database.py
import os
import asyncio
from google.adk.sessions import DatabaseSessionService
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_NAME")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = 5432


DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


session_service = DatabaseSessionService(db_url=DB_URL)


async def list_sessions(user_id: str):
    """Lists all sessions for a given user."""
    return await session_service.list_sessions(
        app_name="FactCheckerADK", user_id=user_id
    )


async def get_session(session_id: str, user_id: str):
    """Retrieves a specific session by its ID."""
    return await session_service.get_session(
        app_name="FactCheckerADK", user_id=user_id, session_id=session_id
    )


async def create_new_session(user_id: str):
    """Creates a new session with a default initial state."""
    initial_state = {
        "user_name": user_id,
        "search_query": "",
        "gathered_urls": {},
        "web_analysis": "",
        "video_analysis": "",
        "final_fact_check_result": {},
    }
    return await session_service.create_session(
        app_name="FactCheckerADK", user_id=user_id, state=initial_state
    )


def get_session_history_sync(session_id: str, user_id: str):
    """Synchronous wrapper to retrieve chat history for use in Streamlit."""
    session = asyncio.run(get_session(session_id, user_id))
    history = []
    if session and session.messages:
        for msg in session.messages:
            role = "user" if msg.role == "user" else "assistant"
            if msg.parts and hasattr(msg.parts[0], "text"):
                history.append({"role": role, "content": msg.parts[0].text})
    return history


def list_sessions_sync(user_id: str):
    """Synchronous wrapper to list sessions for use in Streamlit."""
    return asyncio.run(list_sessions(user_id))
