# app.py
import asyncio
import streamlit as st
from google.adk.runners import Runner
from google.genai import types

# Import your business logic and data layer
from fact_checker_agent.agent import root_agent
from fact_checker_agent.db import database
from fact_checker_agent.models.agent_output_models import FactCheckResult
# Import the newly created database module


# --- Configuration ---
APP_NAME = "Fact Checked"
DEFAULT_USER_ID = "streamlit_user"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Fact Checked",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Streamlit session state variables ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_status_markdown" not in st.session_state:
    st.session_state.final_status_markdown = None
if "credibility_score" not in st.session_state:
    st.session_state.credibility_score = None

# --- Helper Functions for Terminal Logging ---
def colored_terminal_log(text, color_code="0"):
    print(f"\033[{color_code}m{text}\033[0m")

def log_event_to_terminal(event):
    # This function remains the same as it's for logging/debugging
    colored_terminal_log(f"\n--- Event from {event.author} (ID: {event.id}) ---", "1;30")
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                colored_terminal_log(f"  Text: '{part.text.strip()}'", "34")
            if hasattr(part, "function_call") and part.function_call:
                colored_terminal_log(f"  Tool Call: {part.function_call.name} with args: {part.function_call.args}", "1;35")
            if hasattr(part, "function_response") and part.function_response:
                colored_terminal_log(f"  Tool Response for {part.function_response.name}: {part.function_response.response}", "1;33")
    if event.actions and event.actions.state_delta:
        colored_terminal_log(f"  State Change: {event.actions.state_delta}", "32")
    colored_terminal_log("-" * 40, "30")

# --- Main Application Logic ---
st.title(f"🔍 {APP_NAME}")

# Sidebar for configuration and session loading
with st.sidebar:
    st.header("Configuration")
    user_id_input = st.text_input("Enter User ID", value=DEFAULT_USER_ID, key="user_id_input")
    st.markdown("---")
    st.header("Session Control")

    try:
        # App logic now calls the clean data access function
        sessions_response = database.list_sessions_sync(user_id_input)
        session_options = {
            f"{s.create_time.strftime('%Y-%m-%d %H:%M')} (ID: {s.id[:8]})": s.id
            for s in sessions_response.sessions if hasattr(s, 'create_time')
        }
    except Exception as e:
        st.error(f"Could not load sessions: {e}")
        session_options = {}

    session_keys = ["--- Start New Session ---"] + list(session_options.keys())

    def handle_session_change():
        selected_option = st.session_state.session_selectbox_key
        st.session_state.final_status_markdown = None # Reset verdict on change
        st.session_state.credibility_score = None
        if selected_option == "--- Start New Session ---":
            st.session_state.session_id = None
            st.session_state.chat_history = []
        else:
            selected_id = session_options.get(selected_option)
            if selected_id:
                st.session_state.session_id = selected_id
                # App logic calls the clean data access function
                st.session_state.chat_history = database.get_session_history_sync(selected_id, user_id_input)
        st.rerun()

    st.selectbox(
        "Load Session or Start New",
        options=session_keys,
        key="session_selectbox_key",
        on_change=handle_session_change
    )

    if st.session_state.session_id:
        st.info(f"Active Session ID: `{st.session_state.session_id}`")
    else:
        st.warning("No active session. Enter a query to start one.")

# --- Chat Interface ---
# This UI section remains unchanged
chat_container = st.container()
for message in st.session_state.chat_history:
    with chat_container.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if st.session_state.final_status_markdown:
    st.markdown("---")
    st.markdown(st.session_state.final_status_markdown, unsafe_allow_html=True)
    if st.session_state.credibility_score is not None:
        st.progress(st.session_state.credibility_score / 100.0, text=f"Credibility Score: {st.session_state.credibility_score}%")

# --- ADK Runner Logic ---
@st.cache_resource
def get_adk_runner():
    """Initializes and caches the ADK Runner."""
    colored_terminal_log("--- Initializing ADK Runner ---", "36")
    # The runner now gets the session_service imported from our database module
    return Runner(agent=root_agent, app_name="FactCheckerADK", session_service=database.session_service)

async def run_agent_pipeline(user_query, user_id):
    """Executes the ADK agent pipeline and updates Streamlit UI."""
    if not st.session_state.session_id:
        # The app asks the database module to create a session, without knowing the details
        new_session = await database.create_new_session(user_id)
        st.session_state.session_id = new_session.id
        colored_terminal_log(f"--- Created new session: {st.session_state.session_id} ---", "32")

    runner = get_adk_runner()
    
    with st.status(f"Initializing {APP_NAME}...", expanded=True) as status_message:
        try:
            full_response_parts_for_chat = []
            stage_messages = {
                "QueryProcessorAgent": "Step 1/4: Understanding your request...",
                "InfoGathererAgent": "Step 2/4: Searching the web and YouTube...",
                "WebSummarizerAgent": "Step 3/4: Analyzing web articles...",
                "VideoSummarizerAgent": "Step 3/4: Analyzing YouTube videos...",
                "FactRankerAgent": "Step 4/4: Formulating a verdict...",
            }
            
            async for event in runner.run_async(
                user_id=user_id,
                session_id=st.session_state.session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=user_query)]),
            ):
                log_event_to_terminal(event)
                status_message.update(label=stage_messages.get(event.author, "Processing..."))
                
                # The logic for processing the agent's final response remains the same
                if event.is_final_response() and event.author == "FactRankerAgent":
                    # ... (rest of the event processing logic is unchanged)
                    if event.content and event.content.parts:
                        final_json_string = event.content.parts[0].text
                        try:
                            fact_check_result = FactCheckResult.model_validate_json(final_json_string)
                            full_response_parts_for_chat.append(f"**Short Summary:** {fact_check_result.short_summary}")
                            full_response_parts_for_chat.append(f"**Detailed Explanation:**\n{fact_check_result.full_explanation}")
                            full_response_parts_for_chat.append("\n**Sources:**\n" + "\n".join([f"- <{s}>" for s in fact_check_result.sources]))

                            st.session_state.credibility_score = fact_check_result.credibility_score
                            
                            if fact_check_result.verdict == "Likely True":
                                st.session_state.final_status_markdown = "## ✅ **Verdict: Verified**\n<p style='font-size: smaller; color: grey;'>Our analysis indicates with high confidence that this claim is true based on the available sources.</p>"
                            elif fact_check_result.verdict == "Likely False":
                                st.session_state.final_status_markdown = "## ❌ **Verdict: Unverified**\n<p style='font-size: smaller; color: grey;'>Our analysis indicates with high confidence that this claim is false based on the available sources.</p>"
                            elif fact_check_result.verdict == "Mixed / Misleading":
                                st.session_state.final_status_markdown = "## ⚠️ **Verdict: Mixed / Misleading**\n<p style='font-size: smaller; color: grey;'>The claim contains elements of truth but is presented in a misleading way or lacks full context.</p>"
                            else:
                                st.session_state.final_status_markdown = f"## ❓ **Verdict: {fact_check_result.verdict}**\n<p style='font-size: smaller; color: grey;'>There was not enough credible information available to make a confident determination.</p>"
                        except Exception as e:
                            colored_terminal_log(f"Error processing final agent response: {e}", "31;1")
                            full_response_parts_for_chat.append(f"Error processing final output: `{e}`")
                            st.session_state.final_status_markdown = "## 🛑 **VERDICT: Error in Agent Output**"
                elif event.is_final_response() and event.content and event.content.parts and hasattr(event.content.parts[0], "text"):
                     full_response_parts_for_chat.append(event.content.parts[0].text.strip())
                     st.session_state.final_status_markdown = "## ℹ️ **VERDICT: Pipeline Completed (Generic Response)**"

            status_message.update(label="Fact-check complete!", state="complete", expanded=False)
            final_display_text = "\n\n".join(full_response_parts_for_chat)
            if final_display_text:
                st.session_state.chat_history.append({"role": "assistant", "content": final_display_text})
        except Exception as e:
            status_message.update(label="Agent Error!", state="error", expanded=True)
            st.error(f"An error occurred during agent execution: {e}")
            colored_terminal_log(f"FATAL ERROR IN APP: {e}", "31;1")
            st.session_state.chat_history.append({"role": "assistant", "content": f"I encountered a critical error: {e}"})
            st.session_state.final_status_markdown = "## 🛑 **VERDICT: Execution Failed**"

# User input at the bottom
user_input = st.chat_input("Enter your query:", key="user_query_input")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    asyncio.run(run_agent_pipeline(user_input, user_id_input))
    st.rerun()