import json
from typing import Dict, Any
from fact_checker_agent.logger import get_logger, log_tool_call, log_success, log_warning, log_error
from google.adk.tools.tool_context import ToolContext


logger = get_logger(__name__)

def validate_fact_check_draft(draft_json: str) -> Dict[str, Any]:
    """
    Validates a draft of a FactCheckResult for logical consistency.
    """
    log_tool_call(logger, "validate_fact_check_draft", "Validating draft...")
    try:
        data = json.loads(draft_json)
        verdict = data.get("verdict", "")
        score = data.get("credibility_score", 50)
        explanation = data.get("full_explanation", "").lower()

        if score > 85 and verdict not in ["Likely True", "Likely False"]:
            msg = f"Credibility score is {score}, which is too high for a neutral verdict like '{verdict}'. High scores require a strong stance."
            log_warning(logger, f"Validation Failed: {msg}")
            return {"status": "fail", "message": msg}

        if score < 40 and verdict in ["Likely True", "Likely False"]:
            msg = f"Credibility score is too low ({score}) for a strong verdict like '{verdict}'. The verdict should be 'Mixed' or 'Unverified'."
            log_warning(logger, f"Validation Failed: {msg}")
            return {"status": "fail", "message": msg}
            
        if "conflict" in explanation and score > 65:
            msg = "The explanation mentions conflicting sources, but the credibility score is too high. It should be below 65."
            log_warning(logger, f"Validation Failed: {msg}")
            return {"status": "fail", "message": msg}

        log_success(logger, "Validation Passed: Draft is logically consistent.")
        return {"status": "pass", "message": "Draft is logically consistent."}
    except json.JSONDecodeError:
        log_error(logger, "Validation Failed: The provided draft is not valid JSON.")
        return {"status": "fail", "message": "The provided draft is not valid JSON."}
    except Exception as e:
        log_error(logger, f"Validation Failed: An unexpected error occurred: {e}")
        return {"status": "fail", "message": f"An unexpected error occurred during validation: {e}"}


def exit_loop(tool_context: ToolContext):
    """
    Signals the LoopAgent to terminate its iterative process.
    """
    log_tool_call(logger, "exit_loop", "--- EXIT LOOP TRIGGERED ---")
    tool_context.actions.escalate = True
    return {}