from typing import Dict, Any
import json
from google.adk.tools.tool_context import ToolContext

def validate_fact_check_draft(draft_json: str) -> Dict[str, Any]:
    """
    Validates a draft of a FactCheckResult. It checks for logical consistency
    between the evidence, verdict, and score.

    Args:
        draft_json: A JSON string of the FactCheckResult draft.

    Returns:
        A dictionary with a 'status' ('pass' or 'fail') and a 'message'.
    """
    print("--- TOOL: Validating fact-check draft ---")
    try:
        data = json.loads(draft_json)
        verdict = data.get("verdict", "")
        score = data.get("credibility_score", 50)
        explanation = data.get("full_explanation", "").lower()

        # Rule 1: High scores require strong verdicts.
        if score > 85 and verdict not in ["Likely True", "Likely False"]:
            return {
                "status": "fail",
                "message": f"Credibility score is {score}, which is too high for a neutral verdict like '{verdict}'. High scores require a strong stance.",
            }

        # Rule 2: Low scores should not have strong verdicts.
        if score < 40 and verdict in ["Likely True", "Likely False"]:
            return {
                "status": "fail",
                "message": f"Credibility score is too low ({score}) for a strong verdict like '{verdict}'. The verdict should be 'Mixed' or 'Unverified'.",
            }
            
        # Rule 3: Mention of conflict should result in a lower score.
        if "conflict" in explanation and score > 65:
            return {
                "status": "fail",
                "message": "The explanation mentions conflicting sources, but the credibility score is too high. It should be below 65."
            }

        return {"status": "pass", "message": "Draft is logically consistent."}
    except json.JSONDecodeError:
        return {"status": "fail", "message": "The provided draft is not valid JSON."}
    except Exception as e:
        return {"status": "fail", "message": f"An unexpected error occurred during validation: {e}"}


# --- START: THE FIX ---
def exit_loop(tool_context: ToolContext):
    """
    Signals the LoopAgent to terminate its iterative process. This function takes
    no arguments from the LLM and returns nothing meaningful to it. Its only
    purpose is the side-effect of escalating control.
    """
    print("\n----------- EXIT LOOP TRIGGERED -----------")
    print("Fact-check review completed successfully. The loop will now exit.")
    print("------------------------------------------\n")

    # This action tells the parent LoopAgent to stop iterating.
    tool_context.actions.escalate = True
    
    # Return an empty dictionary to signify successful execution without a data payload.
    return {}
# --- END: THE FIX ---