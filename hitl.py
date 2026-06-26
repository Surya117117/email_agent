from langgraph.types import interrupt

def human_intervention(state):
    """Pause execution and wait for human approval."""

    draft = state["draft_reply"]
    decision = interrupt(
        {
            "draft_reply": draft,
            "message": (
                "Review the draft and choose one of the following:\n"
                "- approve\n"
                "- edit\n"
                "- reject"
            )
        }
    )
    return {
        "decision": decision
    }