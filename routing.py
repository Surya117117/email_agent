def approval_router(state):
    """Route graph execution based on the user's decision."""

    action = state["decision"]["action"].lower()

    if action == "approve":
        return "send_email"
    
    elif action == "edit":
        return "edit_reply"
    
    elif action == "reject":
        return "regenerate_reply"
    
    else:
        raise ValueError(f"Unknown action: {action}")
    