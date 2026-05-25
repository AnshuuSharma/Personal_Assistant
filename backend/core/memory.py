sessions={}

def get_history(session_id):
    return sessions.get(session_id,[])

def add_to_history(session_id, user_message, assistant_message):
    if session_id not in sessions:
        sessions[session_id]=[]

    sessions[session_id].append({
        "user":user_message,
        "assistant":assistant_message
    })

def get_recent_history(session_id, max_turns=6):
    history=get_history(session_id)
    return history[-max_turns:]

def clear_history(session_id):
    if session_id in sessions:
        del sessions[session_id]
        