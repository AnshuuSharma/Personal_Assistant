import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from core.embeddings import create_embeddings

load_dotenv()

# session structure:
# {
#   "summary": str,
#   "all_turns": [...],
#   "history_store": [(embedding, turn), ...]
# }
sessions = {}
SUMMARIZE_AFTER = 6


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "summary": "",
            "all_turns": [],
            "history_store": []
        }
    return sessions[session_id]


def add_to_history(session_id, user_message, assistant_message):
    session = get_session(session_id)

    turn = {"user": user_message, "assistant": assistant_message}
    session["all_turns"].append(turn)

    # embed and store turn for semantic search
    turn_text = f"Q: {user_message} A: {assistant_message}"
    embedding = create_embeddings(turn_text)
    session["history_store"].append({
        "embedding": embedding,
        "turn": turn
    })

    # trigger summarization every 6 turns
    if len(session["all_turns"]) % SUMMARIZE_AFTER == 0:
        session["summary"] = _summarize(
            session["all_turns"],
            session["summary"]
        )


def get_memory(session_id, current_query):
    session = get_session(session_id)

    # running summary
    summary = session["summary"]

    # semantic search on history
    # find past turns relevant to current query
    relevant = _semantic_search_history(
        session["history_store"],
        current_query,
        top_k=2
    )

    # last 3 raw turns
    recent = session["all_turns"][-3:]

    # remove relevant turns that are already in recent
    recent_texts = [t["user"] for t in recent]
    relevant = [t for t in relevant if t["user"] not in recent_texts]

    return {
        "summary": summary,
        "relevant": relevant,   
        "recent": recent        
    }


def _semantic_search_history(history_store, query, top_k=2):
    if not history_store:
        return []

    query_embedding = create_embeddings(query)
    q = np.array(query_embedding)

    scores = []
    for item in history_store:
        h = np.array(item["embedding"])
        score = np.dot(q, h) / (np.linalg.norm(q) * np.linalg.norm(h) + 1e-9)
        scores.append((score, item["turn"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [turn for _, turn in scores[:top_k]]


def _summarize(all_turns, existing_summary):
    convo = ""
    for turn in all_turns[-6:]:  
        convo += f"Recruiter: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = f"""
Summarize this conversation between a recruiter and an AI assistant 
representing job candidate Anshu.

Previous summary:
{existing_summary if existing_summary else "None"}

New turns:
{convo}

Write 3-4 lines capturing:
- Topics discussed
- Recruiter's interests
- Important details mentioned

Summary:"""

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summarization failed: {str(e)}")
        return existing_summary


def clear_history(session_id):
    if session_id in sessions:
        del sessions[session_id]