# app.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.core.embeddings import create_embeddings
from backend.core.retriever import retrieve, format_context
from backend.core.llm import llm_response
from backend.core.memory import get_recent_history, add_to_history
from backend.agent.router import route
from backend.tools.resumeiq_tool import call_resumeiq

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── REQUEST MODELS ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ButtonRequest(BaseModel):
    topic: str
    session_id: str

class ResumeIQRequest(BaseModel):
    job_description: str
    session_id: str


# ─── HEALTH CHECK ────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "Anshu's personal assistant is running!"}


# ─── CHAT ENDPOINT ───────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.message
    session_id = request.session_id

    memory = get_memory(session_id, user_query)
    context = route(user_query)
    response = llm_response(user_query, context, memory)
    add_to_history(session_id, user_query, response)

    return {"response": response}


# ─── BUTTON ENDPOINT ─────────────────────────────────────────

@app.post("/button")
async def button(request: ButtonRequest):
    session_id = request.session_id

    button_queries = {
        "skills":         "What are all of Anshu's technical skills?",
        "education":      "Tell me about Anshu's educational background",
        "experience":     "Tell me about Anshu's work experience",
        "projects":       "Tell me about all of Anshu's projects in detail including ResumeIQ and personal assistant",
        "certifications": "What certifications does Anshu have?",
        "contact":        "How can I contact Anshu or reach out to her?"
    }

    query = button_queries.get(request.topic)

    if not query:
        return {"response": "Invalid button topic"}

    query_embedding = create_embeddings(query)

    n = 6 if request.topic == "projects" else 4
    chunks = retrieve(query_embedding, n_results=n)

    print(f"\n--- DEBUG ---")
    print(f"Topic: {request.topic}")
    print(f"Query: {query}")
    print(f"Chunks retrieved: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:80]}...")
    print(f"--- END DEBUG ---\n")
    
    context = format_context(chunks)

    response = llm_response(query, context, chat_history=[])

    add_to_history(session_id, query, response)

    return {"response": response}


# ─── RESUMEIQ ENDPOINT ───────────────────────────────────────

@app.post("/analyze")
async def analyze(request: ResumeIQRequest):
    session_id = request.session_id
    job_description = request.job_description

    result = call_resumeiq(job_description)
    
    print(f"ResumeIQ result keys: {result.keys()}")  
    print(f"Error in result: {'error' in result}")     
    print(f"Analysis length: {len(result.get('analysis', ''))}")

    if len(job_description.split()) < 30:
        return {
            "response": """That job description seems too short 
    for a meaningful analysis. Could you paste the full JD 
    with requirements and responsibilities? The more detail 
    you provide, the more accurate the analysis will be!"""
        }

    result = call_resumeiq(job_description)

    if "error" in result:
        return {
            "response": f"""Hmm, I wasn't able to run the analysis 
            right now. You can try Anshu's ResumeIQ directly at 
            https://agentic-resume-analyzer.netlify.app/ or reach 
            her at anshusharma27165@gmail.com"""
        }

    context = f"""
    ResumeIQ has analyzed Anshu's resume against the provided job description.
    Here is the full analysis:
    {result['analysis']}
    """

    response = llm_response(
        user_query="""Based on the ResumeIQ analysis provided, give an honest assessment of how Anshu fits this role.
        Cover these points:
        - ATS match score and whether she passed
        - Skills she has that match the role
        - Skills that are missing or gaps identified  
        - Overall honest fit assessment

        If the fit is weak, say so honestly but constructively.
        A recruiter needs accurate information to make good decisions.""",
        context=context,
        chat_history=[]
    )

    add_to_history(session_id, "JD Analysis Request", response)

    return {"response": response}