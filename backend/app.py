import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context
from core.llm import llm_response, llm_response_stream
from core.memory import get_memory, add_to_history
from agent.router import route
from tools.resumeiq_tool import call_resumeiq

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi.responses import StreamingResponse
import json

import time
import traceback

from core.retriever import OUT_OF_SCOPE_MESSAGE,CLARIFY_INSTRUCTION


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




# ─── serving frontend static files  ───────────────
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/index.html")

# ─── HEALTH CHECK ────────────────────────────────────────────

@app.get("/health")
async def root():
    return {"status": "Anshu's personal assistant is running!"}


# ─── CHAT ENDPOINT ───────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.message
    session_id = request.session_id
 
    t0=time.time()
 
    memory = get_memory(session_id, user_query)
 
    t1=time.time()
    print(f"Memory retrieval : {t1-t0:.3f}s")
    context, confidence = route(user_query)
 
    t2=time.time()
    print(f"Rag retrieval : {t2-t1:.3f}s (confidence={confidence})")
 
    if confidence == "out_of_scope":
        def generate_out_of_scope():
            yield f"data: {json.dumps({'text': OUT_OF_SCOPE_MESSAGE, 'done': False})}\n\n"
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        return StreamingResponse(
            generate_out_of_scope(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
 
    full_response=""
    def generate():
        nonlocal full_response
        try:
            provider, stream = llm_response_stream(user_query, context, memory)
            print(f"DEBUG: provider={provider}, stream={stream}")
 
            if stream is None:
               print("DEBUG: stream is None, sending fallback message")
               yield f"data: {json.dumps({'text': 'I am a little busy right now — please try again!', 'done': True})}\n\n"
               return
 
            if provider == "gemini":
                for chunk in stream:
                 print(f"DEBUG: gemini chunk = {chunk}")
                 if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text, 'done': False})}\n\n"
            else:  # groq
              for chunk in stream:
                print(f"DEBUG: groq chunk = {chunk}")
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'text': delta, 'done': False})}\n\n"
 
            add_to_history(session_id, user_query, full_response)
            print(f"DEBUG: full_response = {full_response[:100]}")
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
 
        except Exception as e:
           print(f"Streaming error: {str(e)[:200]}")
           yield f"data: {json.dumps({'text': 'Something went wrong — please try again shortly!', 'done': True})}\n\n"
 
        print(f"Total pre-stream:{t2-t0:.3f}s")
 
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

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
    chunks, confidence = retrieve(query_embedding, n_results=n, query_text=query)
 
    if confidence == "out_of_scope":
        # shouldn't realistically happen for fixed button queries -- but
        # handle it defensively rather than crash downstream
        def generate_out_of_scope():
            yield f"data: {json.dumps({'text': OUT_OF_SCOPE_MESSAGE, 'done': False})}\n\n"
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        return StreamingResponse(
            generate_out_of_scope(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
 
    context = format_context(chunks)
    if confidence == "low_confidence":
        context += CLARIFY_INSTRUCTION  
 
    full_response = ""
 
    def generate():
        nonlocal full_response
        try:
            provider, stream = llm_response_stream(query, context, memory={})
 
            if stream is None:
                yield f"data: {json.dumps({'text': 'Something went wrong — please try again!', 'done': True})}\n\n"
                return
 
            if provider == "gemini":
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'text': chunk.text, 'done': False})}\n\n"
            else:  # groq fallback
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        yield f"data: {json.dumps({'text': delta, 'done': False})}\n\n"
 
            add_to_history(session_id, query, full_response)
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
 
        except Exception as e:
             traceback.print_exc()
             yield f"data: {json.dumps({'text': 'Something went wrong — please try again shortly!', 'done': True})}\n\n"
 
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
# ─── RESUMEIQ ENDPOINT ───────────────────────────────────────

@app.post("/analyze")
async def analyze(request: ResumeIQRequest):
    session_id = request.session_id
    job_description = request.job_description

    result = call_resumeiq(job_description)
    
    print(f"ResumeIQ result keys: {result.keys()}")  
    print("full error:",result)     
    print(f"Analysis length: {len(result.get('analysis', ''))}")

    if len(job_description.split()) < 30:
        return {
            "response": """That job description seems too short 
    for a meaningful analysis. Could you paste the full JD 
    with requirements and responsibilities? The more detail 
    you provide, the more accurate the analysis will be!"""
        }


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
        memory={}
    )

    add_to_history(session_id, "JD Analysis Request", response)

    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

