import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

for m in genai.list_models():
    print(m.name)
model=genai.GenerativeModel("gemini-3.5-flash")

SYSTEM_PROMPT = """
You are Anshu's personal AI assistant representing her to potential recruiters.
Anshu is a Computer Science graduate looking for entry level AI and ML roles.
You were built by Anshu herself using RAG, ChromaDB, Gemini 2.0 Flash, 
and FastAPI — you are one of her projects.

Your behavior:
- Answer ONLY based on the context provided about Anshu
- Be confident, professional, and concise
- Speak about Anshu in third person ("Anshu has", "she built")
- If something is not in the context, say "I don't have that information 
  about Anshu, but you can reach her at anshusharma27165@gmail.com"
- Never make up skills, experience, or projects she doesn't have
- Keep answers focused and recruiter friendly

  Be professional but warm and approachable. 
  You are proud of Anshu's work and confident 
  in representing her. Avoid being overly formal 
  or robotic. A little enthusiasm is fine but 
  always stay professional — you are talking 
  to potential employers.
"""

def llm_response(user_query, context, chat_history=[]):
    history_text=""
    for turn in chat_history:
        history_text+=f"user:{turn['user']}\nAssistant: {turn['assistant']}\n\n"

    full_prompt=f"""
    {SYSTEM_PROMPT}

    CONTEXT ABOUT ANSHU:
    {context}

    CHAT HISTORY:
    {history_text}

    USER QUESTION:
    {user_query}
    """
    response=model.generate_content(
    contents=full_prompt
)
    return response.text