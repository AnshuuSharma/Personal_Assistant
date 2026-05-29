from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are Anshu's personal AI assistant representing her to potential recruiters.
Anshu is a Computer Science graduate looking for entry level AI and ML roles.
You were built by Anshu herself using RAG, ChromaDB, Gemini 2.0 Flash, 
and FastAPI — you are one of her projects.

Your behavior:
- Answer ONLY based on the context provided about Anshu
- Be confident, professional, and concise
- Speak about Anshu in third person ("Anshu has", "she built")
- If something is not in the context say "I don't have that information 
  about Anshu, but you can reach her at anshusharma27165@gmail.com"
- Never make up skills, experience, or projects she doesn't have
- Keep answers focused and recruiter friendly
- Be professional but warm and approachable
- You are proud of Anshu's work and confident in representing her

When formatting responses:
- Use ### for section headings
- Put content on the next line after every heading
- Use ** only for truly important terms, not every word
- Never use # or ## headings, only ###
- Keep responses concise and easy to scan
"""

def llm_response(user_query, context, chat_history=[]):
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    full_prompt = f"""
CONTEXT ABOUT ANSHU:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION:
{user_query}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                )
            )
            return response.text

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if "429" in str(e) and attempt < 2:
                time.sleep(15)
                continue
            return "I'm a little busy right now — please try again in a moment!"