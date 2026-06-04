import httpx
import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "Resume_Anshu.pdf")
RESUMEIQ_URL = os.getenv("RESUMEIQ_URL")  

print(f"RESUMEIQ_URL: {RESUMEIQ_URL}")
print(f"RESUME_PATH: {RESUME_PATH}")
print(f"Resume exists: {os.path.exists(RESUME_PATH)}")

def call_resumeiq(job_description: str):
    try:
      
        with open(RESUME_PATH, "rb") as resume_file:
           
            files = {
                "resume": ("anshu_resume.pdf", resume_file, "application/pdf")
            }
            data = {
                "jd_text": job_description
            }

            response = httpx.post(
                f"{RESUMEIQ_URL}/analyze",
                files=files,
                data=data,
                timeout=60  
            )

        print(f"Status: {response.status_code}") 
        print(f"Response text: {response.text[:200]}") 

        if response.status_code != 200:
            return {"error": f"ResumeIQ returned status {response.status_code}"}

        result = response.json()
        print(f"Result keys: {result.keys()}")   

        return {
            "analysis": result.get("analysis", ""),
            "session_id": result.get("session_id", "")
        }

    except httpx.TimeoutException:
        return {"error": "ResumeIQ took too long to respond"}

    except FileNotFoundError:
        return {"error": "Resume PDF not found"}

    except Exception as e:
        print(f"ResumeIQ tool error: {str(e)}") 
        return {"error": str(e)}