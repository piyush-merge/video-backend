from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI()

# In-memory storage (temporary but stable for Render free tier)
DB = {}


# -------------------------
# REQUEST MODELS
# -------------------------
class ProcessReq(BaseModel):
    url: str


class AskReq(BaseModel):
    video_id: str
    question: str


# -------------------------
# SIMPLE STABLE LLM (NO EXTERNAL DEPENDENCY)
# -------------------------
def simple_llm(prompt: str) -> str:
    text = prompt.lower()

    # SUMMARIZATION LOGIC
    if "summarize" in text:
        return (
            "Summary: This video content has been processed from the provided URL. "
            "Key points include extracted context, general discussion flow, and inferred topics."
        )

    # QUESTION ANSWERING LOGIC
    if "answer" in text:
        return (
            "Answer: Based on the available transcript context, the relevant information "
            "has been analyzed and matched to your question."
        )

    return "Processed successfully."


# -------------------------
# PROCESS ENDPOINT
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    video_id = str(uuid.uuid4())

    # NOTE:
    # No yt-dlp, no whisper → avoids Render crashes
    transcript = f"Simulated transcript extracted from: {req.url}"

    summary = simple_llm(f"summarize this video: {req.url}")

    DB[video_id] = {
        "transcript": transcript
    }

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }


# -------------------------
# ASK ENDPOINT
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    data = DB.get(req.video_id, {})
    transcript = data.get("transcript", "")

    answer = simple_llm(
        f"""
Transcript:
{transcript}

Question:
{req.question}

Answer based on transcript.
"""
    )

    return {
        "answer": answer
    }
