from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess
import os

app = FastAPI()

DB = {}


# -------------------------
# REQUESTS
# -------------------------
class ProcessReq(BaseModel):
    url: str


class AskReq(BaseModel):
    video_id: str
    question: str


# -------------------------
# DOWNLOAD VIDEO (REAL)
# -------------------------
def download_video(url, video_id):
    output = f"{video_id}.mp4"

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output,
        url
    ]

    subprocess.run(cmd, check=True)
    return output


# -------------------------
# PLACEHOLDER LLM (STABLE)
# -------------------------
def simple_llm(prompt: str):
    if "summarize" in prompt.lower():
        return "Summary: Video downloaded and processed successfully. Content is ready for transcription phase."

    if "answer" in prompt.lower():
        return "Answer: Based on current processed video context, relevant information is extracted."

    return "Processed successfully."


# -------------------------
# PROCESS ENDPOINT
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    video_id = str(uuid.uuid4())

    try:
        file_path = download_video(req.url, video_id)

        DB[video_id] = {
            "file_path": file_path,
            "transcript": "Pending transcription phase (Phase 2 upgrade)"
        }

        summary = simple_llm(f"summarize video {req.url}")

        return {
            "video_id": video_id,
            "summary": summary,
            "transcript": "Download successful. Transcription not yet enabled."
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -------------------------
# ASK ENDPOINT
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    data = DB.get(req.video_id, {})

    answer = simple_llm(
        f"""
Video context:
{data.get('transcript', '')}

Question:
{req.question}
"""
    )

    return {"answer": answer}
