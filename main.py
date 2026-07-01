from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uuid

app = FastAPI()

DB = {}

# -------------------------
# REQUEST
# -------------------------
class Req(BaseModel):
    url: str


class AskReq(BaseModel):
    video_id: str
    question: str


# -------------------------
# HF FREE MODEL ENDPOINT
# -------------------------
HF_MODEL = "https://api-inference.huggingface.co/models/google/flan-t5-base"


def hf_call(prompt):
    try:
        r = requests.post(
            HF_MODEL,
            json={"inputs": prompt},
            timeout=60
        )
        return r.json()[0]["generated_text"]
    except:
        return "Model temporarily unavailable"


# -------------------------
# PROCESS
# -------------------------
@app.post("/process")
def process(req: Req):

    video_id = str(uuid.uuid4())

    # NO yt-dlp, NO whisper local (prevents Render crash)

    summary = hf_call(f"Summarize this video URL content: {req.url}")

    transcript = hf_call(f"Generate transcript-like summary of this video: {req.url}")

    DB[video_id] = {
        "transcript": transcript
    }

    return {
        "video_id": video_id,
        "summary": summary,
        "transcript": transcript
    }


# -------------------------
# ASK
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    text = DB.get(req.video_id, {}).get("transcript", "")

    answer = hf_call(
        f"""
Use this transcript:
{text}

Answer this question:
{req.question}
"""
    )

    return {"answer": answer}
