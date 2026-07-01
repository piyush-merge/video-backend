from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess
import requests
import os


app = FastAPI()
DB = {}

# -------------------------
# MODELS
# -------------------------
class ProcessReq(BaseModel):
    url: str

class AskReq(BaseModel):
    video_id: str
    question: str


# -------------------------
# INIT WHISPER (SAFE ON STARTUP)
# -------------------------
model = WhisperModel("base", device="cpu", compute_type="int8")


# -------------------------
# DOWNLOAD VIDEO
# -------------------------
def download_video(url, vid):
    out = f"{vid}.mp4"
    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", out, url],
        check=True
    )
    return out


# -------------------------
# EXTRACT AUDIO
# -------------------------
def extract_audio(video, vid):
    audio = f"{vid}.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-ar", "16000", "-ac", "1", audio],
        check=True
    )
    return audio


# -------------------------
# TRANSCRIBE
# -------------------------
def transcribe(audio):
    segments, _ = model.transcribe(audio)
    return " ".join([s.text for s in segments])


# -------------------------
# FREE LLM (HF INFERENCE API)
# -------------------------
HF_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def llm(prompt):
    payload = {
        "inputs": prompt
    }

    r = requests.post(HF_API, headers=headers, json=payload, timeout=120)

    try:
        return r.json()[0]["generated_text"]
    except:
        return str(r.text)


# -------------------------
# PROCESS PIPELINE
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    vid = str(uuid.uuid4())

    video = download_video(req.url, vid)
    audio = extract_audio(video, vid)
    transcript = transcribe(audio)

    summary = llm(f"Summarize this transcript:\n{transcript[:8000]}")

    DB[vid] = {
        "transcript": transcript
    }

    return {
        "video_id": vid,
        "summary": summary,
        "transcript": transcript[:4000]
    }


# -------------------------
# Q&A
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    transcript = DB.get(req.video_id, {}).get("transcript", "")

    answer = llm(
        f"""
Use the transcript to answer the question.

Transcript:
{transcript[:8000]}

Question:
{req.question}
"""
    )

    return {"answer": answer}
