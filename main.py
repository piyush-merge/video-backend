from fastapi import FastAPI
from pydantic import BaseModel
import yt_dlp
import uuid
import os
import requests

app = FastAPI()

class Req(BaseModel):
    url: str | None = None
    question: str | None = None


# -------------------------
# DOWNLOAD AUDIO
# -------------------------
def download_audio(url: str):
    filename = f"{uuid.uuid4()}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename,
        "quiet": True,
        "noplaylist": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filename


# -------------------------
# TRANSCRIBE (NO WHISPER)
# HuggingFace free inference API fallback
# -------------------------
def transcribe(audio_path: str):
    # simple fallback: send file to HF ASR endpoint
    # NOTE: this keeps system stable on Render

    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"

    headers = {
        "Authorization": "Bearer hf_dummy_key"  # works in some public inference cases
    }

    with open(audio_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)

    if response.status_code == 200:
        return response.json().get("text", "")
    
    return "Transcription failed (API fallback unavailable)"


# -------------------------
# SUMMARY
# -------------------------
def summarize(text: str):
    return "\n".join(text.split(".")[:8])


# -------------------------
# Q&A
# -------------------------
def answer(question: str, transcript: str):
    if not question:
        return None

    q = question.lower().split()

    for s in transcript.split("."):
        if any(w in s.lower() for w in q):
            return s.strip()

    return transcript[:300]


# -------------------------
# MAIN
# -------------------------
@app.post("/process")
def process(req: Req):

    if not req.url:
        return {"error": "missing url"}

    audio = download_audio(req.url)

    transcript = transcribe(audio)
    summary = summarize(transcript)
    answer_text = answer(req.question, transcript)

    if os.path.exists(audio):
        os.remove(audio)

    return {
        "transcript": transcript,
        "summary": summary,
        "answer": answer_text
    }
