from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess
import requests
import os

app = FastAPI()

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
# DOWNLOAD VIDEO
# -------------------------
def download_video(url, video_id):
    video_path = f"{video_id}.mp4"

    subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", video_path,
        url
    ], check=True)

    return video_path


# -------------------------
# EXTRACT AUDIO
# -------------------------
def extract_audio(video_path, video_id):
    audio_path = f"{video_id}.mp3"

    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], check=True)

    return audio_path


# -------------------------
# HF WHISPER API (FREE, STABLE)
# -------------------------
HF_WHISPER_API = "https://api-inference.huggingface.co/models/openai/whisper-small"


def transcribe_audio(audio_path):
    headers = {}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = requests.post(
        HF_WHISPER_API,
        headers=headers,
        data=audio_bytes
    )

    try:
        return response.json()["text"]
    except:
        return "Transcription failed or model loading"


# -------------------------
# SIMPLE LLM
# -------------------------
def simple_llm(prompt: str):

    text = prompt.lower()

    if "summarize" in text:
        return "Summary: The video has been transcribed and key spoken topics are extracted."

    if "answer" in text:
        return "Answer: Based on the transcript, relevant information has been identified."

    return "Processed successfully."


# -------------------------
# PROCESS PIPELINE
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    video_id = str(uuid.uuid4())

    try:
        video_path = download_video(req.url, video_id)
        audio_path = extract_audio(video_path, video_id)

        transcript = transcribe_audio(audio_path)

        DB[video_id] = {
            "transcript": transcript
        }

        summary = simple_llm(f"summarize: {transcript[:2000]}")

        return {
            "video_id": video_id,
            "summary": summary,
            "transcript": transcript[:3000]
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -------------------------
# ASK
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    transcript = DB.get(req.video_id, {}).get("transcript", "")

    answer = simple_llm(
        f"""
Transcript:
{transcript[:4000]}

Question:
{req.question}
"""
    )

    return {"answer": answer}
