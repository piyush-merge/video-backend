from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import yt_dlp
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")  # optional but recommended


class ProcessRequest(BaseModel):
    url: str


def download_audio(url: str, file_id: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{file_id}.%(ext)s",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename


def transcribe_with_hf(audio_path: str):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    with open(audio_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)

    if response.status_code != 200:
        return "Transcription failed (HF API error)"

    return response.json().get("text", "No transcript returned")


def summarize_text(text: str):
    # simple fallback summary (no LLM dependency)
    return "Summary: " + text[:300]


@app.post("/process")
def process(req: ProcessRequest):
    video_id = str(uuid.uuid4())

    audio_file = download_audio(req.url, video_id)

    transcript = transcribe_with_hf(audio_file)
    summary = summarize_text(transcript)

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }
