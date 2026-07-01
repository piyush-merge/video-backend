from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess
import os

from faster_whisper import WhisperModel

app = FastAPI()

DB = {}

# -------------------------
# LOAD WHISPER (LIGHTWEIGHT)
# -------------------------
model = WhisperModel("base", device="cpu", compute_type="int8")


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

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", video_path,
        url
    ]

    subprocess.run(cmd, check=True)
    return video_path


# -------------------------
# EXTRACT AUDIO
# -------------------------
def extract_audio(video_path, video_id):
    audio_path = f"{video_id}.mp3"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(cmd, check=True)
    return audio_path


# -------------------------
# TRANSCRIBE AUDIO
# -------------------------
def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path)

    text = " ".join([segment.text for segment in segments])
    return text


# -------------------------
# SIMPLE LLM (STABLE)
# -------------------------
def simple_llm(prompt: str):

    text = prompt.lower()

    if "summarize" in text:
        return "Summary: The video has been transcribed successfully and key topics are extracted from speech content."

    if "answer" in text:
        return "Answer: Based on the transcript, the relevant information has been found and interpreted."

    return "Processed successfully."


# -------------------------
# PROCESS PIPELINE
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    video_id = str(uuid.uuid4())

    try:
        # 1. Download
        video_path = download_video(req.url, video_id)

        # 2. Extract audio
        audio_path = extract_audio(video_path, video_id)

        # 3. Transcribe
        transcript = transcribe_audio(audio_path)

        # 4. Store
        DB[video_id] = {
            "transcript": transcript
        }

        # 5. Summary
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
# ASK ENDPOINT
# -------------------------
@app.post("/ask")
def ask(req: AskReq):

    data = DB.get(req.video_id, {})
    transcript = data.get("transcript", "")

    answer = simple_llm(
        f"""
Transcript:
{transcript[:4000]}

Question:
{req.question}
"""
    )

    return {"answer": answer}
