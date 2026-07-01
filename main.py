from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import os
import subprocess
import requests

from faster_whisper import WhisperModel

app = FastAPI()

# ----------------------------
# LOAD WHISPER MODEL
# ----------------------------
model = WhisperModel("base", device="cpu", compute_type="int8")

# ----------------------------
# REQUEST SCHEMAS
# ----------------------------
class ProcessRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    video_id: str
    question: str


# ----------------------------
# IN-MEMORY STORAGE (simple v1)
# later replaced with Drive/DB
# ----------------------------
DB = {}


# ----------------------------
# DOWNLOAD VIDEO AUDIO
# ----------------------------
def download_audio(url, video_id):
    output = f"{video_id}.mp3"

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", output,
        url
    ]

    subprocess.run(cmd, check=True)
    return output


# ----------------------------
# TRANSCRIBE AUDIO
# ----------------------------
def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path)

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text.strip()


# ----------------------------
# SIMPLE LLM (FREE VIA HF INFERENCE)
# ----------------------------
HF_API = "https://api-inference.huggingface.co/models/google/flan-t5-large"

def call_llm(prompt):
    response = requests.post(
        HF_API,
        json={"inputs": prompt}
    )

    try:
        return response.json()[0]["generated_text"]
    except:
        return "LLM error or rate limit."


# ----------------------------
# SUMMARIZE TRANSCRIPT
# ----------------------------
def summarize(text):
    prompt = f"""
Summarize the following transcript in clear bullet points:

{text[:4000]}
"""
    return call_llm(prompt)


# ----------------------------
# ANSWER QUESTIONS (RAG-lite)
# ----------------------------
def answer_question(transcript, question):
    prompt = f"""
Use the transcript to answer the question.

Transcript:
{transcript[:4000]}

Question:
{question}
"""
    return call_llm(prompt)


# ----------------------------
# PROCESS ENDPOINT
# ----------------------------
@app.post("/process")
def process(req: ProcessRequest):

    video_id = str(uuid.uuid4())

    audio_file = download_audio(req.url, video_id)
    transcript = transcribe_audio(audio_file)
    summary = summarize(transcript)

    DB[video_id] = {
        "transcript": transcript,
        "summary": summary
    }

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }


# ----------------------------
# ASK ENDPOINT
# ----------------------------
@app.post("/ask")
def ask(req: AskRequest):

    if req.video_id not in DB:
        return {"error": "video_id not found"}

    transcript = DB[req.video_id]["transcript"]

    answer = answer_question(transcript, req.question)

    return {
        "answer": answer
    }
