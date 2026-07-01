from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import os
import subprocess
import requests

from faster_whisper import WhisperModel

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

DB = {}

# -------------------------
# REQUEST SCHEMA (URL ONLY)
# -------------------------
class URLRequest(BaseModel):
    url: str


# -------------------------
# DOWNLOAD FROM URL
# -------------------------
def download_from_url(url, video_id):
    output = f"{video_id}.mp4"

    cmd = [
        "yt-dlp",
        "-o", output,
        url
    ]

    subprocess.run(cmd, check=True)
    return output


# -------------------------
# SAVE UPLOADED FILE
# -------------------------
def save_upload(file: UploadFile, video_id):
    path = f"{video_id}.mp4"

    with open(path, "wb") as f:
        f.write(file.file.read())

    return path


# -------------------------
# TRANSCRIBE
# -------------------------
def transcribe(path):
    segments, _ = model.transcribe(path)

    text = ""
    for s in segments:
        text += s.text + " "

    return text.strip()


# -------------------------
# SIMPLE LLM (HF FREE)
# -------------------------
HF_API = "https://api-inference.huggingface.co/models/google/flan-t5-large"

def llm(prompt):
    r = requests.post(HF_API, json={"inputs": prompt})

    try:
        return r.json()[0]["generated_text"]
    except:
        return "LLM error"


def summarize(text):
    return llm(f"Summarize:\n{text[:4000]}")


def answer(text, question):
    return llm(f"Answer using transcript:\n{text[:4000]}\nQ: {question}")


# -------------------------
# PROCESS URL
# -------------------------
@app.post("/process")
def process_url(req: URLRequest):

    video_id = str(uuid.uuid4())

    file_path = download_from_url(req.url, video_id)

    transcript = transcribe(file_path)
    summary = summarize(transcript)

    DB[video_id] = {"transcript": transcript}

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }


# -------------------------
# PROCESS FILE UPLOAD
# -------------------------
@app.post("/process-file")
def process_file(file: UploadFile = File(...)):

    video_id = str(uuid.uuid4())

    file_path = save_upload(file, video_id)

    transcript = transcribe(file_path)
    summary = summarize(transcript)

    DB[video_id] = {"transcript": transcript}

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }


# -------------------------
# ASK
# -------------------------
class AskRequest(BaseModel):
    video_id: str
    question: str


@app.post("/ask")
def ask(req: AskRequest):

    text = DB.get(req.video_id, {}).get("transcript", "")

    return {
        "answer": answer(text, req.question)
    }
