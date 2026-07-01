from fastapi import FastAPI
from pydantic import BaseModel
import yt_dlp
import uuid
import os

app = FastAPI()

model = None  # lazy load fix

class Req(BaseModel):
    url: str | None = None
    question: str | None = None


# -------------------------
# LOAD WHISPER ONLY WHEN NEEDED
# -------------------------
def get_model():
    global model
    if model is None:
        import whisper
        model = whisper.load_model("base")
    return model


# -------------------------
# DOWNLOAD AUDIO
# -------------------------
def download_audio(url: str):
    filename = f"{uuid.uuid4()}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename,
        "quiet": True,
        "noplaylist": True,
        "retries": 3
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filename


# -------------------------
# TRANSCRIBE
# -------------------------
def transcribe_audio(path: str):
    model = get_model()
    result = model.transcribe(path)
    return result["text"]


# -------------------------
# SUMMARY (TEMP LOGIC)
# -------------------------
def summarize(text: str):
    sentences = text.split(".")
    return "\n".join([s.strip() for s in sentences[:8] if s.strip()])


# -------------------------
# Q&A (SIMPLE MATCH)
# -------------------------
def answer(question: str, transcript: str):
    if not question:
        return None

    q_words = question.lower().split()
    sentences = transcript.split(".")

    for s in sentences:
        if any(w in s.lower() for w in q_words):
            return s.strip()

    return sentences[0][:300] if sentences else ""


# -------------------------
# MAIN ENDPOINT
# -------------------------
@app.post("/process")
def process(req: Req):

    if not req.url:
        return {"error": "url missing"}

    audio = download_audio(req.url)
    transcript = transcribe_audio(audio)

    summary = summarize(transcript)
    answer_text = answer(req.question, transcript)

    if os.path.exists(audio):
        os.remove(audio)

    return {
        "transcript": transcript,
        "summary": summary,
        "answer": answer_text
    }
