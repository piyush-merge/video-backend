from fastapi import FastAPI
from pydantic import BaseModel
import yt_dlp
import uuid
import os
import traceback

app = FastAPI()

model = None

class Req(BaseModel):
    url: str | None = None
    question: str | None = None


# -------------------------
# LAZY LOAD WHISPER
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
        "quiet": False,
        "noplaylist": True,
        "retries": 3,
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
# SIMPLE SUMMARY
# -------------------------
def summarize(text: str):
    return "\n".join(text.split(".")[:8])


# -------------------------
# SIMPLE Q&A
# -------------------------
def answer(question: str, transcript: str):
    if not question:
        return None

    q = question.lower().split()
    for s in transcript.split("."):
        if any(w in s.lower() for w in q):
            return s

    return transcript[:300]


# -------------------------
# MAIN ENDPOINT (NOW WITH DEBUG)
# -------------------------
@app.post("/process")
def process(req: Req):

    try:
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

    except Exception as e:
        # CRITICAL: expose real Render error
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
