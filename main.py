from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI()

DB = {}


class ProcessReq(BaseModel):
    url: str


class AskReq(BaseModel):
    video_id: str
    question: str


# -------------------------
# ORCHESTRATOR ONLY
# -------------------------
@app.post("/process")
def process(req: ProcessReq):

    video_id = str(uuid.uuid4())

    DB[video_id] = {
        "url": req.url,
        "status": "queued"
    }

    return {
        "video_id": video_id,
        "status": "queued",
        "message": "Send this URL to Colab worker for processing"
    }


@app.post("/ask")
def ask(req: AskReq):

    return {
        "answer": "Q&A disabled in orchestrator mode. Use worker pipeline."
    }
