import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_mines_mali import ask_details, load_env_file

app = FastAPI(title="Assistant Mines Mali API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    engine: str = "local"
    confidence: str = "faible"
    evidence_count: int = 0

@app.get("/health")
def health():
    return {"status": "ok", "service": "assistant-mines-mali"}

def normalize_encoding(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text

def public_answer(text: str) -> str:
    text = re.sub(r"\n\nSources utilisees:\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\nSources trouvees:.*", "", text, flags=re.DOTALL)
    return normalize_encoding(text).strip()

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        return AskResponse(answer="Pose une question sur les mines au Mali.")

    load_env_file()
    details = ask_details(question)
    return AskResponse(
        answer=public_answer(details["answer"]),
        engine=details["engine"],
        confidence=details["confidence"],
        evidence_count=len(details["sources"]),
    )
