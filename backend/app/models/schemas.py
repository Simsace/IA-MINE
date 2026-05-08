from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question in French or another language.")
    top_k: int | None = Field(default=None, ge=1, le=10, description="Number of chunks to retrieve.")
    user_id: str | None = Field(default=None, description="Future-ready user identifier.")
    chat_id: str | None = Field(default=None, description="Future-ready chat/session identifier.")


class SourceChunk(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: float | None = None
    engine: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str

