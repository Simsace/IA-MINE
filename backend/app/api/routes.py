import logging

from app.api.dependencies import optional_auth_dependency
from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.rag.rag_service import RagService
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter()
rag_service = RagService()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="mines-mali-rag-api")


@router.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask(
    payload: AskRequest,
    _: dict = Depends(optional_auth_dependency),
) -> AskResponse:
    """Ask a question about mining in Mali.

    The answer is generated only from retrieved internal documents.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        return await run_in_threadpool(
            rag_service.answer_question,
            question,
            payload.top_k,
            payload.chat_id,
            payload.user_id,
        )
    except FileNotFoundError as exc:
        logger.exception("RAG index is missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG index is not ready: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to answer question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while answering the question.",
        ) from exc

