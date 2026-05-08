import logging

import numpy as np
from app.core.config import get_settings
from app.models.schemas import AskResponse, SourceChunk
from app.rag.build_rag import configure_huggingface_cache, get_model_path
from app.rag.faiss_store import FaissStore
from app.services.llm_service import LlmService
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = FaissStore()
        self.llm_service = LlmService()
        self._embedding_model: SentenceTransformer | None = None

    def _model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            configure_huggingface_cache()
            self._embedding_model = SentenceTransformer(get_model_path(self.settings.embedding_model_name))
        return self._embedding_model

    def _embed_question(self, question: str) -> np.ndarray:
        embedding = self._model().encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")

    def retrieve(self, question: str, top_k: int | None = None) -> list[Document]:
        index = self.store.load_index()
        metadata = self.store.load_metadata()
        chunks = metadata["chunks"]

        k = top_k or self.settings.default_top_k
        k = min(k, self.settings.max_top_k)

        query_embedding = self._embed_question(question)
        scores, ids = index.search(query_embedding, k)

        docs: list[Document] = []
        for chunk_id, score in zip(ids[0], scores[0]):
            if chunk_id < 0:
                continue
            chunk = chunks[int(chunk_id)]
            docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": chunk["source"],
                        "path": chunk["path"],
                        "score": float(score),
                    },
                )
            )
        return docs

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> AskResponse:
        logger.info("Answering question user_id=%s chat_id=%s", user_id, chat_id)
        docs = self.retrieve(question, top_k)
        answer = self.llm_service.generate_answer(question, docs)
        sources = [
            SourceChunk(
                source=doc.metadata["source"],
                score=round(float(doc.metadata["score"]), 4),
                text=doc.page_content,
            )
            for doc in docs
        ]
        return AskResponse(answer=answer, sources=sources)

