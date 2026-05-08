import logging
import json
from typing import List

from app.core.config import get_settings
from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Vous êtes un assistant professionnel spécialisé dans le secteur minier au Mali."
    " Répondez uniquement à partir du contexte fourni. Si l'information n'est pas dans le"
    " contexte, répondez clairement que l'information n'est pas disponible. Répondez en"
    " français, de façon concise et professionnelle. Ne fabriquez pas de chiffres, noms"
    " ou procédures."
)


class LlmService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _client(self) -> OpenAI | None:
        if not self.settings.openai_api_key:
            return None
        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.llm_timeout_seconds,
        )

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", f"source-{index}")
            parts.append(f"[{source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _fallback_result(self, question: str, docs: List[Document]) -> dict:
        # structured fallback when LLM client is not configured
        if not docs:
            return {
                "answer": "Aucune information disponible dans la base documentaire.",
                "sources": [],
                "confidence": 0.0,
                "engine": self.settings.engine_name,
            }

        sources = [doc.metadata.get("source") for doc in docs]
        preview = "\n\n".join(f"- {doc.page_content[:400].strip()}" for doc in docs[:3])
        return {
            "answer": (
                "LLM non configuré. Extraits pertinents:\n" + preview
            ),
            "sources": sources,
            "confidence": 0.0,
            "engine": self.settings.engine_name,
        }

    def generate_answer(self, question: str, docs: List[Document], *,
                        temperature: float | None = None, max_tokens: int | None = None) -> dict:
        """Generate a structured answer dict: {answer, sources, confidence, engine}.

        The LLM is asked to reply strictly from provided docs and to output JSON.
        """
        client = self._client()
        if client is None:
            logger.warning("OPENAI_API_KEY is missing; returning fallback result")
            return self._fallback_result(question, docs)

        temperature = temperature if temperature is not None else self.settings.llm_temperature
        max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_tokens

        context = self._format_context(docs)

        user_instructions = (
            "Fournis une réponse concise et professionnelle en français basée uniquement sur le contexte."
            " Ne pas inventer d'informations. Retourne strictement un JSON avec les clés: answer, sources, confidence."
            " Exemple: {\"answer\": \"...\", \"sources\":[\"file1\"], \"confidence\":0.8}"
        )

        user_prompt = f"Question: {question}\n\nContexte:\n{context}\n\n{user_instructions}"

        try:
            response = client.chat.completions.create(
                model=self.settings.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.exception("LLM call failed")
            return self._fallback_result(question, docs)

        raw = (response.choices[0].message.content or "").strip()

        # Try to extract JSON from the model output
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer") or parsed.get("text") or ""
            sources = parsed.get("sources") or [doc.metadata.get("source") for doc in docs]
            confidence = float(parsed.get("confidence", 0.0))
            return {"answer": answer, "sources": sources, "confidence": confidence, "engine": self.settings.engine_name}
        except Exception:
            # Fallback: return raw text and infer sources
            used = [doc.metadata.get("source") for doc in docs if doc.metadata.get("source") in raw]
            return {"answer": raw, "sources": used or [doc.metadata.get("source") for doc in docs], "confidence": 0.5, "engine": self.settings.engine_name}

