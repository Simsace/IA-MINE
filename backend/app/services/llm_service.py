import logging

from app.core.config import get_settings
from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Tu es un assistant professionnel specialise dans le secteur minier au Mali.
Tu dois repondre uniquement a partir du contexte fourni.
Si le contexte ne contient pas l'information, dis clairement que l'information n'est pas disponible dans la base documentaire.
Reponds en francais, avec un style clair, utile et professionnel.
Ne fabrique pas de chiffres, de lois, de noms de societes ou de procedures.
"""


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
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", f"source-{index}")
            parts.append(f"[Source {index}: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_answer(question: str, docs: list[Document]) -> str:
        if not docs:
            return "Je n'ai pas trouve d'information pertinente dans la base documentaire."

        context_preview = "\n\n".join(
            f"- {doc.page_content[:700].strip()}" for doc in docs[:3]
        )
        return (
            "L'API LLM n'est pas configuree. Voici les passages les plus pertinents "
            "retrouves dans la base documentaire interne:\n\n"
            f"{context_preview}\n\n"
            "Configure OPENAI_API_KEY et MODEL_NAME pour obtenir une reponse redigee par le modele."
        )

    def generate_answer(self, question: str, docs: list[Document]) -> str:
        client = self._client()
        if client is None:
            logger.warning("OPENAI_API_KEY is missing; using fallback answer")
            return self._fallback_answer(question, docs)

        context = self._format_context(docs)
        user_prompt = f"Question utilisateur:\n{question}\n\nContexte documentaire:\n{context}"

        response = client.chat.completions.create(
            model=self.settings.model_name,
            temperature=self.settings.llm_temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

