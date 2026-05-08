import json
import logging

import faiss
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class FaissStore:
    """Lazy FAISS index and metadata loader."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: dict | None = None

    def load_index(self) -> faiss.Index:
        settings = get_settings()
        if self._index is None:
            if not settings.faiss_index_path.exists():
                raise FileNotFoundError(settings.faiss_index_path)
            logger.info("Loading FAISS index from %s", settings.faiss_index_path)
            self._index = faiss.read_index(str(settings.faiss_index_path))
        return self._index

    def load_metadata(self) -> dict:
        settings = get_settings()
        if self._metadata is None:
            if not settings.metadata_path.exists():
                raise FileNotFoundError(settings.metadata_path)
            logger.info("Loading metadata from %s", settings.metadata_path)
            self._metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        return self._metadata

    def reload(self) -> None:
        self._index = None
        self._metadata = None

