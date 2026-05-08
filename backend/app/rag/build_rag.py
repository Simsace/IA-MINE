import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from app.core.config import get_settings
from app.utils.text import clean_whitespace
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_REQUIRED_FILES = (
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
)
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
MODEL_ALLOW_PATTERNS = [
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "pytorch_model.bin",
    "1_Pooling/config.json",
]


def configure_huggingface_cache() -> None:
    settings = get_settings()
    os.environ.setdefault("HF_HOME", str(settings.hf_cache_dir.resolve()))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(settings.hf_cache_dir.resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def get_model_path(model_name: str) -> str:
    """Download model files locally without relying on OS symlinks."""
    settings = get_settings()
    local_path = settings.local_models_dir / model_name
    has_required = all((local_path / file).exists() for file in MODEL_REQUIRED_FILES)
    has_weights = any((local_path / file).exists() for file in MODEL_WEIGHT_FILES)
    if local_path.exists() and has_required and has_weights:
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=f"sentence-transformers/{model_name}",
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        allow_patterns=MODEL_ALLOW_PATTERNS,
    )
    return str(local_path)


def load_chunks(chunks_dir: Path) -> list[dict]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunks = []
    for path in sorted(chunks_dir.glob("*.txt")):
        text = clean_whitespace(path.read_text(encoding="utf-8"))
        if not text:
            continue
        chunks.append(
            {
                "id": len(chunks),
                "source": path.name,
                "path": str(path.as_posix()),
                "text": text,
            }
        )

    if not chunks:
        raise ValueError(f"No non-empty chunks found in {chunks_dir}")
    return chunks


def create_embeddings(chunks: list[dict], model_name: str) -> np.ndarray:
    configure_huggingface_cache()
    model = SentenceTransformer(get_model_path(model_name))
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def write_metadata(chunks: list[dict], embedding_dim: int) -> None:
    settings = get_settings()
    payload = {
        "embedding_model_name": settings.embedding_model_name,
        "embedding_dim": embedding_dim,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_rag_index() -> None:
    settings = get_settings()
    chunks_dir = settings.chunks_dir
    if not chunks_dir.is_absolute():
        chunks_dir = Path(__file__).resolve().parents[3] / chunks_dir
        chunks_dir = chunks_dir.resolve()

    logger.info("Loading chunks from %s", chunks_dir)
    chunks = load_chunks(chunks_dir)
    logger.info("Loaded %s chunks", len(chunks))

    logger.info("Creating embeddings with %s", settings.embedding_model_name)
    embeddings = create_embeddings(chunks, settings.embedding_model_name)

    logger.info("Building FAISS index")
    index = create_faiss_index(embeddings)
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.faiss_index_path))
    write_metadata(chunks, embeddings.shape[1])

    logger.info("Saved FAISS index to %s", settings.faiss_index_path)
    logger.info("Saved metadata to %s", settings.metadata_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_rag_index()

