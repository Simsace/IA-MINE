import json
import os
from pathlib import Path

import faiss
import numpy as np

CHUNKS_DIR = Path("data_mines_mali/chunks")
INDEX_PATH = Path("mines_index.faiss")
METADATA_PATH = Path("metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"
HF_CACHE_DIR = Path(".hf-cache")
LOCAL_MODELS_DIR = Path(".models")
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

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.resolve()))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR.resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download


def read_chunks(chunks_dir: Path) -> list[dict]:
    """Read all UTF-8 chunk files and return metadata records."""
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    records = []
    for path in sorted(chunks_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"Could not read {path} as UTF-8: {exc.reason}",
            ) from exc

        if not text:
            continue

        records.append(
            {
                "id": len(records),
                "source": path.name,
                "path": str(path.as_posix()),
                "text": text,
            }
        )

    if not records:
        raise ValueError(f"No non-empty .txt chunks found in {chunks_dir}")

    return records


def get_model_path(model_name: str) -> str:
    """Download the embedding model locally without symlinks for Windows compatibility."""
    local_path = LOCAL_MODELS_DIR / model_name
    has_required_files = all((local_path / file).exists() for file in MODEL_REQUIRED_FILES)
    has_weights = any((local_path / file).exists() for file in MODEL_WEIGHT_FILES)
    if local_path.exists() and has_required_files and has_weights:
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=f"sentence-transformers/{model_name}",
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        allow_patterns=[
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
        ],
    )
    return str(local_path)


def build_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Create normalized sentence embeddings for cosine-similarity search."""
    model = SentenceTransformer(get_model_path(model_name))
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index. Inner product works as cosine similarity after normalization."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D numpy array")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_metadata(records: list[dict], model_name: str, embedding_dim: int) -> None:
    """Save metadata needed to interpret FAISS result IDs."""
    payload = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "chunk_count": len(records),
        "chunks": records,
    }
    METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    try:
        print(f"Reading chunks from: {CHUNKS_DIR}")
        records = read_chunks(CHUNKS_DIR)

        print(f"Generating embeddings with: {MODEL_NAME}")
        texts = [record["text"] for record in records]
        embeddings = build_embeddings(texts, MODEL_NAME)

        print("Building FAISS index...")
        index = build_faiss_index(embeddings)

        print(f"Saving FAISS index to: {INDEX_PATH}")
        faiss.write_index(index, str(INDEX_PATH))

        print(f"Saving metadata to: {METADATA_PATH}")
        save_metadata(records, MODEL_NAME, embeddings.shape[1])

        print(f"Done. Indexed {len(records)} chunks.")
    except Exception as exc:
        print(f"Error while building RAG index: {exc}")
        raise


if __name__ == "__main__":
    main()
