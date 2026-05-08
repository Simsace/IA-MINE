import json
from pathlib import Path

import numpy as np

from build_index import INDEX_FILE

VECTOR_INDEX_FILE = Path("data_mines_mali/vector_index.npz")
VECTOR_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_documents():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index introuvable: {INDEX_FILE}. Lance d'abord: python build_index.py"
        )

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return index["documents"]

def build_vector_index():
    from sentence_transformers import SentenceTransformer

    documents = load_documents()
    texts = [document["text"] for document in documents]

    print(f"Chargement du modele: {VECTOR_MODEL_NAME}")
    model = SentenceTransformer(VECTOR_MODEL_NAME)

    print(f"Vectorisation de {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sources = np.array([document["source"] for document in documents], dtype=object)
    texts = np.array(texts, dtype=object)

    VECTOR_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VECTOR_INDEX_FILE,
        embeddings=embeddings.astype("float32"),
        sources=sources,
        texts=texts,
        model_name=np.array(VECTOR_MODEL_NAME, dtype=object),
    )
    print(f"Index vectoriel cree: {VECTOR_INDEX_FILE}")

if __name__ == "__main__":
    build_vector_index()
