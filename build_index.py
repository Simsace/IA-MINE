import json
import math
import re
from collections import Counter
from pathlib import Path

CHUNKS_FOLDER = Path("data_mines_mali/chunks")
INDEX_FILE = Path("data_mines_mali/index.json")

TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "en",
    "et", "la", "le", "les", "leur", "leurs", "l", "d", "un", "une", "pour",
    "par", "sur", "que", "qui", "se", "sont", "est", "mali", "malien",
}

def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]

def build_index():
    if not CHUNKS_FOLDER.exists():
        raise FileNotFoundError(
            f"Le dossier {CHUNKS_FOLDER} n'existe pas. Lance d'abord: python chunks.py"
        )

    documents = []
    document_frequency = Counter()

    for path in sorted(CHUNKS_FOLDER.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        tokens = tokenize(text)
        token_counts = Counter(tokens)
        document_frequency.update(token_counts.keys())
        documents.append(
            {
                "source": path.name,
                "text": text,
                "tokens": dict(token_counts),
                "length": len(tokens),
            }
        )

    total_documents = len(documents)
    idf = {
        token: math.log((1 + total_documents) / (1 + freq)) + 1
        for token, freq in document_frequency.items()
    }

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(
        json.dumps({"documents": documents, "idf": idf}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Index cree: {INDEX_FILE} ({total_documents} passages)")

if __name__ == "__main__":
    build_index()
