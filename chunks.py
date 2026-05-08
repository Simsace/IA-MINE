import re
from pathlib import Path

INPUT_FOLDER = Path("data_mines_mali/clean")
OUTPUT_FOLDER = Path("data_mines_mali/chunks")

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]

def chunk_text(text, max_words=180, overlap_sentences=1):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        next_chunk = current_chunk + [sentence]
        if current_chunk and len(" ".join(next_chunk).split()) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []

        current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_chunks():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Le dossier {INPUT_FOLDER} n'existe pas. Lance d'abord: python texte.py"
        )

    for path in INPUT_FOLDER.glob("*.txt"):
        print(f"Decoupage de {path.name}...")
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks, start=1):
            chunk_file = f"{path.stem}_chunk_{i:04d}.txt"
            (OUTPUT_FOLDER / chunk_file).write_text(chunk, encoding="utf-8")

if __name__ == "__main__":
    process_chunks()
