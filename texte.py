import re
from pathlib import Path

import pdfplumber

INPUT_FOLDER = Path("data_mines_mali/raw")
OUTPUT_FOLDER = Path("data_mines_mali/clean")

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\wÀ-ÿ.,;:!?()%°/\-'\n ]", "", text)
    return text.strip()

def extract_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_all():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Le dossier {INPUT_FOLDER} n'existe pas. Cree-le et ajoute tes PDF dedans."
        )

    for path in INPUT_FOLDER.glob("*.pdf"):
        output_file = OUTPUT_FOLDER / f"{path.stem}.txt"
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"Deja extrait: {path.name}")
            continue

        print(f"Extraction de {path.name}...")
        raw_text = extract_pdf(path)
        clean = clean_text(raw_text)
        output_file.write_text(clean, encoding="utf-8")

if __name__ == "__main__":
    process_all()
