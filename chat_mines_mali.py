import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_index import INDEX_FILE, tokenize
from build_vector_index import VECTOR_INDEX_FILE, VECTOR_MODEL_NAME

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

TOP_K = 5
ENV_FILE = Path(".env")

QUESTION_EXPANSIONS = {
    "creer": "exploitation permis demande attribution recherche faisabilite petite mine grande mine code minier decret",
    "créer": "exploitation permis demande attribution recherche faisabilite petite mine grande mine code minier decret",
    "ouvrir": "exploitation permis demande attribution recherche faisabilite petite mine grande mine code minier decret",
    "exploiter": "exploitation permis demande attribution recherche faisabilite petite mine grande mine code minier decret",
    "permis": "code minier decret attribution demande titre minier administration mines faisabilite environnemental",
    "petite mine": "permis exploitation petite mine rapport faisabilite permis environnemental rehabilitation developpement communautaire",
    "contribution": "revenus secteur extractif budget etat recettes total pourcentage milliards fcfa itie",
    "budget": "revenus secteur extractif budget etat recettes total pourcentage milliards fcfa itie",
    "environnement": "code minier permis environnemental etude impact environnemental social rehabilitation fermeture",
    "mines d or": "societes mines or production exploitation permis gold mining yatela loulo gounkoto morila fekola",
    "or": "societes mines or production exploitation permis gold mining yatela loulo gounkoto morila fekola",
}

def load_env_file():
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index introuvable: {INDEX_FILE}. Lance: python texte.py, python chunks.py, puis python build_index.py"
        )
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def load_vector_index():
    if not VECTOR_INDEX_FILE.exists():
        return None

    data = np.load(VECTOR_INDEX_FILE, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "sources": data["sources"],
        "texts": data["texts"],
        "model_name": str(data["model_name"].item()),
    }

@lru_cache(maxsize=1)
def get_vector_model(model_name):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)

def expand_question(question):
    extra_terms = []
    lowered = question.lower()
    for keyword, expansion in QUESTION_EXPANSIONS.items():
        if keyword in lowered:
            extra_terms.append(expansion)
    return " ".join([question, *extra_terms])

def search_keywords(question, index, top_k=TOP_K):
    query_tokens = tokenize(expand_question(question))
    if not query_tokens:
        return []

    idf = index["idf"]
    results = []

    for document in index["documents"]:
        score = 0.0
        for token in query_tokens:
            if token in document["tokens"]:
                tf = document["tokens"][token] / max(document["length"], 1)
                score += tf * idf.get(token, 1.0)

        if score > 0:
            document = dict(document)
            document["score"] = score
            results.append(document)

    return sorted(results, key=lambda document: document["score"], reverse=True)[:top_k]

def search_vector(question, vector_index, top_k=TOP_K):
    try:
        model_name = vector_index.get("model_name") or VECTOR_MODEL_NAME
        model = get_vector_model(model_name)
    except ImportError:
        return []

    query_embedding = model.encode([expand_question(question)], normalize_embeddings=True)[0]
    scores = vector_index["embeddings"] @ query_embedding
    best_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "source": str(vector_index["sources"][index]),
            "text": str(vector_index["texts"][index]),
            "score": float(scores[index]),
        }
        for index in best_indices
        if scores[index] > 0
    ]

def rerank_documents(question, documents):
    lowered = question.lower()

    def score(document):
        source = document["source"].lower()
        text = document["text"].lower()
        value = float(document.get("score", 0))

        if "permis" in lowered or "autorisation" in lowered or "titre" in lowered:
            if "code_minier" in source or "decret_application_code_minier" in source:
                value += 2.0
            if "itie_mali_rapport" in source:
                value -= 0.8

        if "petite mine" in lowered:
            if "petite mine" in text:
                value += 1.0
            if "article 71" in text or "article 69" in text or "article 95" in text:
                value += 1.0

        asks_steps_or_docs = any(term in lowered for term in ("etape", "étape", "document", "dossier"))
        if asks_steps_or_docs:
            for term in (
                "rapport de faisabil",
                "permis environnemental",
                "plan de fermeture",
                "administration chargee des mines",
            ):
                if term in text:
                    value += 0.6

        return value

    return sorted(documents, key=score, reverse=True)

def hybrid_search(question, index, top_k=TOP_K):
    vector_index = load_vector_index()
    vector_results = search_vector(question, vector_index, top_k=top_k * 3) if vector_index else []
    keyword_results = search_keywords(question, index, top_k=top_k * 3)

    merged = {}
    for document in vector_results + keyword_results:
        existing = merged.get(document["source"])
        if not existing or document.get("score", 0) > existing.get("score", 0):
            merged[document["source"]] = document

    return rerank_documents(question, list(merged.values()))[:top_k]

def compact_context(documents):
    context_parts = []
    for i, document in enumerate(documents, start=1):
        text = re.sub(r"\s+", " ", document["text"]).strip()
        context_parts.append(f"[Source {i}: {document['source']}]\n{text}")
    return "\n\n".join(context_parts)

def answer_with_openai(question, documents):
    try:
        from openai import OpenAI
    except ImportError:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        client = OpenAI()
        context = compact_context(documents)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant specialise sur les mines au Mali. "
                        "Reponds uniquement avec le contexte fourni. "
                        "Si l'information n'est pas dans les sources, dis clairement que tu ne sais pas. "
                        "Reponds en francais et cite les sources utiles."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContexte:\n{context}",
                },
            ],
        )
        return response.output_text
    except Exception as error:
        print(f"API OpenAI indisponible ({error.__class__.__name__}). Passage en mode local.")
        return None

def clean_excerpt(text, max_chars=450):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def source_label(document):
    return document["source"].replace("_chunk_", " passage ").replace(".txt", "")

def answer_mining_creation(question, documents):
    lowered = question.lower()
    wants_creation = any(word in lowered for word in ("creer", "créer", "ouvrir", "exploiter"))
    mentions_mine = "mine" in lowered or "minier" in lowered
    if not (wants_creation and mentions_mine):
        return None

    sources = "\n".join(f"- {source_label(document)}" for document in documents[:5])
    return (
        "Pour creer ou exploiter une mine au Mali, les passages retrouves indiquent surtout "
        "qu'il faut passer par un titre minier, pas simplement commencer les travaux.\n\n"
        "Etapes principales:\n"
        "1. Identifier le type de projet: petite mine ou grande mine. Le Code minier distingue "
        "le permis d'exploitation de petite mine et le permis d'exploitation de grande mine.\n"
        "2. Disposer d'une personne morale de droit malien lorsque le texte l'exige, notamment "
        "pour le permis d'exploitation de petite mine.\n"
        "3. Justifier les capacites techniques et financieres du demandeur.\n"
        "4. Preparer un dossier de demande avec un rapport de faisabilite montrant les reserves, "
        "la faisabilite technique et la faisabilite economique de l'exploitation.\n"
        "5. Deposer la demande aupres de l'administration chargee des mines.\n"
        "6. Obtenir le permis ou l'autorisation officielle avant l'exploitation. Pour une petite mine, "
        "les sources parlent d'un arrete interministeriel des ministres charges des Mines, de l'Economie "
        "et des Finances.\n"
        "7. Respecter les obligations fiscales, sociales et environnementales, qui peuvent etre verifiees "
        "dans le dossier et lors du renouvellement.\n\n"
        "Donc, en pratique, il faut d'abord securiser le droit minier, monter un dossier technique et "
        "financier solide, puis obtenir le permis d'exploitation adapte au projet.\n\n"
        f"Sources utilisees:\n{sources}"
    )

def answer_permit_steps(question, documents):
    lowered = question.lower()
    asks_documents = any(word in lowered for word in ("document", "dossier", "piece", "pièce"))
    asks_steps = any(word in lowered for word in ("etape", "étape", "procedure", "procédure", "obtenir"))
    asks_permit = "permis" in lowered or "autorisation" in lowered
    asks_small_mine = "petite mine" in lowered or "petite" in lowered
    if not ((asks_documents or asks_steps) and asks_permit and asks_small_mine):
        return None

    sources = "\n".join(f"- {source_label(document)}" for document in documents[:5])
    return (
        "Pour obtenir un permis d'exploitation de petite mine au Mali, les sources juridiques indiquent "
        "qu'il faut passer par une demande formelle aupres de l'administration chargee des mines.\n\n"
        "Etapes principales:\n"
        "1. Etre une personne morale de droit malien et, selon le Code minier, etre titulaire d'un permis "
        "de recherche dans le perimetre concerne.\n"
        "2. Verifier que le projet correspond bien a une petite mine, et non a une grande mine.\n"
        "3. Preparer un rapport de faisabilite montrant l'existence des reserves, la faisabilite technique "
        "et economique de l'exploitation, ainsi que la commercialisation des produits.\n"
        "4. Obtenir un permis environnemental fonde sur une etude d'impact environnemental et social.\n"
        "5. Preparer un plan de fermeture et de rehabilitation de la mine.\n"
        "6. Preparer un plan de developpement communautaire.\n"
        "7. Justifier les capacites techniques et financieres du demandeur.\n"
        "8. Deposer le dossier aupres de l'administration chargee des mines.\n"
        "9. Attendre l'attribution officielle du permis. Pour la petite mine, les sources parlent d'un "
        "arrete interministeriel des ministres charges des Mines, de l'Economie et des Finances.\n\n"
        "A retenir: le dossier doit prouver que le gisement existe, que le projet est exploitable, "
        "que le demandeur peut financer et conduire les travaux, et que les obligations "
        "environnementales et sociales sont prises en compte.\n\n"
        f"Sources utilisees:\n{sources}"
    )

def document_text(documents):
    return "\n".join(document["text"] for document in documents)

def extract_percentages(text, limit=5):
    return re.findall(r"\d+(?:,\d+)?\s?%", text)[:limit]

def extract_amounts(text, limit=5):
    patterns = [
        r"\d+(?:,\d+)?\s*(?:milliards?|millions?)\s*(?:de\s*)?(?:FCFA|francs CFA)",
        r"\d+(?:,\d+)?\s*(?:FCFA|F CFA)",
    ]
    values = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return values[:limit]

def extract_company_names(text, limit=12):
    candidates = re.findall(r"\b[A-Z][A-Z0-9&' -]{3,}(?:S\.?A\.?|SARL|SASU|LTD|MALI|GOLD|MINING)?\b", text)
    cleaned = []
    banned = {"MALI", "SOURCE", "TABLEAU", "FIGURE", "ARTICLE", "MINISTERE", "REPUBLIQUE"}
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        if len(candidate) < 5 or candidate in banned:
            continue
        if any(word in candidate for word in ("GOLD", "MINING", "SOMILO", "SEMOS", "MORILA", "LOULO", "GOUNKOTO", "FEKOLA", "YATELA")):
            if candidate not in cleaned:
                cleaned.append(candidate)
        if len(cleaned) >= limit:
            break
    return cleaned

def classify_question(question):
    lowered = question.lower()
    if any(term in lowered for term in ("rôle du code", "role du code", "code minier sert", "explique le code minier")):
        return "code_role"
    if any(term in lowered for term in ("permis", "autorisation", "titre minier", "petite mine", "exploiter une mine", "creer une mine", "créer une mine")):
        return "procedure"
    if any(term in lowered for term in ("budget", "contribution", "recette", "revenu", "fiscal", "impot", "impôt", "taxe")):
        return "statistics"
    if any(term in lowered for term in ("environnement", "impact", "rehabilitation", "réhabilitation", "fermeture", "pollution")):
        return "environment"
    if any(term in lowered for term in ("mine d'or", "mines d'or", "mines d or", "mine d or", "societe miniere", "société minière", "compagnie", "entreprise", "exploitant")):
        return "operators"
    return "general"

def answer_statistics(question, documents):
    if classify_question(question) != "statistics":
        return None
    text = document_text(documents)
    percentages = extract_percentages(text)
    amounts = extract_amounts(text)

    details = []
    if percentages:
        details.append(f"Les passages retrouves mentionnent notamment les taux suivants: {', '.join(percentages)}.")
    if amounts:
        details.append(f"Ils mentionnent aussi des montants comme: {', '.join(amounts)}.")

    detail_text = "\n".join(f"- {item}" for item in details) if details else "- Les documents retrouves parlent de revenus, paiements et recettes du secteur extractif, mais les chiffres exacts doivent etre verifies dans le rapport ITIE correspondant."
    return (
        "Le secteur extractif contribue au budget malien principalement a travers les paiements des societes minieres, "
        "les recettes fiscales, les droits, taxes et autres revenus publics lies a l'exploitation.\n\n"
        "Elements chiffres retrouves:\n"
        f"{detail_text}\n\n"
        "A retenir: pour une annee precise, il faut toujours distinguer les revenus totaux du secteur extractif, "
        "la part affectee au budget de l'Etat, et les autres affectations comme les collectivites ou fonds specifiques."
    )

def answer_environment(question, documents):
    if classify_question(question) != "environment":
        return None
    return (
        "Sur l'environnement, les documents indiquent que l'exploitation miniere ne doit pas etre traitee uniquement "
        "comme un projet technique: elle doit aussi integrer les impacts environnementaux et sociaux.\n\n"
        "Points essentiels:\n"
        "1. Le dossier minier doit comprendre une evaluation ou etude d'impact environnemental et social lorsque le texte l'exige.\n"
        "2. Le demandeur doit prevoir un permis ou une validation environnementale avant l'exploitation.\n"
        "3. Le projet doit inclure un plan de fermeture et de rehabilitation de la mine.\n"
        "4. Les obligations environnementales peuvent etre controlees pendant la vie du titre minier et lors du renouvellement.\n\n"
        "En pratique, une mine doit donc prouver qu'elle peut exploiter le gisement tout en gerant les impacts, "
        "la rehabilitation du site et les obligations sociales associees."
    )

def answer_operators(question, documents):
    if classify_question(question) != "operators":
        return None
    text = document_text(documents)
    known = []
    for name in ("SOCIÉTÉ DES MINES DE SYAMA", "SOMISY", "SEMOS", "SOCIÉTÉ D'EXPLOITATION DES MINES D'OR DE SADIOLA", "MORILA", "LOULO", "GOUNKOTO", "FEKOLA", "YATELA"):
        if name.lower() in text.lower() and name not in known:
            known.append(name)
    names = known or extract_company_names(text)
    if names:
        listed = "\n".join(f"- {name.title()}" for name in names[:10])
        return (
            "Les documents retrouves mentionnent plusieurs mines, projets ou societes lies a l'or au Mali.\n\n"
            f"Exemples identifies:\n{listed}\n\n"
            "Attention: selon les rapports, une liste peut melanger mines en exploitation, societes titulaires, projets, "
            "ou entites retenues dans le perimetre ITIE. Pour une liste officielle et a jour, il faut verifier le cadastre minier le plus recent."
        )
    return (
        "Les documents indiquent que le secteur aurifere malien comprend plusieurs societes et titres miniers, "
        "mais les passages retrouves ne permettent pas d'etablir une liste propre et definitive.\n\n"
        "Pour une reponse fiable, il faut preciser si tu veux: les mines industrielles en production, les societes titulaires, "
        "les permis de recherche, ou les projets auriferes."
    )

def answer_code_role(question, documents):
    if classify_question(question) != "code_role":
        return None
    return (
        "Le Code minier est le texte central qui organise l'activite miniere au Mali.\n\n"
        "Son role principal est de:\n"
        "1. definir les titres miniers et les autorisations necessaires pour rechercher ou exploiter une substance minerale;\n"
        "2. fixer les conditions d'attribution, de renouvellement, de cession ou de retrait de ces titres;\n"
        "3. encadrer les droits et obligations des titulaires, notamment les obligations techniques, fiscales, sociales et environnementales;\n"
        "4. distinguer les regimes applicables selon le type d'activite: recherche, petite mine, grande mine, carriere ou autres substances;\n"
        "5. donner a l'administration miniere une base juridique pour controler et suivre les activites du secteur.\n\n"
        "En clair, le Code minier sert a transformer une activite miniere en activite autorisee, encadree et controlee par l'Etat."
    )

def answer_general(question, documents):
    if not documents:
        return None
    text = clean_excerpt(documents[0]["text"], max_chars=700)
    return (
        "Voici ce que la base documentaire permet de dire de facon prudente:\n\n"
        f"{text}\n\n"
        "Si tu veux, reformule la question avec une annee, un type de permis, une societe ou un theme precis "
        "pour obtenir une reponse plus nette."
    )

def answer_without_llm(question, documents):
    if not documents:
        return (
            "Je n'ai pas trouve de passage pertinent dans les documents indexes. "
            "Ajoute plus de PDF dans data_mines_mali/raw ou reformule la question."
        )

    guided_answer = answer_mining_creation(question, documents)
    if guided_answer:
        return guided_answer

    guided_answer = answer_permit_steps(question, documents)
    if guided_answer:
        return guided_answer

    guided_answer = answer_statistics(question, documents)
    if guided_answer:
        return guided_answer

    guided_answer = answer_code_role(question, documents)
    if guided_answer:
        return guided_answer

    guided_answer = answer_environment(question, documents)
    if guided_answer:
        return guided_answer

    guided_answer = answer_operators(question, documents)
    if guided_answer:
        return guided_answer

    return answer_general(question, documents)

def ask(question):
    load_env_file()
    index = load_index()
    documents = hybrid_search(question, index)
    answer = answer_with_openai(question, documents)
    return answer or answer_without_llm(question, documents)

def ask_details(question):
    load_env_file()
    index = load_index()
    documents = hybrid_search(question, index)
    answer = answer_with_openai(question, documents)
    engine = "openai" if answer else "local"
    answer = answer or answer_without_llm(question, documents)

    sources = [
        {
            "label": source_label(document),
            "source": document["source"],
            "excerpt": clean_excerpt(document["text"], max_chars=320),
            "score": round(float(document.get("score", 0)), 4),
        }
        for document in documents
    ]

    confidence = "faible"
    if len(sources) >= 4:
        confidence = "eleve"
    elif len(sources) >= 2:
        confidence = "moyen"

    return {
        "answer": answer,
        "sources": sources,
        "engine": engine,
        "confidence": confidence,
    }

def main():
    print("Assistant Mines Mali. Tape 'exit' pour quitter.")
    if VECTOR_INDEX_FILE.exists():
        print("Mode recherche: vectoriel + mots-cles")
    else:
        print("Mode recherche: mots-cles")

    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        print("\n" + ask(question))

if __name__ == "__main__":
    main()
