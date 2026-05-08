# IA sur les mines au Mali

Ce projet permet de construire une IA documentaire capable de repondre aux questions sur les mines au Mali a partir de tes propres PDF.

## Installation

```powershell
pip install -r requirements.txt
```

## Ajouter les documents

Place tes PDF dans:

```text
data_mines_mali/raw
```

Exemples de documents utiles: code minier, rapports ITIE, rapports ministeriels, etudes geologiques, communiques officiels, rapports de societes minieres.

## Construire la base de connaissance

```powershell
python texte.py
python chunks.py
python build_index.py
```

Pour une recherche plus performante, construis aussi l'index vectoriel:

```powershell
python build_vector_index.py
```

## Poser des questions

```powershell
python chat_mines_mali.py
```

Sans cle API, le script affiche les passages les plus pertinents. Avec une cle OpenAI, il redige une reponse en francais en citant les passages retrouves.

Le plus simple est de creer un fichier `.env` a partir de `.env.example`:

```text
OPENAI_API_KEY=ta_nouvelle_cle_api
OPENAI_MODEL=gpt-4.1-mini
```

Le fichier `.env` est ignore par Git pour eviter de publier la cle.

```powershell
python chat_mines_mali.py
```

## Principe

1. `texte.py` extrait et nettoie le texte des PDF.
2. `chunks.py` decoupe les textes en passages courts.
3. `build_index.py` cree un index de recherche local.
4. `build_vector_index.py` cree un index semantique multilingue plus performant.
5. `chat_mines_mali.py` retrouve les passages utiles et genere la reponse.
