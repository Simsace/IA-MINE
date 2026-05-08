# Mines Mali RAG Backend

Production-ready FastAPI backend for an AI assistant specialized in mining information about Mali.

## Features

- FastAPI API with Swagger documentation
- FAISS vector database
- SentenceTransformers embeddings
- LangChain `Document` retrieval pipeline
- OpenAI-compatible LLM API architecture
- CORS support for Flutter mobile, Flutter web, and future websites
- UTF-8 chunk handling
- Modular architecture
- Future-ready authentication hook
- Future-ready chat history fields

## Structure

```text
backend/
  app/
    api/
    core/
    models/
    rag/
    services/
    utils/
  data/
  build_rag.py
  main.py
  requirements.txt
  .env.example
  README.md
```

## Setup

From the project root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-4o-mini
```

The backend uses an OpenAI-compatible API. You can also set:

```text
OPENAI_BASE_URL=https://your-compatible-provider/v1
```

## Build the FAISS RAG index

The chunks are read from:

```text
../data_mines_mali/chunks
```

Build:

```powershell
python build_rag.py
```

This creates:

```text
backend/data/mines_index.faiss
backend/data/metadata.json
```

## Run locally

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```text
GET http://localhost:8000/health
```

Swagger:

```text
http://localhost:8000/docs
```

## Ask endpoint

```http
POST /ask
Content-Type: application/json
```

Request:

```json
{
  "question": "Quels sont les types d'exploitation miniere au Mali ?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "Mali_Code_Minier_2023_Loi_2023_040_chunk_0082.txt",
      "score": 0.72,
      "text": "..."
    }
  ]
}
```

## Flutter API call

Example with Dart `http`:

```dart
final response = await http.post(
  Uri.parse('https://your-backend-url.com/ask'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'question': 'Quels sont les types d’exploitation minière au Mali ?',
  }),
);

final data = jsonDecode(response.body);
print(data['answer']);
```

For local Android emulator:

```text
http://10.0.2.2:8000/ask
```

For a physical phone on the same Wi-Fi, use your PC IP:

```text
http://192.168.x.x:8000/ask
```

## Render deployment

1. Push the repository to GitHub.
2. Create a new Render Web Service.
3. Root directory: `backend`
4. Build command:

```bash
pip install -r requirements.txt && python build_rag.py
```

5. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

6. Add environment variables:

```text
OPENAI_API_KEY
MODEL_NAME
OPENAI_BASE_URL
```

## Railway deployment

1. Create a Railway project from GitHub.
2. Set root directory to `backend`.
3. Add variables:

```text
OPENAI_API_KEY
MODEL_NAME
OPENAI_BASE_URL
```

4. Build command:

```bash
pip install -r requirements.txt && python build_rag.py
```

5. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Production notes

- Restrict `CORS_ORIGINS` to your real Flutter web/public website domains.
- Replace `optional_auth_dependency` with JWT/API key/Firebase/Supabase validation.
- Store chat history in PostgreSQL, Supabase, Firebase, or another managed database.
- Keep FAISS and metadata persistent in cloud storage or rebuild during deployment.
- Monitor latency and token usage.
- Use a multilingual embedding model for better French retrieval if needed.

