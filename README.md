# RAG Application with Azure services and Docling

This repository contains a Retrieval-Augmented Generation (RAG) application that uses Azure OpenAI for embeddings and chat completions, and Azure Cognitive Search for vector search capabilities.

## Features

- 🚀 FastAPI backend for request processing
- 📝 Support for document upload and indexing
- 🔍 Vector search using Azure Cognitive Search
- 🤖 Response generation with Azure OpenAI
- 📊 Multi-agent approach to query processing

## Requirements

- Python 3.8+
- Azure OpenAI account
# RAG — Retrieval-Augmented Generation (Azure OpenAI + Azure Cognitive Search)

This repository contains a small RAG demo: a FastAPI backend that indexes documents with Azure Cognitive Search (vector search) and answers user questions using Azure OpenAI chat + embeddings. A Streamlit frontend provides a simple UI to upload documents and ask questions.

This README is focused on running the project locally on Windows (cmd) and includes exact commands and examples.

## What is included

- `app/` — FastAPI backend and `rag_engine.py` (engine that handles indexing, embedding, retrieval and answering)
- `frontend/` — Streamlit app (`app.py`) and small config
- `test.py` — small example client to query the `/query` endpoint
- `README.md` — this file

## Requirements

- Python 3.8+
- Azure OpenAI access (endpoint + API key)
- Azure Cognitive Search (endpoint + API key)
- Recommended: create a Python venv

Optional (if you don't have Azure): you can adapt the engine to use another embedding/search backend, but the provided code expects Azure services.

## Environment variables (.env)

Create a `.env` file in the repository root (same level as `app/` and `frontend/`) with the following values:

```
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o                # or your chat deployment name
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_key
AZURE_SEARCH_INDEX=rag-index                        # name used for the index
EMBEDING_DIM=1536                                   # embedding dimension used by model
```

Make sure the values match your Azure deployments and index configuration.

## Install dependencies

If `requirements.txt` exists, install it. From repository root on Windows (cmd):

```cmd
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the main packages used in the project:

```cmd
pip install fastapi uvicorn python-dotenv python-multipart requests streamlit rich docling azure-search-documents azure-identity openai langgraph
```

Note: `docling` and `langgraph` may require additional native dependencies; check their docs if installation fails.

## Run backend (FastAPI)

Start the backend from repository root (cmd):

```cmd
cd app
venv\\Scripts\\activate      # if not already activated
uvicorn main:app --reload --port 8000
```

The backend endpoints:

- POST /initialize — create or recreate the Azure Cognitive Search index
- POST /upload — upload and index files (multipart form, field name `files`)
- POST /query — send a JSON body {"question": "..."} and get a generated answer + debug info

Quick curl example (Windows cmd):

```cmd
curl -X POST http://localhost:8000/initialize

curl -X POST -F "files=@C:\\path\\to\\doc1.pdf" -F "files=@C:\\path\\to\\doc2.txt" http://localhost:8000/upload

curl -X POST -H "Content-Type: application/json" -d "{\\"question\\": \\"What is attention mechanism?\\"}" http://localhost:8000/query
```

Or use the provided `test.py` (from repo root):

```cmd
venv\\Scripts\\activate
python test.py
```

## Run frontend (Streamlit)

From repository root:

```cmd
venv\\Scripts\\activate
streamlit run frontend\\app.py
```

Streamlit UI will open in your browser (default: http://localhost:8501). The UI provides:

- Initialize Index button (calls `/initialize`)
- Document upload widget (sends files to `/upload`)
- Chat: enter questions and receive answers; processing details are shown under each answer

Notes:

- Uploading files creates a temporary folder `temp_uploads` on the server which is removed after indexing.
- Logs are written to `rag.txt` (in the working directory) by default.

## Example flow

1. Start backend (uvicorn).
2. Start frontend (streamlit).
3. In the frontend sidebar, press `Initialize Index`.
4. Upload 1..N files via the Document Upload widget — click `Upload Selected Files`.
5. Ask a question in the main UI.

The UI will show the generated answer and a `Processing Details` expander with `rewrites` and `ranked` debug info returned from the engine.

## Troubleshooting

- Missing environment variables: the backend will raise a KeyError. Ensure `.env` is present and correct.
- `docling` conversion errors: some file types or encodings may fail to convert — check `rag.txt` for details.
- Azure authentication errors: verify keys/endpoints and that the Azure resources exist and deployments are correct.
- If indexing fails due to schema or vectorizer mismatches, check `EMBEDING_DIM` and vectorizer/deployment names.

## Security & Production

- Do NOT use `allow_origins=["*"]` in production. Limit origins to your frontend URL.
- Store secrets in a secure vault or environment (don't commit `.env` to git).
- Add authentication and rate limiting for public deployments.

## Development notes

- `rag_engine.py` is the place to adapt behaviour (e.g., switch embedding/source or change chunking strategy).
- `frontend/app.py` is a minimal Streamlit client; you can replace it with a React/Next.js app if you prefer a more advanced UI.

## License

MIT

---

If you want, I can:

- Add a `requirements.txt` pinned file for Windows
- Add a small Dockerfile to run backend+frontend
- Add automated tests for `/query` and `/upload`

Tell me which of these you want next.
