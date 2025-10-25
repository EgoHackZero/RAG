# RAG Application with Azure OpenAI and Vector Search

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
- Azure Cognitive Search account

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-application.git
cd rag-application
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add the following variables:
```env
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_EMBED_DEPLOYMENT=your_embedding_deployment
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_API_KEY=your_search_api_key
AZURE_SEARCH_INDEX=your_index_name
EMBEDING_DIM=1536
```

## Running the Application

1. Start the backend:
```bash
cd app
uvicorn main:app --reload
```

2. The API will be available at: `http://localhost:8000`

## API Endpoints

### POST /initialize
Initializes the search index in Azure Cognitive Search.

### POST /upload
Uploads and indexes documents. Accepts multiple files.

### POST /query
Processes user query and returns response based on indexed documents.

Example request:
```json
{
    "question": "What is attention mechanism?"
}
```

## Project Structure

```
app/
├── main.py              # FastAPI application
├── rag_engine.py        # Core RAG engine
└── requirements.txt     # Project dependencies
```

## Usage

1. First, initialize the index:
```bash
curl -X POST http://localhost:8000/initialize
```

2. Upload documents:
```bash
curl -X POST -F "files=@document1.pdf" -F "files=@document2.txt" http://localhost:8000/upload
```

3. Send a query:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question":"What is attention mechanism?"}' http://localhost:8000/query
```

## Frontend

For the frontend part, we use [streamlit](https://streamlit.io/) as a simple and effective solution for creating the user interface. Install it:

```bash
pip install streamlit
```

## Security

- Configure CORS policy in `main.py` for production
- Use secure secret storage
- Add authentication for API endpoints
- Set up rate limiting

## License

MIT

## Support

If you encounter any issues, please create an issue in the project repository.