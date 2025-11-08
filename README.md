# RAG Application with Azure OpenAI + Azure Cognitive Search

A production-ready Retrieval-Augmented Generation (RAG) application that combines Azure OpenAI for embeddings and chat completions with Azure Cognitive Search for vector search capabilities. The system processes documents using Docling, indexes them with vector embeddings, and answers questions using a sophisticated multi-agent LangGraph workflow.

## 🌟 Key Features

### Core Capabilities
- **FastAPI Backend**: High-performance REST API for document processing and querying
- **Document Processing**: Support for PDF, DOCX, TXT, and other formats via Docling
- **Vector Search**: Hybrid search combining keyword and vector similarity using Azure Cognitive Search
- **Multi-Agent Query Workflow**: Sophisticated LangGraph-based pipeline with 5 specialized agents
- **Streamlit Frontend**: User-friendly interface for document upload and chat-based querying

### Advanced Features
- **Multilingual Support**: Detects and responds in ANY language (English, Russian, Spanish, Chinese, Japanese, French, German, and more)
- **Intelligent Query Rewriting**: Generates 3-5 optimized search variants and ranks them by relevance
- **Metadata Fast-Path**: Direct answers for author and citation count queries without full retrieval
- **Accurate Citation Counting**: Handles documents with or without explicit "References" sections
- **Smart Document Chunking**: Hierarchical chunking that preserves document structure
- **Comprehensive Logging**: All activities logged to `rag.txt` with module-level detail

## 📋 Requirements

- **Python**: 3.8 or higher
- **Azure Services**:
  - Azure OpenAI account with chat and embedding deployments
  - Azure Cognitive Search service
- **Environment**: Windows, Linux, or macOS

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and create a virtual environment:

```cmd
git clone <repository-url>
cd RAG
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

### 2. Install Dependencies

```cmd
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```cmd
pip install fastapi uvicorn python-dotenv python-multipart requests streamlit rich docling azure-search-documents azure-identity openai langgraph
```

### 3. Configure Environment Variables

Create a `.env` file in the `RAG/` directory with the following:

```env
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_CHAT_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBED_API_VERSION=2024-12-01-preview
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_key
AZURE_SEARCH_INDEX=rag-index
EMBEDING_DIM=1536
```

**Note**: Replace all placeholder values with your actual Azure resource credentials.

### 4. Run the Backend

Start the FastAPI server:

```cmd
cd RAG\app
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### 5. Run the Frontend

In a new terminal, start the Streamlit UI:

```cmd
cd RAG\frontend
streamlit run app.py
```

The frontend will open automatically in your browser at `http://localhost:8501`

## 📚 Architecture

### System Components

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Streamlit UI   │─────▶│  FastAPI Backend │─────▶│   RAG Engine        │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
                                                              │
                                    ┌─────────────────────────┴─────────────┐
                                    ▼                                       ▼
                          ┌──────────────────┐                  ┌─────────────────┐
                          │  Azure Cognitive │                  │  Azure OpenAI   │
                          │     Search       │                  │   Chat + Embed  │
                          └──────────────────┘                  └─────────────────┘
```

### Core Files

- **`app/main.py`**: FastAPI application with 3 REST endpoints
  - `POST /initialize`: Create/recreate Azure Search index
  - `POST /upload`: Upload and index documents
  - `POST /query`: Process user questions

- **`app/rag_engine.py`**: Core RAG engine with:
  - Document conversion and chunking (Docling)
  - Metadata extraction (title, authors, citation count)
  - Multi-agent query workflow (LangGraph)
  - Multilingual language detection and translation
  - Hybrid vector + keyword search

- **`frontend/app.py`**: Streamlit interface for document upload and querying

### Multi-Agent Query Workflow

The RAG engine uses a **5-node LangGraph workflow**:

```
User Question
     │
     ▼
┌─────────────────────┐
│ 1. Language Detector│  Detects user's language, translates to English for search
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   2. Rewriter       │  Generates 3-5 search query variants with document context
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    3. Ranker        │  Probes search with each variant, scores by relevance
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   4. Retriever      │  Uses best-ranked query to retrieve top-k passages
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   5. Answerer       │  Generates answer in user's original language
└─────────────────────┘
```

**Metadata Fast-Path**: Questions about authors or citation counts bypass the full workflow and answer directly from metadata chunks.

## 🔧 API Usage

### Initialize Index

```bash
curl -X POST http://localhost:8000/initialize
```

**Response**:
```json
{
  "message": "Search index initialized successfully"
}
```

### Upload Documents

```bash
curl -X POST -F "files=@path/to/document.pdf" http://localhost:8000/upload
```

**Response**:
```json
{
  "message": "Successfully processed 1 documents",
  "processed_chunks": 45
}
```

### Query Documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many citations does this paper have?"}'
```

**Response**:
```json
{
  "answer": "This paper has 16 citations.",
  "debug": {
    "rewrites": ["citation count", "number of references", "bibliography size"],
    "ranked": [
      {"q": "citation count", "score": 0.95},
      {"q": "number of references", "score": 0.89}
    ],
    "user_language": "English",
    "search_language": "en",
    "answer_language": "English"
  }
}
```

## 🌍 Multilingual Support

The system automatically:
1. **Detects** the user's language using LLM-based detection
2. **Translates** the question to English for optimal search performance
3. **Retrieves** relevant passages from the indexed documents
4. **Responds** in the user's original language

### Example: Russian Query

**Query**: "Сколько ссылок в документе?"
**Answer**: "В документе 16 ссылок."

The system detected Russian, translated to English for search, and responded in Russian.

## 🧪 Testing

### Test Files

The project includes several test scripts:
- **`test_end_to_end_citations.py`**: Full workflow test for citation queries
- **`test_usage_of_backend.py`**: Standalone RAG engine test (no FastAPI)
- **`test_call_to_app.py`**: HTTP client test for `/query` endpoint
- **`test_api_multilingual.py`**: Tests queries in multiple languages

### Run Tests

```cmd
cd RAG
python test_citation_count.py
python test_end_to_end_citations.py
```

## 📝 Citation Counting

The system uses a **multi-strategy approach** to accurately count citations:

### Strategy 1: Numbered Markers
Detects patterns like:
- `[1]`, `[2]`, `[3]` (bracket notation)
- `- [1]`, `- [2]` (dash + bracket)
- `1.`, `2.`, `3.` (numbered list)
- `1)`, `2)`, `3)` (parenthesis notation)

### Strategy 2: References Section Detection
1. Looks for explicit "References", "Bibliography", or "Literature" headings
2. Falls back to detecting where numbered citations begin (e.g., `- [1]`)
3. Stops at acknowledgments, contributor lists, or other non-reference sections

### Strategy 3: Block/Line Analysis
Counts logical blocks or individual lines that match reference patterns (year, DOI, author names, etc.)

**Result**: Accurate citation counts even for documents without explicit "References" headings.

## 📊 Logging

All system activities are logged to **`rag.txt`** in the working directory:

- **Append Mode**: Logs accumulate across restarts for historical tracking
- **Module-Level Detail**: Shows which component generated each log entry
- **Comprehensive Coverage**: Captures logs from all modules (RAG engine, Azure SDK, FastAPI, etc.)
- **Noise Reduction**: Third-party library logs (Azure, urllib3, httpx) set to WARNING level

### Log Format

```
2025-01-15 10:30:45 [INFO] RAG - ================================================================================
2025-01-15 10:30:45 [INFO] RAG - RAG Engine initialized - New session started
2025-01-15 10:30:45 [INFO] RAG - ================================================================================
2025-01-15 10:30:50 [INFO] RAG - Loaded D:\documents\paper.pdf
2025-01-15 10:30:52 [INFO] RAG - Uploaded batch 0 -> 50; all_succeeded: True
```

## 🔒 Security & Production

### Important Considerations

- **CORS**: The default configuration uses `allow_origins=["*"]`. **Change this** in production to only allow your frontend URL.

  ```python
  # In app/main.py
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://your-frontend-domain.com"],  # Restrict this!
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

- **Environment Variables**: Never commit `.env` files to version control. Use Azure Key Vault or similar secret management in production.

- **Rate Limiting**: Add rate limiting middleware for public deployments to prevent abuse.

- **Authentication**: Implement proper authentication (JWT, OAuth) for production use.

## 🛠️ Development

### Adapting the Engine

**`rag_engine.py`** is designed to be modular:

- **Change embedding model**: Update `AZURE_OPENAI_EMBED_DEPLOYMENT` in `.env`
- **Switch search backend**: Replace Azure Search client with Pinecone, Weaviate, etc.
- **Modify chunking**: Adjust `HierarchicalChunker` parameters or use a different strategy
- **Customize workflow**: Add/remove nodes in the LangGraph StateGraph

### Example: Add Custom Workflow Node

```python
def _custom_filter(self, state: dict) -> dict:
    """Custom node to filter results."""
    passages = state.get("passages", [])
    # Apply custom filtering logic
    state["passages"] = [p for p in passages if some_condition(p)]
    return state

# In _setup_workflow():
workflow.add_node("custom_filter", self._custom_filter)
workflow.add_edge("retriever", "custom_filter")
workflow.add_edge("custom_filter", "answerer")
```

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **KeyError on startup** | Verify all `.env` variables are set correctly |
| **Docling conversion errors** | Check file format compatibility and encoding |
| **Azure authentication errors** | Verify API keys, endpoints, and deployment names |
| **Index schema mismatch** | Ensure `EMBEDING_DIM` matches your embedding model |
| **No citations detected** | Document may lack numbered references; check `debug_references.txt` |

### Debug Mode

To enable detailed logging:

```python
# In rag_engine.py, change:
root_logger.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)
```

## 📦 Project Structure

```
RAG/
├── app/
│   ├── main.py              # FastAPI application
│   └── rag_engine.py        # Core RAG logic
├── frontend/
│   └── app.py               # Streamlit UI
├── data/                    # Sample documents (optional)
├── .env                     # Environment variables (create this)
├── requirements.txt         # Python dependencies
├── test_citation_count.py   # Citation counting test
├── test_end_to_end_citations.py
├── test_usage_of_backend.py
├── test_call_to_app.py
├── rag.txt                  # Log file (generated)
└── README.md                # This file
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ using Azure OpenAI, LangGraph, Docling, and FastAPI**
