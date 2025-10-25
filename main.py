# pip install crewai azure-search-documents openai pydantic
# pip install langchain-docling langchain-core langchain python-dotenv
# pip install "docling~=2.12" azure-search-documents==11.5.2 azure-identity openai rich torch python-dotenv
import os
import logging
from typing import List
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import json

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[RichHandler(), logging.FileHandler("rag.log", mode="w")],
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAG")
logger.setLevel(logging.INFO)

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_CHAT_API_VERSION = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION", "2024-05-01-preview")
AZURE_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_EMBED_API_VERSION = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2024-12-01-preview")

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
EMBEDING_DIM = int(os.environ["EMBEDING_DIM"])
BATCH_SIZE = 50

llm_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_CHAT_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
search_client = SearchClient(
        AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

def create_search_index(index_name: str, index_client:SearchIndexClient):
    # Define fields
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            vector_search_dimensions=EMBEDING_DIM,
            vector_search_profile_name="default",
        ),
    ]
    # Vector search config with an AzureOpenAIVectorizer
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default")],
        profiles=[
            VectorSearchProfile(
                name="default",
                algorithm_configuration_name="default",
                vectorizer_name="default",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="default",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=AZURE_OPENAI_ENDPOINT,
                    deployment_name=AZURE_EMBED_DEPLOYMENT,
                    model_name=AZURE_EMBED_DEPLOYMENT,
                    api_key=AZURE_OPENAI_API_KEY,
                ),
            )
        ],
    )

    # Create or update the index
    new_index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    try:
        index_client.delete_index(index_name)
        logger.info("Deleted existing index '%s'", index_name)
    except Exception:
        logger.debug("Index '%s' did not exist or could not be deleted", index_name, exc_info=True)

    index_client.create_or_update_index(new_index)
    logger.info("Created/updated index '%s'", index_name)

def load_folder(folder: str, converter:DocumentConverter, chunker:HierarchicalChunker) -> List:
    all_chunks = []
    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                result = converter.convert(fpath)
                doc_chunks = list(chunker.chunk(result.document))

                for idx, c in enumerate(doc_chunks,start=len(all_chunks)):
                    chunk_text = c.text
                    all_chunks.append((f"chunk_{idx}", chunk_text))
                logger.info("Loaded %s", fpath)
            except Exception as e:
                logger.exception("Skipped %s due to error", fpath)
    return all_chunks

def embed_text(text: str, llm_client:AzureOpenAI):
    """
    Helper to generate embeddings with Azure OpenAI.
    """
    try:
        logger.debug("Generating embedding for text of length %d", len(text))
        response = llm_client.embeddings.create(
            input=text, model=AZURE_EMBED_DEPLOYMENT
        )
        return response.data[0].embedding
    except Exception:
        logger.exception("Failed to generate embedding")
        raise

def call_llm(prompt: str, system_message: Optional[str] = None):
        """
        Generates a single-turn chat response using Azure OpenAI Chat.
        If you need multi-turn conversation or follow-up queries, you'll have to
        maintain the messages list externally.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug("Sending prompt to chat model (deployment=%s)", AZURE_CHAT_DEPLOYMENT)
            completion = llm_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT, messages=messages, temperature=0
            )
            content = completion.choices[0].message.content
            logger.debug("Received chat response of length %d", len(content))
            return content
        except Exception:
            logger.exception("Chat completion failed")
            raise

def safe_json_list(s: str) -> List[str]:
    """Best-effort parse of a JSON list of strings from an LLM response."""
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    # try to extract [...]-block
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(s[start : end + 1])
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    # fallback: split lines
    lines = [re.sub(r'^[\-\*\d\.\)\s]+', "", ln).strip().strip('"') for ln in s.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:5] or [s.strip()]

class AppState(TypedDict):
    question: str
    rewrites: List[str]
    ranked: List[Dict[str, Any]]   # [{"q": str, "score": float}]
    passages: List[Dict[str, Any]] # [{"text": str, "meta": {...}}]
    answer: str

def rewriter(state: AppState) -> AppState:
    prompt = f"""You are a query rewriter for Azure AI Search.
User question: {state['question']}

Return 3–5 search-optimized rewrites as a JSON array of strings.
Guidelines:
- Include canonical names/titles, synonyms, acronyms expanded.
- Add likely filters (dates, versions, product names) if obvious.
- Keep each rewrite concise and unambiguous.
Return ONLY valid JSON."""
    resp = call_llm(prompt)
    state["rewrites"] = safe_json_list(resp)
    return state

def probe_top_score(
    query: str,
    *,
    vector_field: str = "content_vector",
    knn: int = 5,
) -> float:
    """
    Return the blended @search.score of the FIRST result for this rewrite.
    If no results, return 0.0.
    """
    vq = VectorizableTextQuery(text=query, k_nearest_neighbors=knn, fields=vector_field)
    kwargs = {
        "search_text":query,
        "vector_queries": [vq],
        "select": ["content"],
        "top": 1,
    }

    try:
        resp = search_client.search(**kwargs)
        for doc in resp:
            score = doc.get("@search.score")
            return float(score) if score is not None else 0.0
        return 0.0
    except Exception:
        return 0.0

def ranker(state: AppState) -> AppState:
    scored: List[Dict[str, Any]] = []
    for q in state["rewrites"]:
        s = probe_top_score(q, knn=5)
        scored.append({"q": q, "score": s})
    scored.sort(key=lambda x: x["score"], reverse=True)
    state["ranked"] = scored
    return state

def retrieve_from_db(
    query: str,
    *,
    vector_field: str = "content_vector",
    select_fields: List[str] = ["content"],
    knn: int = 5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Hybrid (keyword+vector) or vector-only retrieval using VectorizableTextQuery.
    Returns a list of dicts with "text" and "meta" (including @search.score).
    """
    vq = VectorizableTextQuery(text=query, k_nearest_neighbors=knn, fields=vector_field)

    kwargs: Dict[str, Any] = {
        "search_text":query,
        "vector_queries": [vq],
        "select": select_fields,
        "top": top_k,
    }

    results: List[Dict[str, Any]] = []
    resp = search_client.search(**kwargs)
    for i, doc in enumerate(resp):
        text = (
            doc.get("content")
            or ""
        )
        results.append(
            {
                "text": text,
                "meta": {
                    "rank": i + 1,
                    "score": doc.get("@search.score"),
                },
            }
        )
    return results

def retriever(state: AppState) -> AppState:
    best_q = (
        state["ranked"][0]["q"]
        if state.get("ranked")
        else (state.get("rewrites", [state["question"]])[0])
    )
    state["passages"] = retrieve_from_db(
        best_q
    )
    return state

def answerer(state: AppState) -> AppState:
    ctx_lines: List[str] = []
    for idx, p in enumerate(state.get("passages", []), start=1):
        meta = p.get("meta", {})
        score = meta.get("score")
        snippet = (p.get("text") or "")[:1500]
        ctx_lines.append(f"[{idx}] score={score}\n{snippet}")
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "(no results)"

    best_q = (
        state["ranked"][0]["q"]
        if state.get("ranked")
        else (state.get("rewrites", [state["question"]])[0])
    )

    prompt = f"""You are a grounded QA assistant. Use ONLY the provided context to answer, and cite sources as [index].
If the context is insufficient, say so and propose a sharper rewrite.

Original question: {state['question']}
Winning rewrite: {best_q}

Context:
{ctx}
"""
    state["answer"] = call_llm(prompt)
    return state

if __name__ == "__main__":
    console = Console()
    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_API_KEY))
    logger.info("Starting RAG indexing process")
    create_search_index(AZURE_SEARCH_INDEX, index_client)
    all_chunks = load_folder("data", converter, chunker)
    search_client = SearchClient(
        AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    llm_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_CHAT_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    upload_docs = []
    for chunk_id, chunk_text in all_chunks:
        try:
            embedding_vector = embed_text(chunk_text, llm_client)
            upload_docs.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "content_vector": embedding_vector,
                }
            )
        except Exception:
            logger.exception("Skipping chunk %s due to embedding failure", chunk_id)

    for i in range(0, len(upload_docs), BATCH_SIZE):
        subset = upload_docs[i : i + BATCH_SIZE]
        resp = search_client.upload_documents(documents=subset)

        all_succeeded = all(r.succeeded for r in resp)
        logger.info(
            "Uploaded batch %d -> %d; all_succeeded: %s, first_doc_status_code: %s",
            i,
            i + len(subset),
            all_succeeded,
            getattr(resp[0], "status_code", None),
        )

    logger.info("All chunks uploaded to Azure Search. Total uploaded: %d", len(upload_docs))

    workflow = StateGraph(AppState)
    workflow.add_node("rewriter", rewriter)
    workflow.add_node("ranker", ranker)
    workflow.add_node("retriever", retriever)
    workflow.add_node("answerer", answerer)

    workflow.set_entry_point("rewriter")
    workflow.add_edge("rewriter", "ranker")
    workflow.add_edge("ranker", "retriever")
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", END)

    graph = workflow.compile()

    question = "What is attention mechanizm?"
    result = graph.invoke({"question": question})

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])
    print("\n=== DEBUG ===")
    print("Rewrites:", result.get("rewrites"))
    print("Ranked:", result.get("ranked"))
    logging.shutdown()