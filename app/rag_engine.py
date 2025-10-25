from typing import List, Dict, Any, Optional, TypedDict
import logging
import json
import re
import os
from rich.logging import RichHandler
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
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
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END

class RAGEngine:
    def __init__(self):
        load_dotenv()
        self._setup_logging()
        self._load_config()
        self._initialize_clients()
        self._setup_workflow()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[RichHandler(), logging.FileHandler("rag.txt", mode="w")],
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging.INFO)

    def _load_config(self):
        self.AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
        self.AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
        self.AZURE_OPENAI_CHAT_API_VERSION = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION", "2024-05-01-preview")
        self.AZURE_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
        self.AZURE_OPENAI_EMBED_API_VERSION = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2024-12-01-preview")
        self.AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
        self.AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
        self.AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
        self.EMBEDING_DIM = int(os.environ["EMBEDING_DIM"])
        self.BATCH_SIZE = 50

    def _initialize_clients(self):
        self.llm_client = AzureOpenAI(
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_CHAT_API_VERSION,
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
        )
        self.index_client = SearchIndexClient(
            self.AZURE_SEARCH_ENDPOINT, 
            AzureKeyCredential(self.AZURE_SEARCH_API_KEY)
        )
        self.search_client = SearchClient(
            self.AZURE_SEARCH_ENDPOINT, 
            self.AZURE_SEARCH_INDEX, 
            AzureKeyCredential(self.AZURE_SEARCH_API_KEY)
        )
        self.document_converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def create_search_index(self):
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
                vector_search_dimensions=self.EMBEDING_DIM,
                vector_search_profile_name="default",
            ),
        ]
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
                        resource_url=self.AZURE_OPENAI_ENDPOINT,
                        deployment_name=self.AZURE_EMBED_DEPLOYMENT,
                        model_name=self.AZURE_EMBED_DEPLOYMENT,
                        api_key=self.AZURE_OPENAI_API_KEY,
                    ),
                )
            ],
        )

        new_index = SearchIndex(
            name=self.AZURE_SEARCH_INDEX, 
            fields=fields, 
            vector_search=vector_search
        )
        
        try:
            self.index_client.delete_index(self.AZURE_SEARCH_INDEX)
            self.logger.info(f"Deleted existing index '{self.AZURE_SEARCH_INDEX}'")
        except Exception:
            self.logger.debug(
                f"Index '{self.AZURE_SEARCH_INDEX}' did not exist or could not be deleted", 
                exc_info=True
            )

        self.index_client.create_or_update_index(new_index)
        self.logger.info(f"Created/updated index '{self.AZURE_SEARCH_INDEX}'")

    def load_documents(self, folder_path: str) -> List[Dict[str, Any]]:
        all_chunks = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    result = self.document_converter.convert(fpath)
                    doc_chunks = list(self.chunker.chunk(result.document))

                    for idx, c in enumerate(doc_chunks, start=len(all_chunks)):
                        chunk_text = c.text
                        all_chunks.append((f"chunk_{idx}", chunk_text))
                    self.logger.info(f"Loaded {fpath}")
                except Exception as e:
                    self.logger.exception(f"Skipped {fpath} due to error")
        
        return self._process_chunks(all_chunks)

    def _process_chunks(self, chunks: List[tuple]) -> List[Dict[str, Any]]:
        upload_docs = []
        for chunk_id, chunk_text in chunks:
            try:
                embedding_vector = self._embed_text(chunk_text)
                upload_docs.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "content_vector": embedding_vector,
                })
            except Exception:
                self.logger.exception(f"Skipping chunk {chunk_id} due to embedding failure")
        
        self._upload_documents(upload_docs)
        return upload_docs

    def _upload_documents(self, documents: List[Dict[str, Any]]):
        for i in range(0, len(documents), self.BATCH_SIZE):
            subset = documents[i : i + self.BATCH_SIZE]
            resp = self.search_client.upload_documents(documents=subset)
            
            all_succeeded = all(r.succeeded for r in resp)
            self.logger.info(
                f"Uploaded batch {i} -> {i + len(subset)}; all_succeeded: {all_succeeded}"
            )

    def _embed_text(self, text: str) -> List[float]:
        try:
            self.logger.debug(f"Generating embedding for text of length {len(text)}")
            response = self.llm_client.embeddings.create(
                input=text, model=self.AZURE_EMBED_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception:
            self.logger.exception("Failed to generate embedding")
            raise

    def _call_llm(self, prompt: str, system_message: Optional[str] = None) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            self.logger.debug(f"Sending prompt to chat model (deployment={self.AZURE_CHAT_DEPLOYMENT})")
            completion = self.llm_client.chat.completions.create(
                model=self.AZURE_CHAT_DEPLOYMENT, 
                messages=messages, 
                temperature=0
            )
            content = completion.choices[0].message.content
            self.logger.debug(f"Received chat response of length {len(content)}")
            return content
        except Exception:
            self.logger.exception("Chat completion failed")
            raise

    def _setup_workflow(self):
        workflow = StateGraph(dict)
        workflow.add_node("rewriter", self._rewriter)
        workflow.add_node("ranker", self._ranker)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("answerer", self._answerer)

        workflow.set_entry_point("rewriter")
        workflow.add_edge("rewriter", "ranker")
        workflow.add_edge("ranker", "retriever")
        workflow.add_edge("retriever", "answerer")
        workflow.add_edge("answerer", END)

        self.graph = workflow.compile()

    def _safe_json_list(self, s: str) -> List[str]:
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1:
            try:
                data = json.loads(s[start : end + 1])
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                pass
        
        lines = [
            re.sub(r'^[\-\*\d\.\)\s]+', "", ln).strip().strip('"') 
            for ln in s.splitlines() 
            if ln.strip()
        ]
        return [ln for ln in lines if ln][:5] or [s.strip()]

    def _rewriter(self, state: dict) -> dict:
        prompt = f"""You are a query rewriter for Azure AI Search.
User question: {state['question']}

Return 3–5 search-optimized rewrites as a JSON array of strings.
Guidelines:
- Include canonical names/titles, synonyms, acronyms expanded.
- Add likely filters (dates, versions, product names) if obvious.
- Keep each rewrite concise and unambiguous.
Return ONLY valid JSON."""
        resp = self._call_llm(prompt)
        state["rewrites"] = self._safe_json_list(resp)
        return state

    def _probe_top_score(
        self,
        query: str,
        vector_field: str = "content_vector",
        knn: int = 5,
    ) -> float:
        vq = VectorizableTextQuery(text=query, k_nearest_neighbors=knn, fields=vector_field)
        kwargs = {
            "search_text": query,
            "vector_queries": [vq],
            "select": ["content"],
            "top": 1,
        }

        try:
            resp = self.search_client.search(**kwargs)
            for doc in resp:
                score = doc.get("@search.score")
                return float(score) if score is not None else 0.0
            return 0.0
        except Exception:
            return 0.0

    def _ranker(self, state: dict) -> dict:
        scored = []
        for q in state["rewrites"]:
            s = self._probe_top_score(q, knn=5)
            scored.append({"q": q, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        state["ranked"] = scored
        return state

    def _retrieve_from_db(
        self,
        query: str,
        vector_field: str = "content_vector",
        select_fields: List[str] = ["content"],
        knn: int = 5,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        vq = VectorizableTextQuery(text=query, k_nearest_neighbors=knn, fields=vector_field)
        kwargs = {
            "search_text": query,
            "vector_queries": [vq],
            "select": select_fields,
            "top": top_k,
        }

        results = []
        resp = self.search_client.search(**kwargs)
        for i, doc in enumerate(resp):
            text = doc.get("content") or ""
            results.append({
                "text": text,
                "meta": {
                    "rank": i + 1,
                    "score": doc.get("@search.score"),
                },
            })
        return results

    def _retriever(self, state: dict) -> dict:
        best_q = (
            state["ranked"][0]["q"]
            if state.get("ranked")
            else (state.get("rewrites", [state["question"]])[0])
        )
        state["passages"] = self._retrieve_from_db(best_q)
        return state

    def _answerer(self, state: dict) -> dict:
        ctx_lines = []
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
        state["answer"] = self._call_llm(prompt)
        return state

    def process_query(self, question: str) -> Dict[str, Any]:
        result = self.graph.invoke({"question": question})
        return {
            "answer": result.get("answer"),
            "debug": {
                "rewrites": result.get("rewrites"),
                "ranked": result.get("ranked")
            }
        }