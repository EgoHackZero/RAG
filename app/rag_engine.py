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

# Keywords for detecting metadata queries
AUTHOR_KEYWORDS = [
    "author", "authors", "writer", "who wrote", "who is the author",
    "автор", "авторы", "кто написал", "кто автор"
]

REFERENCE_KEYWORDS = [
    "reference", "citation", "bibliography", "how many", "count",
    "ссылк", "референс", "цитат", "сколько"
]

REFERENCE_SECTION_HEADINGS = [
    "references", "bibliography", "literature", "литература"
]

DEFAULT_LANGUAGE = "en"


class DocumentSummary(TypedDict, total=False):
    source: str
    title: Optional[str]
    authors: List[str]
    reference_count: Optional[int]


class RAGEngine:
    def __init__(self):
        load_dotenv()
        self._setup_logging()
        self._load_config()
        self._initialize_clients()
        self._setup_workflow()
        self.doc_metadata: Dict[str, DocumentSummary] = {}
        self._metadata_hydrated = False
        self._hydrate_metadata_from_index()

    def _setup_logging(self):
        # Get root logger to capture ALL logs from all modules
        root_logger = logging.getLogger()

        # Only configure if not already configured (avoid duplicate handlers)
        if not root_logger.handlers:
            root_logger.setLevel(logging.INFO)

            # Detailed format with module name
            log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

            # Console handler with Rich formatting (for interactive use)
            console_handler = RichHandler(rich_tracebacks=True, show_path=False)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            # File handler - append mode to preserve logs across restarts
            file_handler = logging.FileHandler("rag.txt", mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Capture more detail in file
            file_handler.setFormatter(formatter)

            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

            # Reduce noise from verbose third-party libraries
            logging.getLogger("azure").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

        # Get module-specific logger
        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging.INFO)

        # Log session start marker
        self.logger.info("=" * 80)
        self.logger.info("RAG Engine initialized - New session started")
        self.logger.info("=" * 80)

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

                    chunk_texts: List[str] = []
                    doc_slug = self._slugify(os.path.splitext(fname)[0])

                    for idx, c in enumerate(doc_chunks):
                        chunk_text = c.text
                        chunk_texts.append(chunk_text)
                        chunk_id = f"{doc_slug}_chunk_{idx}"
                        all_chunks.append((chunk_id, chunk_text))

                    metadata_chunk = self._build_metadata_chunk(fpath, result.document, chunk_texts)
                    if metadata_chunk:
                        all_chunks.append((f"{doc_slug}_metadata", metadata_chunk))

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
        workflow.add_node("language_detector", self._language_detector)
        workflow.add_node("rewriter", self._rewriter)
        workflow.add_node("ranker", self._ranker)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("answerer", self._answerer)

        workflow.set_entry_point("language_detector")
        workflow.add_edge("language_detector", "rewriter")
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

    def _build_rewriter_context(self) -> str:
        if not self.doc_metadata:
            self._hydrate_metadata_from_index()
        if not self.doc_metadata:
            return "No additional metadata is available."

        snippets: List[str] = []
        for doc_name, meta in list(self.doc_metadata.items())[:5]:
            title = meta.get("title") or doc_name
            authors = meta.get("authors") or []
            ref_count = meta.get("reference_count")
            parts = [f"Title: {title}"]
            if doc_name != title:
                parts.append(f"File: {doc_name}")
            if authors:
                parts.append("Authors: " + ", ".join(authors[:5]))
            if ref_count is not None:
                parts.append(f"References: {ref_count}")
            snippets.append("; ".join(parts))

        return "\n".join(snippets)

    def _language_detector(self, state: dict) -> dict:
        """Detect user's language and translate question to English for search."""
        question = state.get("question", "")
        language = self._detect_language(question)

        state["user_language"] = language
        state["original_question"] = question
        state["question_en"] = self._translate_to_english(question, language)

        return state

    def _rewriter(self, state: dict) -> dict:
        """Generate multiple search query variants."""
        english_question = state.get("question_en", "")
        context = self._build_rewriter_context()

        prompt = f"""Generate 3-5 search query variants for this question to improve retrieval from Azure AI Search.

Question: {english_question}

Document context:
{context}

Return ONLY a JSON array of 3-5 search queries."""

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
        """Generate answer in user's language based on retrieved context."""
        # Build context from retrieved passages
        ctx_lines = []
        for idx, p in enumerate(state.get("passages", []), start=1):
            snippet = (p.get("text") or "")[:1500]
            ctx_lines.append(f"[{idx}] {snippet}")
        context = "\n\n".join(ctx_lines) if ctx_lines else "(no relevant information found)"

        language = state.get("user_language", "English")
        original_q = state.get("original_question", state.get("question", ""))

        prompt = f"""Answer the question using ONLY the provided context. Cite sources as [1], [2], etc.

IMPORTANT: Respond in {language}.

Question: {original_q}

Context:
{context}

Answer (in {language}):"""

        state["answer"] = self._call_llm(prompt)
        return state

    def process_query(self, question: str) -> Dict[str, Any]:
        language = self._detect_language(question)
        metadata_first = self._metadata_answer_if_possible(question, language)
        if metadata_first:
            metadata_first.setdefault("debug", {})
            metadata_first["debug"]["search_language"] = "en"
            metadata_first["debug"]["answer_language"] = language
            return metadata_first

        initial_state = {"question": question, "user_language": language}
        result = self.graph.invoke(initial_state)
        final_language = result.get("user_language") or language or DEFAULT_LANGUAGE
        return {
            "answer": result.get("answer"),
            "debug": {
                "rewrites": result.get("rewrites"),
                "ranked": result.get("ranked"),
                "user_language": final_language,
                "search_language": "en",
                "answer_language": final_language,
            }
        }

    def _metadata_answer_if_possible(self, question: str, language: str) -> Optional[Dict[str, Any]]:
        """Try to answer from metadata if question is about authors or citation counts."""
        if not self._is_metadata_question(question):
            return None

        if not self.doc_metadata:
            self._hydrate_metadata_from_index()
        if not self.doc_metadata:
            return None

        # Build metadata summary
        metadata_text = []
        for doc_name, meta in self.doc_metadata.items():
            parts = [f"Document: {doc_name}"]
            if meta.get("title"):
                parts.append(f"Title: {meta['title']}")
            if meta.get("authors"):
                parts.append(f"Authors: {', '.join(meta['authors'])}")
            if meta.get("reference_count") is not None:
                parts.append(f"Citations: {meta['reference_count']}")
            metadata_text.append(" | ".join(parts))

        prompt = f"""Answer this question using the document metadata provided. Respond in {language}.

Question: {question}

Metadata:
{chr(10).join(metadata_text)}

Instructions:
- Answer the question directly and concisely
- Respond in {language}
- If the answer is not in the metadata, respond with only: USE_RETRIEVAL

Answer:"""

        response = self._call_llm(prompt).strip()
        if response.upper() == "USE_RETRIEVAL" or not response:
            return None

        return {
            "answer": response,
            "debug": {"used_metadata": True, "user_language": language}
        }

    def _is_metadata_question(self, question: str) -> bool:
        """Check if question is about authors or citations."""
        q = question.lower()
        return (any(kw in q for kw in AUTHOR_KEYWORDS) or
                any(kw in q for kw in REFERENCE_KEYWORDS))

    def _hydrate_metadata_from_index(self) -> None:
        if getattr(self, "_metadata_hydrated", False):
            return

        try:
            response = self.search_client.search(
                search_text="Document-level metadata summary",
                select=["content"],
                top=100,
            )
            for doc in response:
                summary = self._parse_metadata_chunk_text(doc.get("content") or "")
                if summary and summary.get("source"):
                    self.doc_metadata[summary["source"]] = summary
        except Exception:
            self.logger.debug("Unable to hydrate metadata from index", exc_info=True)
            return

        self._metadata_hydrated = True

    def _parse_metadata_chunk_text(self, text: str) -> Optional[DocumentSummary]:
        if not text:
            return None

        doc_name: Optional[str] = None
        title: Optional[str] = None
        authors: List[str] = []
        ref_count: Optional[int] = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("source file:"):
                doc_name = line.split(":", 1)[1].strip()
            elif lower.startswith("document title:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    title = value
            elif lower.startswith("document authors:"):
                value = line.split(":", 1)[1]
                authors = [
                    part.strip()
                    for part in value.split(",")
                    if part.strip()
                ]
            elif lower.startswith("total references counted:") \
                or lower.startswith("referencecount:") \
                or lower.startswith("reference count:") \
                or lower.startswith("numberofreferences:") \
                or lower.startswith("number of references:") \
                or lower.startswith("referencestotal:") \
                or lower.startswith("references total:") \
                or lower.startswith("references total") \
                or lower.startswith("количество ссылок") \
                or lower.startswith("количество референсов"):
                numbers = re.findall(r"\d+", line)
                if numbers:
                    try:
                        ref_count = int(numbers[0])
                    except ValueError:
                        ref_count = None

        if not doc_name:
            return None

        summary: DocumentSummary = {"source": doc_name}
        if title:
            summary["title"] = title
        if authors:
            summary["authors"] = authors
        if ref_count is not None:
            summary["reference_count"] = ref_count
        return summary

    def _detect_language(self, text: str) -> str:
        """Detect language using LLM - supports any language."""
        if not text:
            return DEFAULT_LANGUAGE

        # Quick check for English-only text
        if all(ord(ch) < 128 for ch in text) and not any(ch in text for ch in "ñäöüß"):
            return DEFAULT_LANGUAGE

        # Use LLM to detect language - simpler and supports all languages
        snippet = text[:300]
        prompt = f"""Detect the language of this text and respond with ONLY the full language name (e.g., "English", "Russian", "Spanish", "Chinese", "French", etc.):

Text: {snippet}"""

        try:
            response = self._call_llm(prompt, system_message="You are a language detection assistant. Respond with only the language name.")
            return (response or "English").strip()
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e}")
            return "English"

    def _translate_to_english(self, text: str, source_language: str) -> str:
        """Translate text to English if needed."""
        if not text or source_language.lower() == "english":
            return text

        prompt = f"""Translate this {source_language} text to English. Preserve technical terms:

{text}

Translation:"""

        try:
            return self._call_llm(prompt).strip() or text
        except Exception as e:
            self.logger.debug(f"Translation failed: {e}")
            return text

    def _slugify(self, value: str) -> str:
        value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
        return value or "doc"

    def _normalize_authors(self, authors_obj: Any) -> List[str]:
        if not authors_obj:
            return []

        authors: List[str] = []
        if isinstance(authors_obj, str):
            pieces = re.split(r"[;,]", authors_obj)
            authors = [piece.strip() for piece in pieces if piece.strip()]
        elif isinstance(authors_obj, list):
            collected: List[str] = []
            for item in authors_obj:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("author") or item.get("full_name")
                    if name:
                        collected.append(str(name).strip())
                else:
                    collected.append(str(item).strip())
            authors = [name for name in collected if name]

        seen = set()
        unique_authors: List[str] = []
        for name in authors:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique_authors.append(name)
        return unique_authors

    def _extract_metadata(self, file_path: str, document: Any, chunk_texts: List[str]) -> DocumentSummary:
        doc_name = os.path.basename(file_path)
        summary: DocumentSummary = {"source": doc_name}

        doc_meta = getattr(document, "metadata", None)
        if isinstance(doc_meta, dict):
            summary["title"] = doc_meta.get("title") or doc_meta.get("Title")
            summary["authors"] = self._normalize_authors(
                doc_meta.get("authors") or doc_meta.get("Authors")
            )
        else:
            summary["title"] = None
            summary["authors"] = []

        if not summary["authors"] and chunk_texts:
            header_text = "\n".join(chunk_texts[:2])
            potential: List[str] = []
            for line in header_text.splitlines():
                cleaned = line.strip()
                if not cleaned:
                    if potential:
                        break
                    continue
                if len(cleaned) > 120:
                    break
                if re.search(r"(university|department|school|institute)", cleaned, re.IGNORECASE):
                    break
                if re.search(r"[A-Z][a-z]+", cleaned):
                    potential.append(cleaned)
            if potential:
                summary["authors"] = self._normalize_authors(potential)

        full_text = "\n".join(chunk_texts)
        references_section = self._get_references_section(full_text)
        summary["reference_count"] = self._count_reference_entries(references_section)

        return summary

    def _get_references_section(self, text: str) -> str:
        if not text:
            return ""

        # Strategy 1: Look for explicit "References" heading
        match = None
        for heading in REFERENCE_SECTION_HEADINGS:
            pattern = rf"(?:^|\n)\s*{re.escape(heading)}\s*:?\s*\n"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                break

        if match:
            start = match.start()
            remainder = text[start:]
            stop_match = re.search(
                r"\n\s*(appendix|supplementary|acknowledg(e)?ments?|annex|appendices)\b",
                remainder,
                re.IGNORECASE,
            )
            if stop_match:
                end = stop_match.start()
                return remainder[:end]
            return remainder

        # Strategy 2: No heading found - look for where numbered references start
        # Look for patterns like: "- [1]", "[1]", "1.", "1)" at the start of a line
        lines = text.split('\n')
        first_ref_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match common reference patterns at line start
            if re.match(r'^-?\s*\[\s*1\s*\]', stripped) or \
               re.match(r'^\[?\s*1\s*\]?[\.\)]\s+[A-Z]', stripped) or \
               re.match(r'^1[\.\)]\s+[A-Z]', stripped):
                first_ref_idx = i
                break

        if first_ref_idx is None:
            return ""

        # Extract from first reference to end
        remainder_lines = lines[first_ref_idx:]
        remainder = '\n'.join(remainder_lines)

        # Stop at acknowledgments, contributor lists, or other non-reference sections
        stop_match = re.search(
            r"\n\s*(appendix|supplementary|acknowledg(e)?ments?|annex|appendices|"
            r"the following individuals|contributors?|we completed this work)\b",
            remainder,
            re.IGNORECASE,
        )
        if stop_match:
            end = stop_match.start()
            return remainder[:end]

        return remainder

    def _looks_like_reference(self, block: str) -> bool:
        block = block.strip()
        if not block:
            return False
        if len(block.split()) < 4:
            return False
        if re.match(r"^\[?\d+\]?[\.\)]", block):
            return True
        if re.search(r"\(\d{4}\)", block):
            return True
        if re.search(r"\d{4}", block) and re.search(r"[A-Z][a-z]+", block):
            return True
        if re.search(r"doi:", block, re.IGNORECASE):
            return True
        return False

    def _count_reference_entries(self, references_section: str) -> Optional[int]:
        if not references_section:
            return None
        lines = references_section.splitlines()

        # Strategy 1: use numbered reference markers (e.g., [12], 12., 12))
        numbered_markers: List[int] = []

        # First, try to find [number] patterns that indicate reference starts
        # Look for patterns like "[1]", "[2]", etc. that appear at line starts or after newlines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Try bracket notation with optional dash prefix: - [1], [2], etc.
            bracket_match = re.match(r'^-?\s*\[\s*(\d{1,3})\s*\]', stripped)
            if bracket_match:
                numbered_markers.append(int(bracket_match.group(1)))
                continue

            # Matches forms like "12." or "12)" or "12 -" at the start of a line
            leading_match = re.match(r"^(\d{1,3})(?:[\.\)]|\s+-)\s+", stripped)
            if leading_match:
                numbered_markers.append(int(leading_match.group(1)))
                continue

            # Matches forms like "12 " followed by uppercase (common in IEEE format)
            loose_match = re.match(r"^(\d{1,3})\s+[A-Z]", stripped)
            if loose_match:
                numbered_markers.append(int(loose_match.group(1)))

        # If we found bracket-style markers, also search the entire text for any we missed
        # This handles cases where PDF extraction doesn't preserve line breaks correctly
        if numbered_markers:
            # Find all [number] patterns in the entire references section (with optional dash prefix)
            all_bracket_markers = re.findall(r'-?\s*\[(\d{1,3})\]', references_section)
            for marker in all_bracket_markers:
                num = int(marker)
                # Only include reasonable reference numbers (1-999)
                if 1 <= num <= 999 and num not in numbered_markers:
                    numbered_markers.append(num)

        if numbered_markers:
            highest = max(numbered_markers)
            # Guard against obviously bad detections (e.g., stray page numbers)
            # For academic papers, if we found markers, return the highest number found
            # This assumes sequential numbering (which is standard)
            if highest > 0 and highest <= 200:  # Reasonable upper bound for references
                return highest

        # Strategy 2: split into logical blocks separated by blank lines
        blocks: List[str] = []
        current_block: List[str] = []
        for line in lines:
            if line.strip():
                current_block.append(line)
            elif current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        if current_block:
            blocks.append("\n".join(current_block))

        block_count = sum(1 for blk in blocks if self._looks_like_reference(blk))
        if block_count > 0:
            return block_count

        # Strategy 3: fall back to counting individual lines that look like references
        line_count = sum(1 for line in lines if self._looks_like_reference(line))
        return line_count if line_count > 0 else None

    def _build_metadata_chunk(self, file_path: str, document: Any, chunk_texts: List[str]) -> Optional[str]:
        """Build a metadata chunk for fast author/citation lookups."""
        try:
            summary = self._extract_metadata(file_path, document, chunk_texts)
            doc_name = os.path.basename(file_path)
            self.doc_metadata[doc_name] = summary
            self._metadata_hydrated = True

            lines = [
                "Document-level metadata summary for retrieval.",
                f"Source file: {doc_name}",
            ]

            if summary.get("title"):
                lines.append(f"Document title: {summary['title']}")

            if summary.get("authors"):
                authors_str = ", ".join(summary["authors"])
                lines.append(f"Document authors: {authors_str}")

            if summary.get("reference_count") is not None:
                ref_count = summary["reference_count"]
                lines.append(f"Reference count: {ref_count}")
                lines.append(f"Citations: {ref_count}")
                lines.append(f"Number of references: {ref_count}")

            return "\n".join(lines) if len(lines) > 2 else None
        except Exception as e:
            self.logger.debug(f"Failed to build metadata chunk: {e}")
            return None
