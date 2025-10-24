# pip install crewai azure-search-documents openai pydantic
# pip install langchain-docling langchain-core langchain python-dotenv
import os
import glob
import tiktoken
from typing import List
from pathlib import Path


from dotenv import load_dotenv
from langchain_docling.loader import ExportType
from langchain_core.prompts import PromptTemplate

from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# from langchain_core.document_loaders import BaseLoader
# from langchain_core.text_splitter import RecursiveCharacterTextSplitter

# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_core.chains import RetrievalQA
# from langchain_core.llms import AzureChatOpenAI

load_dotenv()

# ==== ENV (настрой ↓) ====
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_EMBED_API_VERSION = os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2024-12-01-preview")

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
# VECTOR_FIELD = os.environ.get("AZURE_SEARCH_VECTOR_FIELD", "contentVector")
TOP_K = int(os.environ.get("TOP_K", "4"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"
os.environ["AZURESEARCH_FIELDS_TAG"] = "meta_json_string"

SUPPORTED_EXTENSIONS = {
    ".pdf": "docling",
    ".docx": "docling",
    ".pptx": "docling",
    ".html": "docling",
    ".md": "markdown",
    ".json": "docling",
}

def load_file(path: str) -> List:
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS and SUPPORTED_EXTENSIONS[ext] == "docling":
        loader = DoclingLoader(
                    file_path=path,
                    export_type=ExportType.DOC_CHUNKS,
                    # chunker=HybridChunker(
                    #     tokenizer = OpenAITokenizer(
                    #             tokenizer=tiktoken.encoding_for_model("gpt-oss-120b"),
                    #             max_tokens=128 * 1024,
                    #         )
                    #     ),
                )
    elif ext in SUPPORTED_EXTENSIONS and SUPPORTED_EXTENSIONS[ext] == "markdown":
        loader = DoclingLoader(
                    file_path=path,
                    export_type=ExportType.MARKDOWN,
                    # chunker=HybridChunker(
                    #     tokenizer = OpenAITokenizer(
                    #             tokenizer=tiktoken.encoding_for_model("gpt-oss-120b"),
                    #             max_tokens=128 * 1024,
                    #         )
                    #     ),
                )
        docs = loader.load()
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        return [split for doc in docs for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Unsupported file type {ext} for path {path}")
    return loader.load()

def load_folder(folder: str) -> List:
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                docs.extend(load_file(fpath))
                print(f"Loaded {fpath}")
            except Exception as e:
                print(f"Skipped {fpath}: {e}")
    return docs

if __name__ == "__main__":
    all_docs = load_folder("data")
    
    # for d in all_docs:
    #     print(d)

    aoai_embeddings = AzureOpenAIEmbeddings(
        model=AZURE_EMBED_DEPLOYMENT,         # имя деплоймента эмбеддингов в Azure OpenAI
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
    )

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_API_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=aoai_embeddings.embed_query,
        # metadata_key="meta_json_string",
    )

    # добавляем документы (chunk’и Docling → векторный индекс)
    _ = vector_store.add_documents(all_docs)