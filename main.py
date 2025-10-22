import os
from operator import itemgetter
from typing import List, Dict, Any

# --- Azure OpenAI (via Azure AI Foundry) ---
# SDK: https://learn.microsoft.com/azure/ai-foundry/openai/how-to/chatgpt
from openai import OpenAI

# --- Azure AI Search (vector search) ---
# SDK: https://learn.microsoft.com/python/api/overview/azure/search-documents-readme
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# --- CrewAI ---
from crewai import Agent, Task, Crew, Tool

# -------------------------
# ENV VARS (set these!)
# -------------------------
# Azure AI Foundry / Azure OpenAI
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]  # e.g. "https://<your-foundry-endpoint>.openai.azure.com"
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-oss-120b")    # your chat model deployment name
AZURE_OPENAI_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")  # your embed model deployment name

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]  # e.g. "https://<your-search>.search.windows.net"
AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]        # your vector-enabled index name
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]

# Retrieval settings
TOP_K = int(os.environ.get("TOP_K", "4"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "3072"))  # text-embedding-3-large; adjust if you use a different model

# -------------------------
# Azure clients
# -------------------------
# Azure OpenAI (Foundry) client
# Docs: chat completions / embeddings usage via Foundry models.
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1",
    api_key=AZURE_OPENAI_API_KEY,
    default_headers={"api-key": AZURE_OPENAI_API_KEY},  # Foundry expects this header
)

# Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

# -------------------------
# Embedding & LLM helpers
# -------------------------
def embed_text(text: str) -> List[float]:
    """
    Get embeddings from Azure OpenAI (Foundry).
    """
    resp = client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=text,
    )
    return resp.data[0].embedding

def chat_complete(system_prompt: str, user_prompt: str) -> str:
    """
    Call Azure OpenAI chat completion (Foundry).
    """
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# -------------------------
# Retriever over Azure AI Search
# -------------------------
def search_top_k(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Vector search over Azure AI Search.
    Assumes your index has:
      - a vector field (e.g., 'contentVector') with EMBED_DIM dims
      - a content text field (e.g., 'content')
      - optional metadata fields like 'source', 'page'
    """
    query_vec = embed_text(query)
    vector_query = {
        "value": query_vec,
        "fields": "contentVector",  # <-- change to your vector field name
        "k": top_k,
    }

    # You can combine with keyword search by setting search_text=query (hybrid).
    results = search_client.search(
        search_text=None,
        vector=vector_query,
        select=["content", "source", "page"],  # <-- modify to match your schema
        top=top_k,
    )

    hits = []
    for r in results:
        hits.append({
            "content": r["content"],
            "source": r.get("source"),
            "page": r.get("page"),
        })
    return hits

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Concatenate retrieved chunks into a prompt-friendly context.
    """
    parts = []
    for i, ch in enumerate(chunks, start=1):
        src = f"(source={ch.get('source')}, page={ch.get('page')})" if (ch.get("source") or ch.get("page")) else ""
        parts.append(f"[{i}] {src}\n{ch['content']}")
    return "\n\n".join(parts)

# -------------------------
# CrewAI Tool
# -------------------------
def rag_tool_func(question: str) -> str:
    """
    Full RAG flow:
      1) retrieve from Azure Search
      2) build context
      3) ask Azure OpenAI to answer strictly from context
    """
    docs = search_top_k(question, TOP_K)
    context = build_context(docs)

    system_prompt = (
        "You are a helpful assistant that answers ONLY using the provided context. "
        "If the answer cannot be found in the context, say you don't know."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Cite the bracket numbers [1], [2], ... when relevant.\n"
        "- If insufficient information, say 'I don't know based on the provided documents.'"
    )

    answer = chat_complete(system_prompt, user_prompt)

    # Optional: bundle sources at the end
    sources = []
    for i, d in enumerate(docs, start=1):
        tag = f"[{i}]"
        src = d.get("source")
        pg = d.get("page")
        if src or pg:
            sources.append(f"{tag} {src or ''} {('p.' + str(pg)) if pg else ''}".strip())

    if sources:
        answer += "\n\nSources:\n" + "\n".join(sources)

    return answer

rag_tool = Tool(
    name="AzureSearchRAG",
    description="Answers questions using Azure AI Search (vector) and Azure OpenAI (Foundry) based on indexed PDF chunks.",
    func=rag_tool_func,
    return_direct=True,
)

# -------------------------
# CrewAI Agent & Task
# -------------------------
qa_agent = Agent(
    role="PDF Q&A Agent",
    goal="Answer user questions strictly using the enterprise PDF knowledge indexed in Azure AI Search.",
    backstory=(
        "You are an internal assistant for PDF knowledge. "
        "Use the AzureSearchRAG tool; never invent facts beyond retrieved context."
    ),
    tools=[rag_tool],
    verbose=True,
)

qa_task = Task(
    description="Answer the user's question using only the information from the indexed PDF content.",
    agent=qa_agent,
    expected_output="A concise answer grounded in the provided PDF context, with bracketed citations.",
)

crew = Crew(
    agents=[qa_agent],
    tasks=[qa_task],
    verbose=True,
)

if __name__ == "__main__":
    # Example run
    user_q = input("Ask a question about your PDFs: ")
    result = crew.kickoff(inputs={"question": user_q})
    print("\n=== ANSWER ===\n")
    print(result)