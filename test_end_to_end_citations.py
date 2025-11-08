"""
End-to-end test for citation counting in the RAG system.
Tests that the document is indexed correctly and queries return accurate citation counts.
"""
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from rag_engine import RAGEngine

def main():
    print("=" * 80)
    print("End-to-End Citation Count Test")
    print("=" * 80)

    # Initialize RAG engine
    print("\n1. Initializing RAG engine...")
    engine = RAGEngine()

    # Create fresh index
    print("\n2. Creating fresh search index...")
    engine.create_search_index()

    # Load the test document
    print("\n3. Loading document: RAG/data/2509.13348v4.pdf")
    data_path = "D:/astudy/SaSW/RAG/data"
    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}")
        return

    uploaded_docs = engine.load_documents(data_path)
    print(f"   Uploaded {len(uploaded_docs)} chunks to index")

    # Check metadata
    print("\n4. Checking metadata extraction...")
    for doc_name, meta in engine.doc_metadata.items():
        print(f"   Document: {doc_name}")
        print(f"   - Title: {meta.get('title', 'N/A')}")
        print(f"   - Authors: {meta.get('authors', [])}")
        print(f"   - Reference count: {meta.get('reference_count', 'N/A')}")

    # Test queries
    print("\n5. Testing citation count queries...")
    print("-" * 80)

    test_queries = [
        "How many citations are in the document?",
        "How many references does this paper have?",
        "What is the citation count?",
        "Сколько ссылок в документе?"  # Russian: How many references in the document?
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = engine.process_query(query)
        answer = result.get("answer", "")
        print(f"Answer: {answer}")

        # Check if 16 is mentioned in the answer
        if "16" in answer:
            print("✓ PASS: Answer contains '16'")
        else:
            print("✗ FAIL: Answer does not contain '16'")

        # Show debug info
        if result.get("debug", {}).get("used_metadata"):
            print("  (Used metadata fast-path)")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
