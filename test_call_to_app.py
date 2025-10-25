
import requests
import json
from typing import Dict, Any
from dotenv import load_dotenv
import os

def query_rag(question: str) -> Dict[str, Any]:
    """
    Sends a query to the RAG API and returns the response.
    
    Args:
        question (str): Question for the RAG system
        
    Returns:
        Dict[str, Any]: API response containing result and metadata
    """
    
    # Your API URL (change as needed)
    API_URL = "http://localhost:8000/query"
    
    # Prepare request data
    payload = {
        "question": question
    }
    
    try:
        # Send POST request
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response status
        response.raise_for_status()
        
        # Return result
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Usage example
    question = "What is attention mechanism?"
    result = query_rag(question)
    
    print("\nQuery:", question)
    print("\nResponse:")
    if "error" not in result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Error:", result["error"])