import streamlit as st
import requests
import json
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

def initialize_index():
    """Initialize index in Azure Cognitive Search"""
    try:
        response = requests.post(f"{API_URL}/initialize")
        response.raise_for_status()
        st.success("Index initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing index: {str(e)}")

def upload_files(files):
    """Upload files to server"""
    if not files:
        return
    
    try:
        files_data = [("files", file) for file in files]
        response = requests.post(f"{API_URL}/upload", files=files_data)
        response.raise_for_status()
        result = response.json()
        st.success(f"Successfully processed chunks: {result['processed_chunks']}")
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")

def query_rag(question):
    """Send query to RAG system"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def main():
    # Title
    st.title("🤖 RAG Assistant")
    
    # Sidebar for file upload and initialization
    with st.sidebar:
        st.header("Settings")
        
        # Initialize button
        if st.button("Initialize Index"):
            initialize_index()
        
        # File upload
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["txt", "pdf", "doc", "docx"]
        )
        
        if uploaded_files:
            if st.button("Upload Selected Files"):
                upload_files(uploaded_files)
    
    # Main chat area
    st.header("Ask your question")
    
    # Initialize message history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "debug" in message:
                with st.expander("Processing Details"):
                    st.json(message["debug"])
    
    # Question input field
    if prompt := st.chat_input("Enter your question"):
        # Add user question to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display question
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_rag(prompt)
                if response:
                    st.write(response["answer"])
                    # Save response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "debug": response["debug"]
                    })
                    
                    # Show processing details
                    with st.expander("Processing Details"):
                        st.json(response["debug"])

if __name__ == "__main__":
    main()