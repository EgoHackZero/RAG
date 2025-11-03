from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import shutil
from rag_engine import RAGEngine

app = FastAPI(title="RAG API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

class Query(BaseModel):
    question: str

@app.post("/initialize")
async def initialize_index():
    """Initialize search index"""
    try:
        rag_engine.create_search_index()
        return {"message": "Search index initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload documents for indexing"""
    try:
        upload_dir = "temp_uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)

        uploaded_docs = rag_engine.load_documents(upload_dir)

        shutil.rmtree(upload_dir)

        return {
            "message": f"Successfully processed {len(uploaded_docs)} documents",
            "processed_chunks": len(uploaded_docs)
        }
    except Exception as e:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query) -> Dict[str, Any]:
    """Process user query"""
    try:
        result = rag_engine.process_query(query.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)