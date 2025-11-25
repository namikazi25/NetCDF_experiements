from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from typing import List, Optional
import uvicorn
from nc_processor import extract_metadata
from llm_service import chat_with_context
from memory_service import save_memory_entry

app = FastAPI(title="NetCDF LLM Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store metadata in memory for this prototype
# In a real app, use a database
metadata_store = {}

class ChatRequest(BaseModel):
    query: str
    file_id: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract metadata
        metadata = extract_metadata(file_path)
        metadata_store[file.filename] = metadata
        
        return {"filename": file.filename, "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.file_id not in metadata_store:
        raise HTTPException(status_code=404, detail="File not found or not processed")
    
    metadata = metadata_store[request.file_id]
    response = chat_with_context(request.query, metadata)
    return {"response": response}

# 1. Define the Data Model for Feedback
class FeedbackPayload(BaseModel):
    query: str
    code: str
    plan: str

# 2. Add the NEW Feedback Endpoint
@app.post("/feedback/save")
async def save_feedback(payload: FeedbackPayload):
    """
    Called when user clicks 'Thumbs Up' in the UI.
    Saves the successful code recipe to the knowledge base.
    """
    try:
        # This writes to code_memory.json
        save_memory_entry(payload.query, payload.code, payload.plan)
        return {"status": "success", "message": "Memory saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
