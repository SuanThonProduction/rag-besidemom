from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np
import uvicorn
from typing import List, Optional
import os
import shutil
from pathlib import Path
from fastapi import Form
from openai import OpenAI

app = FastAPI(title="RAG Chatbot API", description="Motherhood and Childcare Assistant")
load_dotenv()
chunks = None
embedder = None
index = None
embeddings = None
client = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str
    retrieved_context: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_count: int
    status: str

def load_pdf_chunks(pdf_path, chunk_size=500):
    """Load PDF and split into chunks"""
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or unreadable")
        
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")

def embed_chunks(chunks, embedder):
    """Create embeddings and FAISS index"""
    try:
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return index, embeddings
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")

def retrieve_context(query, chunks, embedder, index, embeddings, top_k=3):
    """Retrieve relevant context for a query"""
    try:
        query_embedding = embedder.encode([query])
        _, indices = index.search(np.array(query_embedding), top_k)
        return "\n---\n".join([chunks[i] for i in indices[0]])
    except Exception as e:
        raise Exception(f"Error retrieving context: {str(e)}")

def initialize_rag_system(pdf_path):
    """Initialize or reinitialize the RAG system with a PDF"""
    global chunks, embedder, index, embeddings
    
    try:
        print(f"Loading PDF from: {pdf_path}")
        chunks = load_pdf_chunks(pdf_path)
        print(f"Loaded {len(chunks)} chunks from PDF")
        
        if embedder is None:
            print("Loading sentence transformer model...")
            embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        
        print("Creating embeddings and FAISS index...")
        index, embeddings = embed_chunks(chunks, embedder)
        print("RAG system initialized successfully!")
        
        return len(chunks)
    except Exception as e:
        raise Exception(f"Error initializing RAG system: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global client
    
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL')
    )
    
    default_pdf_path = "./FAQ_for_Chatbot.pdf"
    if os.path.exists(default_pdf_path):
        try:
            initialize_rag_system(default_pdf_path)
            print("RAG Chatbot API initialized successfully with default PDF!")
        except Exception as e:
            print(f"Warning: Could not load default PDF: {str(e)}")
            print("You can upload a PDF using the /upload-pdf endpoint")
    else:
        print("No default PDF found. You can upload a PDF using the /upload-pdf endpoint")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for the RAG chatbot"""
    try:
        if not (chunks is not None and embedder is not None and index is not None and embeddings is not None and client is not None):
            raise HTTPException(
                status_code=503, 
                detail="RAG system not initialized. Please upload a PDF first using /upload-pdf endpoint"
            )
        
        retrieved_context = retrieve_context(
            request.message, chunks, embedder, index, embeddings
        )
        
        system_prompt = f"""
You are an AI assistant specialized in motherhood and childcare.
Use the following additional context to help answer the question:
{retrieved_context}

Provide helpful, accurate, and caring responses based on the context provided.
"""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        response = client.chat.completions.create(
            model=os.getenv('MODEL_TYPE'),
            messages=conversation,
            max_tokens=request.max_tokens,
        
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            retrieved_context=retrieved_context
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and reinitialize the RAG system"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF. Please upload a .pdf file."
            )
        
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks_count = initialize_rag_system(str(file_path))
        
        default_path = Path("./FAQ_for_Chatbot.pdf")
        shutil.copy2(file_path, default_path)
        
        return UploadResponse(
            message="PDF uploaded and RAG system reinitialized successfully",
            filename=file.filename,
            chunks_count=chunks_count,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading PDF: {str(e)}"
        )

@app.get("/pdf-status")
async def pdf_status():
    """Get status of current PDF and RAG system"""
    is_initialized = (chunks is not None and embedder is not None and index is not None and embeddings is not None)
    
    default_pdf_exists = os.path.exists("./FAQ_for_Chatbot.pdf")
    
    return {
        "rag_initialized": is_initialized,
        "chunks_count": len(chunks) if chunks else 0,
        "default_pdf_exists": default_pdf_exists,
        "embedder_loaded": embedder is not None,
        "message": "RAG system ready" if is_initialized else "No PDF loaded. Upload a PDF to start."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_ready = (chunks is not None and embedder is not None and index is not None and embeddings is not None and client is not None)
    return {
        "status": "healthy" if is_ready else "initializing",
        "message": "RAG Chatbot API is running" if is_ready else "RAG system initializing or PDF not loaded",
        "ready": is_ready
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to RAG Chatbot API",
        "version": "1.1.0",
        "description": "Motherhood and Childcare Assistant with PDF Upload",
        "endpoints": {
            "chat": "/chat - POST request with message",
            "upload_pdf": "/upload-pdf - POST request to upload new PDF",
            "pdf_status": "/pdf-status - GET current PDF and RAG status",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        },
        "usage": {
            "1": "Upload a PDF using /upload-pdf endpoint",
            "2": "Check status using /pdf-status endpoint", 
            "3": "Start chatting using /chat endpoint"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)