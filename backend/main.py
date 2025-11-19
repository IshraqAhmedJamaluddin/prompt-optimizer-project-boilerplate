"""
Simple Chat Application - Starter Code
A basic chat interface using Gemini API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Simple Chat API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")

genai.configure(api_key=api_key)


# Data Models
class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    response_time_ms: float
    error: Optional[str] = None


# In-memory chat history (simple storage)
chat_history = []


@app.get("/")
async def root():
    return {"message": "Simple Chat API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Simple chat endpoint - sends message to Gemini and returns response
    """
    start_time = time.time()
    
    try:
        # Initialize model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Send message and get response
        response = model.generate_content(message.message)
        
        response_time_ms = (time.time() - start_time) * 1000
        response_text = response.text
        
        # Store in history (simple approach)
        chat_history.append({
            "user": message.message,
            "assistant": response_text,
            "timestamp": time.time()
        })
        
        return ChatResponse(
            response=response_text,
            response_time_ms=round(response_time_ms, 2)
        )
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        return ChatResponse(
            response="",
            response_time_ms=round(response_time_ms, 2),
            error=str(e)
        )


@app.get("/api/history")
async def get_history():
    """Get chat history"""
    return {"history": chat_history[-20:]}  # Return last 20 messages


@app.delete("/api/history")
async def clear_history():
    """Clear chat history"""
    chat_history.clear()
    return {"message": "History cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
