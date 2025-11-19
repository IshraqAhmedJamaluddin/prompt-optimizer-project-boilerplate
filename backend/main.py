"""
Prompt Helper Chat - Starter Code
A chat-based prompt optimization assistant with a prompt critic character.
Helps users optimize their prompts using techniques from the Prompt Engineering Foundations course.

TODO: Students will implement the Prompt Critic system prompt and character behavior.
"""

import os
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Prompt Helper Chat API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Create a comprehensive system prompt for the Prompt Critic character
# The system prompt should:
# 1. Define the character's identity (name, role, personality)
# 2. Describe their expertise in prompt engineering techniques from the course
# 3. Explain their communication style and how they help users
# 4. Include guidelines for providing constructive feedback
# 5. Reference specific course modules and techniques
# 6. Set behavioral boundaries and response formatting
# 
# The prompt should be 500+ tokens and demonstrate advanced prompt engineering techniques.
# Look at the solutions branch for a complete example.
PROMPT_CRITIC_SYSTEM_PROMPT = """TODO: Implement the Prompt Critic system prompt here.
This should be a comprehensive prompt that defines the character's personality, expertise, and behavior.
Students will complete this as part of the course exercise."""

# Data Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    character_name: str = "Prompt Critic"


@app.get("/")
async def root():
    return {"message": "Prompt Helper Chat API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/character")
async def get_character_info():
    """Get information about the Prompt Critic character"""
    return {
        "name": "Prompt Critic",
        "role": "Expert Prompt Engineering Consultant",
        "avatar": "💡",
        "personality_traits": ["knowledgeable", "constructive", "encouraging", "detail-oriented", "educational"],
        "tone_of_voice": "clear, educational, constructive, friendly but professional",
        "system_prompt": PROMPT_CRITIC_SYSTEM_PROMPT
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with Prompt Critic persona.
    Uses system prompt and conversation history for context-aware responses.
    
    TODO: Ensure the system prompt is properly used in the LLM call.
    """
    user_message = request.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Build conversation context
    messages = [
        {"role": "system", "content": PROMPT_CRITIC_SYSTEM_PROMPT}
    ]
    
    # Add conversation history if provided
    if request.conversation_history:
        for msg in request.conversation_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # TODO: Implement the get_prompt_critic_response function
    # This function should:
    # 1. Use the Gemini API to get a response
    # 2. Include the system prompt in the conversation
    # 3. Handle conversation history properly (Gemini uses a specific format)
    # 4. Return the response text
    # 
    # See the solutions branch for a complete implementation.
    response_text = await get_prompt_critic_response(user_message, messages, PROMPT_CRITIC_SYSTEM_PROMPT)
    
    return ChatResponse(
        response=response_text,
        timestamp=datetime.now().isoformat(),
        character_name="Prompt Critic"
    )


async def get_prompt_critic_response(user_message: str, messages: List[dict], system_prompt: str = None) -> str:
    """
    Get response from Gemini API with Prompt Critic system prompt.
    Uses the comprehensive system prompt to maintain character consistency.
    
    TODO: Implement this function to:
    1. Check if GEMINI_API_KEY is configured
    2. Initialize the Gemini model (use 'gemini-2.5-flash')
    3. Convert the messages list to Gemini's chat history format
    4. Handle the system prompt appropriately (Gemini doesn't have a separate system role)
    5. Use start_chat() with history and send_message() for the current message
    6. Return the response text
    7. Handle errors appropriately
    
    Hints:
    - Gemini uses {"role": "user", "parts": [...]} and {"role": "model", "parts": [...]} format
    - The system prompt should be added as the first user message with "System: " prefix
    - Use model.start_chat(history=chat_history) to create a chat session
    - Use chat.send_message(user_message) to send the current message
    
    See the solutions branch for a complete implementation.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set it in your .env file. Get a free key from https://aistudio.google.com/app/apikey"
        )
    
    # TODO: Implement the Gemini API call here
    # Replace this with actual implementation
    raise HTTPException(
        status_code=501,
        detail="TODO: Implement get_prompt_critic_response function. See the code comments for guidance."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
