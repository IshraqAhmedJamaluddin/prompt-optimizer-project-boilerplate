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
PROMPT_CRITIC_SYSTEM_PROMPT = ""


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
        "personality_traits": [
            "knowledgeable",
            "constructive",
            "encouraging",
            "detail-oriented",
            "educational",
        ],
        "tone_of_voice": "clear, educational, constructive, friendly but professional",
        "system_prompt": PROMPT_CRITIC_SYSTEM_PROMPT,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with Prompt Critic persona.
    Uses system prompt and conversation history for context-aware responses.
    """
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build conversation context
    messages = [{"role": "system", "content": PROMPT_CRITIC_SYSTEM_PROMPT}]

    # Add conversation history if provided
    if request.conversation_history:
        for msg in request.conversation_history[-10:]:  # Last 10 messages for context
            messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Get response from LLM with system prompt and conversation history
    response_text = await get_prompt_critic_response(
        user_message, messages, PROMPT_CRITIC_SYSTEM_PROMPT
    )

    return ChatResponse(
        response=response_text,
        timestamp=datetime.now().isoformat(),
        character_name="Prompt Critic",
    )


async def get_prompt_critic_response(
    user_message: str, messages: List[dict], system_prompt: str = None
) -> str:
    """
    Get response from Gemini API with Prompt Critic system prompt.
    Uses the comprehensive system prompt to maintain character consistency.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set it in your .env file. Get a free key from https://aistudio.google.com/app/apikey",
        )

    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Build conversation history for Gemini
        chat_history = []

        # Handle system prompt - if provided and not empty, add it as first message
        prompt_to_use = system_prompt
        if not prompt_to_use:
            # Extract system prompt from messages if not provided directly
            for msg in messages:
                if msg["role"] == "system":
                    prompt_to_use = msg["content"]
                    break

        # Add system prompt to chat history if it exists
        if prompt_to_use and prompt_to_use.strip():
            # Add system prompt as first user message with "System: " prefix
            chat_history.append({"role": "user", "parts": [f"System: {prompt_to_use}"]})
            # Add model acknowledgment
            chat_history.append(
                {
                    "role": "model",
                    "parts": [
                        "Understood. I'll follow these instructions and help users optimize their prompts using prompt engineering techniques."
                    ],
                }
            )

        # Convert conversation history (handle system messages separately - they're already processed above)
        for msg in messages:
            if msg["role"] == "system":
                # System messages are handled above via system_prompt parameter
                # Skip them here to avoid duplication
                continue
            elif msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        # Start chat session with history (excluding current user message)
        chat = model.start_chat(history=chat_history)

        # Send current user message
        response = chat.send_message(user_message)
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "404" in error_msg:
            try:
                available_models = [m.name for m in genai.list_models()]
                error_msg += f"\n\nAvailable models: {', '.join(available_models[:10])}"
            except:
                pass
        raise HTTPException(
            status_code=500, detail=f"Error calling Gemini API: {error_msg}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
