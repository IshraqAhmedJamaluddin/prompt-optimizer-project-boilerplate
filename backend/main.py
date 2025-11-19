"""
Prompt Helper Chat - Starter Code
A chat-based prompt optimization assistant with a prompt critic character.
Helps users optimize their prompts using techniques from the Prompt Engineering Foundations course.

TODO: Students will implement the Prompt Critic system prompt and character behavior.
"""

import json
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await init_db()
    yield
    # Shutdown (if needed in the future)


app = FastAPI(title="Prompt Helper Chat API", version="2.0.0", lifespan=lifespan)


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
# NOTE: When using code blocks (```) in triple-quoted strings, do NOT escape backticks.
# Use ``` directly, not \`\`\` (which causes SyntaxWarning: invalid escape sequence).
PROMPT_CRITIC_SYSTEM_PROMPT = """You are Prompt Critic, an expert AI assistant specialized in helping users optimize their prompts using proven prompt engineering techniques from the Prompt Engineering Foundations course.

## Your Identity
- Name: Prompt Critic
- Role: Expert Prompt Engineering Consultant
- Purpose: Help users create better, more effective prompts
- Personality: Knowledgeable, constructive, encouraging, detail-oriented, educational

## Your Expertise
You are an expert in prompt engineering and understand the fundamentals:

### Core Concepts (Module 1)
- Understanding tokens: the building blocks of LLM communication
- Context windows: how much information an LLM can remember at once
- Temperature control: balancing creativity and precision
- LLM limitations: what AI can and cannot do reliably
- When prompting is the right solution vs. when traditional tools are better

## Communication Style
- Use clear, educational, and constructive language
- Be specific in your feedback - point out exact issues and improvements
- Reference specific prompt engineering techniques when making suggestions
- Provide examples of improved prompts when helpful
- Ask clarifying questions to understand the user's goal
- Celebrate improvements and progress
- Use a friendly but professional tone
- Break down complex suggestions into actionable steps

## How You Help Users
1. Analyze prompts and identify areas for improvement
2. Suggest specific prompt engineering techniques that would help
3. Provide before/after examples of improved prompts
4. Explain the reasoning behind suggestions
5. Help users understand how to use prompt engineering effectively

## Response Format
- Keep responses clear and organized
- Use markdown formatting when helpful (headings, lists, code blocks)
- Length: As needed to be helpful and complete (typically 3-8 sentences for simple queries, longer for detailed analysis)
- Use emojis sparingly and appropriately (💡✨📝✅)

Remember: You're Prompt Critic, here to help users master prompt engineering through constructive feedback and educational guidance."""

# In-memory storage (in production, use a database)
conversation_sessions: Dict[str, List[Dict]] = {}
prompt_versions: Dict[str, List[Dict]] = defaultdict(list)
feedback_evaluations: List[Dict] = []

# SQLite database path
DB_PATH = "prompt_library.db"

# Rate limiting tracking
rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# TODO: Feature flags for Module 1 - Setup & Foundation
# Students should activate these features as they progress through Module 1
# Change `False` to `True` to activate each feature
ENABLE_CONVERSATION_HISTORY = False  # TODO: Lesson 1.3 - Activate conversation history to maintain context across messages
ENABLE_TOKEN_COUNTING = (
    False  # TODO: Lesson 1.4, 1.7 - Activate token counting and context window tracking
)
ENABLE_ADDITIONAL_PROVIDERS = (
    False  # TODO: Lesson 1.7 - Activate DeepSeek and OpenRouter support
)

# TODO: Feature flags for Module 2 - Frameworks & Best Practices
ENABLE_PROMPT_VERSION_TRACKING = False  # TODO: Lesson 2.5 - Activate iterative refinement and prompt version tracking

# TODO: Feature flags for Module 3 - Advanced Techniques
ENABLE_JSON_OUTPUT = False  # TODO: Lesson 3.2 - Activate structured JSON output option
ENABLE_TEMPERATURE_CONTROL = (
    False  # TODO: Lesson 3.4 - Activate temperature control for feedback style
)
ENABLE_PROMPT_CHAINING = (
    False  # TODO: Lesson 3.6 - Activate multi-step prompt chaining workflow
)
ENABLE_CONTEXT_WINDOW_MANAGEMENT = (
    False  # TODO: Lesson 3.8 - Activate context window management with summarization
)

# TODO: Feature flags for Module 4 - Business Applications & Optimization
# Change `False` to `True` to activate each feature
ENABLE_DEFENSIVE_PROMPTING = False  # TODO: Lesson 4.2 - Activate enhanced defensive prompting (basic sanitization is always active)
ENABLE_CONVERSATION_EXPORT = (
    False  # TODO: Lesson 4.7 - Activate conversation export functionality
)
ENABLE_PROMPT_LIBRARY = False  # TODO: Lesson 4.6 - Activate prompt library for saving and organizing prompts
ENABLE_FEEDBACK_EVALUATION = (
    False  # TODO: Lesson 4.1 - Activate feedback evaluation tracking
)


# Enums
class LLMProvider(str, Enum):
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class ReasoningStrategy(str, Enum):
    STEP_BY_STEP = "step_by_step"
    REACT = "react"
    DIRECT = "direct"


# Data Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    output_format: Optional[OutputFormat] = OutputFormat.TEXT
    reasoning_strategy: Optional[ReasoningStrategy] = ReasoningStrategy.DIRECT
    session_id: Optional[str] = None
    enable_chaining: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    character_name: str = "Prompt Critic"
    tokens_used: Optional[int] = None
    tokens_remaining: Optional[int] = None
    quality_score: Optional[Dict[str, float]] = None
    prompt_version: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptVersion(BaseModel):
    version: int
    prompt: str
    timestamp: str
    quality_score: Optional[Dict[str, float]] = None


class PromptLibraryEntry(BaseModel):
    id: str
    prompt: str
    optimized_prompt: str
    tags: List[str]
    category: Optional[str] = None
    timestamp: str
    quality_score: Optional[Dict[str, float]] = None


class FeedbackEvaluation(BaseModel):
    suggestion_id: Optional[str] = None
    helpful: bool
    feedback: Optional[str] = None
    timestamp: str


# Utility Functions


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4


def sanitize_input(text: str) -> str:
    """Defensive prompting: sanitize user input to prevent injection attacks"""
    # Remove potential system prompt injection patterns
    patterns = [
        r"(?i)ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"(?i)forget\s+(everything|all|previous)",
        r"(?i)system\s*:\s*",
        r"(?i)assistant\s*:\s*",
        r"(?i)you\s+are\s+now",
    ]
    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, "", sanitized)
    return sanitized.strip()


def check_rate_limit(identifier: str) -> bool:
    """Check if request is within rate limit"""
    now = time.time()
    # Clean old entries
    rate_limit_tracker[identifier] = [
        t for t in rate_limit_tracker[identifier] if now - t < RATE_LIMIT_WINDOW
    ]
    # Check limit
    if len(rate_limit_tracker[identifier]) >= RATE_LIMIT_REQUESTS:
        return False
    rate_limit_tracker[identifier].append(now)
    return True


def calculate_quality_score(prompt: str) -> Dict[str, float]:
    """Calculate prompt quality scores (clarity, completeness, effectiveness)"""
    # Simple heuristic-based scoring (students can enhance this via prompts)
    clarity = min(1.0, len(prompt.split()) / 50)  # More words = clearer
    completeness = min(
        1.0, (prompt.count("?") + prompt.count(":")) / 5
    )  # Questions/instructions
    effectiveness = min(
        1.0, len([w for w in prompt.split() if len(w) > 5]) / 20
    )  # Specific terms

    return {
        "clarity": round(clarity, 2),
        "completeness": round(completeness, 2),
        "effectiveness": round(effectiveness, 2),
        "overall": round((clarity + completeness + effectiveness) / 3, 2),
    }


def summarize_conversation(messages: List[Dict], max_tokens: int = 1000) -> str:
    """Summarize old messages when conversation gets too long"""
    total_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in messages)
    if total_tokens <= max_tokens:
        return ""

    # Simple summarization: take first and last messages, summarize middle
    if len(messages) > 4:
        summary = f"[Previous conversation: {len(messages)-2} messages about prompt optimization]"
        return summary
    return ""


# Database Functions
async def init_db():
    """Initialize SQLite database for prompt library"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_library (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                optimized_prompt TEXT NOT NULL,
                tags TEXT,  -- JSON array stored as text
                category TEXT,
                timestamp TEXT NOT NULL,
                quality_score TEXT  -- JSON object stored as text
            )
        """
        )
        await db.commit()


async def get_db():
    """Get database connection"""
    return await aiosqlite.connect(DB_PATH)


async def call_gemini(
    messages: List[Dict],
    system_prompt: str,
    temperature: float = 0.7,
    user_message: str = "",
) -> Tuple[str, int]:
    """Call Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set it in your .env file.",
        )

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat_history = []

        if system_prompt and system_prompt.strip():
            chat_history.append({"role": "user", "parts": [f"System: {system_prompt}"]})
            chat_history.append(
                {
                    "role": "model",
                    "parts": ["Understood. I'll follow these instructions."],
                }
            )

        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(user_message)
        tokens_used = estimate_tokens(user_message) + estimate_tokens(response.text)
        return response.text, tokens_used
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


async def call_deepseek(
    messages: List[Dict],
    system_prompt: str,
    temperature: float = 0.7,
    user_message: str = "",
) -> Tuple[str, int]:
    """Call DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY not configured. Please set it in your .env file.",
        )

    try:
        async with httpx.AsyncClient() as client:
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                if msg["role"] != "system":
                    api_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )
            api_messages.append({"role": "user", "content": user_message})

            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": api_messages,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get(
                "total_tokens", estimate_tokens(text)
            )
            return text, tokens_used
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {str(e)}")


async def call_openrouter(
    messages: List[Dict],
    system_prompt: str,
    temperature: float = 0.7,
    user_message: str = "",
    model: str = "openai/gpt-3.5-turbo",
) -> Tuple[str, int]:
    """Call OpenRouter API"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not configured. Please set it in your .env file.",
        )

    try:
        async with httpx.AsyncClient() as client:
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            for msg in messages:
                if msg["role"] != "system":
                    api_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )
            api_messages.append({"role": "user", "content": user_message})

            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Prompt Optimizer",
                },
                json={
                    "model": model,
                    "messages": api_messages,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get(
                "total_tokens", estimate_tokens(text)
            )
            return text, tokens_used
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")


def apply_reasoning_strategy(
    system_prompt: str, reasoning_strategy: ReasoningStrategy, user_message: str
) -> str:
    """Apply reasoning strategy to system prompt"""
    if reasoning_strategy == ReasoningStrategy.STEP_BY_STEP:
        return f"{system_prompt}\n\nWhen analyzing prompts, always explain your reasoning step-by-step. Break down your analysis into clear steps: 1) Understanding the prompt, 2) Identifying issues, 3) Suggesting improvements, 4) Providing examples."
    elif reasoning_strategy == ReasoningStrategy.REACT:
        return f"{system_prompt}\n\nUse the ReAct (Reasoning + Acting) approach: Think through the problem, analyze the prompt, then provide actionable suggestions. Format your response as: Thought: [your analysis], Action: [your suggestion], Observation: [expected outcome]."
    else:
        return system_prompt


async def prompt_chaining_workflow(
    user_message: str,
    messages: List[Dict],
    system_prompt: str,
    temperature: float,
) -> Tuple[str, Dict]:
    """Multi-step prompt optimization workflow"""
    # Step 1: Analyze
    analyze_prompt = f"{system_prompt}\n\nAnalyze this prompt and identify areas for improvement. Be specific about what's missing or unclear."
    analyze_msg = [{"role": "user", "content": user_message}]
    analysis, _ = await get_llm_response(
        analyze_msg, analyze_prompt, LLMProvider.GEMINI, temperature, user_message
    )

    # Step 2: Suggest
    suggest_prompt = f"{system_prompt}\n\nBased on this analysis: {analysis}\n\nProvide specific, actionable suggestions for improvement."
    suggestions, _ = await get_llm_response(
        analyze_msg, suggest_prompt, LLMProvider.GEMINI, temperature, user_message
    )

    # Step 3: Refine
    refine_prompt = f"{system_prompt}\n\nBased on these suggestions: {suggestions}\n\nCreate an optimized version of the original prompt."
    refined, _ = await get_llm_response(
        analyze_msg, refine_prompt, LLMProvider.GEMINI, temperature, user_message
    )

    # Step 4: Final
    final_prompt = f"{system_prompt}\n\nProvide a final, polished version of the optimized prompt with a brief explanation of the improvements."
    final, _ = await get_llm_response(
        analyze_msg, final_prompt, LLMProvider.GEMINI, temperature, user_message
    )

    combined_response = f"## Analysis\n{analysis}\n\n## Suggestions\n{suggestions}\n\n## Optimized Prompt\n{refined}\n\n## Final Version\n{final}"

    return combined_response, {
        "workflow": "chaining",
        "steps": ["analyze", "suggest", "refine", "final"],
        "analysis": analysis,
        "suggestions": suggestions,
        "refined": refined,
        "final": final,
    }


async def get_llm_response(
    messages: List[Dict],
    system_prompt: str,
    provider: LLMProvider,
    temperature: float,
    user_message: str,
) -> Tuple[str, int]:
    """Get response from selected LLM provider"""
    # TODO: Lesson 1.7 - Additional LLM providers (DeepSeek, OpenRouter)
    # Activate by setting ENABLE_ADDITIONAL_PROVIDERS = True
    if provider == LLMProvider.GEMINI:
        return await call_gemini(messages, system_prompt, temperature, user_message)
    elif provider == LLMProvider.DEEPSEEK:
        if not ENABLE_ADDITIONAL_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail="DeepSeek provider not enabled. Set ENABLE_ADDITIONAL_PROVIDERS = True to activate. (Lesson 1.7)",
            )
        return await call_deepseek(messages, system_prompt, temperature, user_message)
    elif provider == LLMProvider.OPENROUTER:
        if not ENABLE_ADDITIONAL_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail="OpenRouter provider not enabled. Set ENABLE_ADDITIONAL_PROVIDERS = True to activate. (Lesson 1.7)",
            )
        return await call_openrouter(messages, system_prompt, temperature, user_message)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


@app.get("/")
async def root():
    return {"message": "Prompt Helper Chat API", "version": "2.0.0"}


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
    Supports multiple LLM providers, temperature control, reasoning strategies, and prompt chaining.
    """
    # Rate limiting
    client_id = request.session_id or "default"
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
        )

    # TODO: Lesson 4.2 - Defensive prompting (basic sanitization always active)
    # Enhanced defensive prompting can be activated by setting ENABLE_DEFENSIVE_PROMPTING = True
    # Sanitize user input only when ENABLE_DEFENSIVE_PROMPTING is True
    if ENABLE_DEFENSIVE_PROMPTING:
        user_message = sanitize_input(request.message.strip())
    else:
        user_message = request.message.strip()
    # Additional defensive measures can be added here when ENABLE_DEFENSIVE_PROMPTING is True

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build conversation context
    messages = []

    # Get or create session
    session_id = request.session_id or "default"
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    # TODO: Lesson 1.3 - Enhanced conversation history
    # Basic history is always active for core functionality
    # Enhanced tracking maintains context across messages
    session_messages = conversation_sessions[session_id]
    if ENABLE_CONVERSATION_HISTORY:
        messages.extend(session_messages[-20:])  # Last 20 messages
    else:
        # Basic: only use provided conversation history
        pass

    # Add provided conversation history (always active)
    if request.conversation_history:
        for msg in request.conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

    # TODO: Lesson 3.8 - Context window management: summarize if too long
    # Activate by setting ENABLE_CONTEXT_WINDOW_MANAGEMENT = True
    if ENABLE_CONTEXT_WINDOW_MANAGEMENT:
        summary = summarize_conversation(messages)
        if summary:
            messages = messages[-5:]  # Keep last 5 messages
            messages.insert(0, {"role": "system", "content": summary})

    # Apply reasoning strategy (always active - this is prompt-based)
    enhanced_system_prompt = apply_reasoning_strategy(
        PROMPT_CRITIC_SYSTEM_PROMPT, request.reasoning_strategy, user_message
    )

    # TODO: Lesson 3.4 - Temperature control for feedback style
    # Activate by setting ENABLE_TEMPERATURE_CONTROL = True
    # If disabled, use default temperature
    if not ENABLE_TEMPERATURE_CONTROL:
        request.temperature = 0.7  # Default temperature

    # TODO: Lesson 2.5 - Track prompt version for iterative refinement
    # Activate by setting ENABLE_PROMPT_VERSION_TRACKING = True
    prompt_version = None
    if ENABLE_PROMPT_VERSION_TRACKING:
        prompt_version = len(prompt_versions.get(session_id, [])) + 1

    try:
        # TODO: Lesson 3.6 - Prompt chaining workflow
        # Activate by setting ENABLE_PROMPT_CHAINING = True
        if request.enable_chaining and ENABLE_PROMPT_CHAINING:
            response_text, metadata = await prompt_chaining_workflow(
                user_message,
                messages,
                enhanced_system_prompt,
                request.temperature,
            )
            tokens_used = estimate_tokens(response_text)
        else:
            # Regular single-step response (always use Gemini)
            response_text, tokens_used = await get_llm_response(
                messages,
                enhanced_system_prompt,
                LLMProvider.GEMINI,
                request.temperature,
                user_message,
            )
            metadata = {}

        # Calculate quality score
        quality_score = calculate_quality_score(user_message)

        # TODO: Lesson 2.5 - Store prompt version for iterative refinement
        if ENABLE_PROMPT_VERSION_TRACKING and prompt_version:
            prompt_versions[session_id].append(
                {
                    "version": prompt_version,
                    "prompt": user_message,
                    "timestamp": datetime.now().isoformat(),
                    "quality_score": quality_score,
                }
            )

        # Update session (always active for basic functionality)
        # TODO: Lesson 1.3 - Enhanced conversation history tracking
        # When ENABLE_CONVERSATION_HISTORY is True, maintains full context across messages
        conversation_sessions[session_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_sessions[session_id].append(
            {"role": "assistant", "content": response_text}
        )

        # TODO: Lesson 3.2 - Handle JSON output format
        # Activate by setting ENABLE_JSON_OUTPUT = True
        if request.output_format == OutputFormat.JSON and ENABLE_JSON_OUTPUT:
            try:
                # Try to parse as JSON, if fails return as text
                json_data = json.loads(response_text)
                response_text = json.dumps(json_data, indent=2)
            except:
                # If not JSON, wrap in JSON structure
                response_text = json.dumps(
                    {
                        "response": response_text,
                        "quality_score": quality_score,
                        "suggestions": [],
                    },
                    indent=2,
                )

        # TODO: Lesson 1.4, 1.7 - Token counting and context window tracking
        # Activate by setting ENABLE_TOKEN_COUNTING = True
        tokens_remaining = None
        if not ENABLE_TOKEN_COUNTING:
            # Don't track tokens if feature is disabled
            tokens_used = None
        else:
            # Estimate remaining tokens (assuming 32k context window)
            max_tokens = 32000
            total_tokens = (
                sum(estimate_tokens(msg.get("content", "")) for msg in messages)
                + tokens_used
            )
            tokens_remaining = max(0, max_tokens - total_tokens)

        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            character_name="Prompt Critic",
            tokens_used=tokens_used if ENABLE_TOKEN_COUNTING else None,
            tokens_remaining=tokens_remaining,
            quality_score=quality_score,
            prompt_version=prompt_version if ENABLE_PROMPT_VERSION_TRACKING else None,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Error handling for API failures and rate limits (always active)
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(
                status_code=429, detail=f"API rate limit exceeded: {error_msg}"
            )
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail=f"API authentication failed. Check your API key: {error_msg}",
            )
        elif "timeout" in error_msg.lower():
            raise HTTPException(status_code=504, detail=f"Request timeout: {error_msg}")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error processing request: {error_msg}"
            )


# Additional Endpoints


@app.get("/api/sessions/{session_id}/versions")
async def get_prompt_versions(session_id: str):
    """
    TODO: Lesson 2.5 - Get all prompt versions for a session
    Activate by setting ENABLE_PROMPT_VERSION_TRACKING = True
    """
    if not ENABLE_PROMPT_VERSION_TRACKING:
        raise HTTPException(
            status_code=403,
            detail="Prompt version tracking not enabled. Set ENABLE_PROMPT_VERSION_TRACKING = True to activate. (Lesson 2.5)",
        )
    versions = prompt_versions.get(session_id, [])
    return {"session_id": session_id, "versions": versions}


@app.get("/api/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    history = conversation_sessions.get(session_id, [])
    return {"session_id": session_id, "history": history}


@app.get("/api/export/{session_id}")
async def export_conversation(
    session_id: str, format: str = Query("markdown", pattern="^(markdown|json)$")
):
    """
    TODO: Lesson 4.7 - Export conversation history as markdown or JSON
    Activate by setting ENABLE_CONVERSATION_EXPORT = True
    """
    if not ENABLE_CONVERSATION_EXPORT:
        raise HTTPException(
            status_code=403,
            detail="Conversation export not enabled. Set ENABLE_CONVERSATION_EXPORT = True to activate. (Lesson 4.7)",
        )
    history = conversation_sessions.get(session_id, [])
    versions = prompt_versions.get(session_id, [])

    if format == "json":
        return {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "conversation": history,
            "prompt_versions": versions,
        }
    else:  # markdown
        md_content = f"# Conversation Export - Session {session_id}\n\n"
        md_content += f"**Exported:** {datetime.now().isoformat()}\n\n"

        md_content += "## Prompt Versions\n\n"
        for v in versions:
            md_content += f"### Version {v['version']}\n\n"
            md_content += f"**Prompt:** {v['prompt']}\n\n"
            md_content += f"**Quality Score:** {v.get('quality_score', {})}\n\n"
            md_content += f"**Timestamp:** {v['timestamp']}\n\n---\n\n"

        md_content += "## Conversation History\n\n"
        for msg in history:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            md_content += f"### {role_emoji} {msg['role'].title()}\n\n"
            md_content += f"{msg['content']}\n\n---\n\n"

        return Response(content=md_content, media_type="text/markdown")


@app.post("/api/library")
async def save_to_library(entry: PromptLibraryEntry):
    """
    TODO: Lesson 4.6 - Save a prompt to the library
    Activate by setting ENABLE_PROMPT_LIBRARY = True
    """
    if not ENABLE_PROMPT_LIBRARY:
        raise HTTPException(
            status_code=403,
            detail="Prompt library not enabled. Set ENABLE_PROMPT_LIBRARY = True to activate. (Lesson 4.6)",
        )

    try:
        # Initialize database if it doesn't exist
        await init_db()

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO prompt_library 
                (id, prompt, optimized_prompt, tags, category, timestamp, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.prompt,
                    entry.optimized_prompt,
                    json.dumps(entry.tags),
                    entry.category,
                    entry.timestamp,
                    json.dumps(entry.quality_score) if entry.quality_score else None,
                ),
            )
            await db.commit()

        return {"status": "saved", "id": entry.id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving to library: {str(e)}"
        )


@app.get("/api/library")
async def get_library(category: Optional[str] = None, tag: Optional[str] = None):
    """
    TODO: Lesson 4.6 - Get prompts from library with optional filtering
    Activate by setting ENABLE_PROMPT_LIBRARY = True
    """
    if not ENABLE_PROMPT_LIBRARY:
        raise HTTPException(
            status_code=403,
            detail="Prompt library not enabled. Set ENABLE_PROMPT_LIBRARY = True to activate. (Lesson 4.6)",
        )

    try:
        await init_db()

        query = "SELECT * FROM prompt_library WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY timestamp DESC"

        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                entries = []
                for row in rows:
                    entry_dict = {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "optimized_prompt": row["optimized_prompt"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "category": row["category"],
                        "timestamp": row["timestamp"],
                        "quality_score": (
                            json.loads(row["quality_score"])
                            if row["quality_score"]
                            else None
                        ),
                    }
                    entries.append(entry_dict)

        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading library: {str(e)}")


@app.get("/api/library/{entry_id}")
async def get_library_entry(entry_id: str):
    """
    TODO: Lesson 4.6 - Get a specific library entry
    Activate by setting ENABLE_PROMPT_LIBRARY = True
    """
    if not ENABLE_PROMPT_LIBRARY:
        raise HTTPException(
            status_code=403,
            detail="Prompt library not enabled. Set ENABLE_PROMPT_LIBRARY = True to activate. (Lesson 4.6)",
        )

    try:
        await init_db()

        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM prompt_library WHERE id = ?", (entry_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Entry not found")

                return {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "optimized_prompt": row["optimized_prompt"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "category": row["category"],
                    "timestamp": row["timestamp"],
                    "quality_score": (
                        json.loads(row["quality_score"])
                        if row["quality_score"]
                        else None
                    ),
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading entry: {str(e)}")


@app.delete("/api/library/{entry_id}")
async def delete_library_entry(entry_id: str):
    """
    TODO: Lesson 4.6 - Delete a library entry
    Activate by setting ENABLE_PROMPT_LIBRARY = True
    """
    if not ENABLE_PROMPT_LIBRARY:
        raise HTTPException(
            status_code=403,
            detail="Prompt library not enabled. Set ENABLE_PROMPT_LIBRARY = True to activate. (Lesson 4.6)",
        )

    try:
        await init_db()

        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "DELETE FROM prompt_library WHERE id = ?", (entry_id,)
            )
            await db.commit()
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Entry not found")

        return {"status": "deleted", "id": entry_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting entry: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackEvaluation):
    """
    TODO: Lesson 4.1 - Submit feedback evaluation for suggestions
    Activate by setting ENABLE_FEEDBACK_EVALUATION = True
    """
    if not ENABLE_FEEDBACK_EVALUATION:
        raise HTTPException(
            status_code=403,
            detail="Feedback evaluation not enabled. Set ENABLE_FEEDBACK_EVALUATION = True to activate. (Lesson 4.1)",
        )
    feedback_evaluations.append(
        {
            "suggestion_id": feedback.suggestion_id,
            "helpful": feedback.helpful,
            "feedback": feedback.feedback,
            "timestamp": feedback.timestamp,
        }
    )
    return {"status": "received", "total_evaluations": len(feedback_evaluations)}


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """
    TODO: Lesson 4.1 - Get feedback evaluation statistics
    Activate by setting ENABLE_FEEDBACK_EVALUATION = True
    """
    if not ENABLE_FEEDBACK_EVALUATION:
        raise HTTPException(
            status_code=403,
            detail="Feedback evaluation not enabled. Set ENABLE_FEEDBACK_EVALUATION = True to activate. (Lesson 4.1)",
        )
    if not feedback_evaluations:
        return {"total": 0, "helpful": 0, "not_helpful": 0, "helpful_rate": 0.0}

    total = len(feedback_evaluations)
    helpful = sum(1 for f in feedback_evaluations if f.get("helpful"))
    not_helpful = total - helpful
    helpful_rate = helpful / total if total > 0 else 0.0

    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": not_helpful,
        "helpful_rate": round(helpful_rate, 2),
    }


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    if session_id in prompt_versions:
        del prompt_versions[session_id]
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
