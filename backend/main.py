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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

app = FastAPI(title="Prompt Helper Chat API", version="2.0.0")

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

# In-memory storage (in production, use a database)
conversation_sessions: Dict[str, List[Dict]] = {}
prompt_versions: Dict[str, List[Dict]] = defaultdict(list)
prompt_library: Dict[str, Dict] = {}
feedback_evaluations: List[Dict] = []

# Rate limiting tracking
rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


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
    provider: Optional[LLMProvider] = LLMProvider.GEMINI
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
    provider: LLMProvider,
    temperature: float,
) -> Tuple[str, Dict]:
    """Multi-step prompt optimization workflow"""
    # Step 1: Analyze
    analyze_prompt = f"{system_prompt}\n\nAnalyze this prompt and identify areas for improvement. Be specific about what's missing or unclear."
    analyze_msg = [{"role": "user", "content": user_message}]
    analysis, _ = await get_llm_response(
        analyze_msg, analyze_prompt, provider, temperature, user_message
    )

    # Step 2: Suggest
    suggest_prompt = f"{system_prompt}\n\nBased on this analysis: {analysis}\n\nProvide specific, actionable suggestions for improvement."
    suggestions, _ = await get_llm_response(
        analyze_msg, suggest_prompt, provider, temperature, user_message
    )

    # Step 3: Refine
    refine_prompt = f"{system_prompt}\n\nBased on these suggestions: {suggestions}\n\nCreate an optimized version of the original prompt."
    refined, _ = await get_llm_response(
        analyze_msg, refine_prompt, provider, temperature, user_message
    )

    # Step 4: Final
    final_prompt = f"{system_prompt}\n\nProvide a final, polished version of the optimized prompt with a brief explanation of the improvements."
    final, _ = await get_llm_response(
        analyze_msg, final_prompt, provider, temperature, user_message
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
    if provider == LLMProvider.GEMINI:
        return await call_gemini(messages, system_prompt, temperature, user_message)
    elif provider == LLMProvider.DEEPSEEK:
        return await call_deepseek(messages, system_prompt, temperature, user_message)
    elif provider == LLMProvider.OPENROUTER:
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

    # Sanitize input (defensive prompting)
    user_message = sanitize_input(request.message.strip())

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build conversation context
    messages = []

    # Get or create session
    session_id = request.session_id or "default"
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    session_messages = conversation_sessions[session_id]
    messages.extend(session_messages[-20:])  # Last 20 messages

    # Add provided conversation history
    if request.conversation_history:
        for msg in request.conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

    # Context window management: summarize if too long
    summary = summarize_conversation(messages)
    if summary:
        messages = messages[-5:]  # Keep last 5 messages
        messages.insert(0, {"role": "system", "content": summary})

    # Apply reasoning strategy
    enhanced_system_prompt = apply_reasoning_strategy(
        PROMPT_CRITIC_SYSTEM_PROMPT, request.reasoning_strategy, user_message
    )

    # Track prompt version
    prompt_version = len(prompt_versions.get(session_id, [])) + 1

    try:
        # Prompt chaining workflow
        if request.enable_chaining:
            response_text, metadata = await prompt_chaining_workflow(
                user_message,
                messages,
                enhanced_system_prompt,
                request.provider,
                request.temperature,
            )
            tokens_used = estimate_tokens(response_text)
        else:
            # Regular single-step response
            response_text, tokens_used = await get_llm_response(
                messages,
                enhanced_system_prompt,
                request.provider,
                request.temperature,
                user_message,
            )
            metadata = {}

        # Calculate quality score
        quality_score = calculate_quality_score(user_message)

        # Store prompt version
        prompt_versions[session_id].append(
            {
                "version": prompt_version,
                "prompt": user_message,
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
            }
        )

        # Update session
        conversation_sessions[session_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_sessions[session_id].append(
            {"role": "assistant", "content": response_text}
        )

        # Handle JSON output format
        if request.output_format == OutputFormat.JSON:
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
            tokens_used=tokens_used,
            tokens_remaining=tokens_remaining,
            quality_score=quality_score,
            prompt_version=prompt_version,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Enhanced error handling
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
    """Get all prompt versions for a session"""
    versions = prompt_versions.get(session_id, [])
    return {"session_id": session_id, "versions": versions}


@app.get("/api/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    history = conversation_sessions.get(session_id, [])
    return {"session_id": session_id, "history": history}


@app.post("/api/meta-prompt")
async def meta_prompt_optimize(
    current_prompt: str = Query(..., description="Current system prompt to optimize"),
    optimization_goal: str = Query(
        "improve clarity and effectiveness", description="Goal for optimization"
    ),
):
    """
    Meta-prompting endpoint: Optimize Prompt Critic's own system prompt.
    Students can use this to iteratively improve the system prompt.
    """
    if not current_prompt:
        current_prompt = PROMPT_CRITIC_SYSTEM_PROMPT

    meta_prompt = f"""You are an expert at optimizing system prompts for AI assistants.

Current system prompt:
{current_prompt}

Optimization goal: {optimization_goal}

Analyze the current prompt and provide an improved version that:
1. Maintains the core identity and purpose
2. Addresses the optimization goal
3. Uses advanced prompt engineering techniques
4. Is clear, structured, and effective

Provide the optimized prompt with a brief explanation of improvements."""

    try:
        # Use Gemini for meta-prompting (or allow provider selection)
        messages = [{"role": "user", "content": meta_prompt}]
        optimized, _ = await call_gemini(messages, "", 0.7, meta_prompt)
        return {
            "original": current_prompt,
            "optimized": optimized,
            "goal": optimization_goal,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meta-prompting error: {str(e)}")


@app.get("/api/export/{session_id}")
async def export_conversation(
    session_id: str, format: str = Query("markdown", regex="^(markdown|json)$")
):
    """Export conversation history as markdown or JSON"""
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
    """Save a prompt to the library"""
    prompt_library[entry.id] = {
        "id": entry.id,
        "prompt": entry.prompt,
        "optimized_prompt": entry.optimized_prompt,
        "tags": entry.tags,
        "category": entry.category,
        "timestamp": entry.timestamp,
        "quality_score": entry.quality_score,
    }
    return {"status": "saved", "id": entry.id}


@app.get("/api/library")
async def get_library(category: Optional[str] = None, tag: Optional[str] = None):
    """Get prompts from library with optional filtering"""
    entries = list(prompt_library.values())

    if category:
        entries = [e for e in entries if e.get("category") == category]
    if tag:
        entries = [e for e in entries if tag in e.get("tags", [])]

    return {"entries": entries, "count": len(entries)}


@app.get("/api/library/{entry_id}")
async def get_library_entry(entry_id: str):
    """Get a specific library entry"""
    if entry_id not in prompt_library:
        raise HTTPException(status_code=404, detail="Entry not found")
    return prompt_library[entry_id]


@app.delete("/api/library/{entry_id}")
async def delete_library_entry(entry_id: str):
    """Delete a library entry"""
    if entry_id not in prompt_library:
        raise HTTPException(status_code=404, detail="Entry not found")
    del prompt_library[entry_id]
    return {"status": "deleted", "id": entry_id}


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackEvaluation):
    """Submit feedback evaluation for suggestions"""
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
    """Get feedback evaluation statistics"""
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
