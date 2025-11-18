"""
Prompt Optimizer Tool - FastAPI Backend
A web application for testing, comparing, and optimizing prompts across multiple LLMs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import uuid

app = FastAPI(title="Prompt Optimizer Tool API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data Models
class Prompt(BaseModel):
    id: Optional[str] = None
    title: str
    content: str
    category: Optional[str] = None  # "writing", "analysis", "coding", etc.
    tags: List[str] = []
    version: Optional[int] = 1
    parent_id: Optional[str] = None  # For tracking iterations
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class LLMProvider(BaseModel):
    id: str
    name: str  # "Claude", "ChatGPT", "Gemini", "DeepSeek"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7


class PromptTest(BaseModel):
    id: Optional[str] = None
    prompt_id: str
    provider_id: str
    input_text: str
    output_text: Optional[str] = None
    temperature: float = 0.7
    response_time_ms: Optional[float] = None
    token_count: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class PromptComparison(BaseModel):
    id: Optional[str] = None
    prompt_id: str
    test_ids: List[str]  # Multiple test results to compare
    created_at: Optional[str] = None


class QualityScore(BaseModel):
    test_id: str
    clarity_score: float  # 0-1
    completeness_score: float  # 0-1
    effectiveness_score: float  # 0-1
    overall_score: float  # 0-1
    notes: Optional[str] = None


class HallucinationCheck(BaseModel):
    test_id: str
    detected: bool
    confidence: float  # 0-1
    details: Optional[str] = None


class ABTest(BaseModel):
    id: Optional[str] = None
    name: str
    prompt_variations: List[str]  # List of prompt IDs
    test_cases: List[str]  # List of input texts
    provider_id: str
    created_at: Optional[str] = None


# In-memory storage (replace with database in production)
prompts_db: Dict[str, Prompt] = {}
providers_db: Dict[str, LLMProvider] = {}
tests_db: Dict[str, PromptTest] = {}
comparisons_db: Dict[str, PromptComparison] = {}
quality_scores_db: Dict[str, QualityScore] = {}
hallucination_checks_db: Dict[str, HallucinationCheck] = {}
ab_tests_db: Dict[str, ABTest] = {}


@app.get("/")
async def root():
    return {"message": "Prompt Optimizer Tool API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


# Prompt Management Endpoints
@app.get("/api/prompts", response_model=List[Prompt])
async def get_prompts(category: Optional[str] = None):
    """Get all prompts, optionally filtered by category"""
    prompts = list(prompts_db.values())
    if category:
        prompts = [p for p in prompts if p.category == category]
    return prompts


@app.get("/api/prompts/{prompt_id}", response_model=Prompt)
async def get_prompt(prompt_id: str):
    """Get a specific prompt by ID"""
    if prompt_id not in prompts_db:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompts_db[prompt_id]


@app.post("/api/prompts", response_model=Prompt)
async def create_prompt(prompt: Prompt):
    """Create a new prompt"""
    prompt_id = str(uuid.uuid4())
    prompt.id = prompt_id
    prompt.created_at = datetime.now().isoformat()
    prompt.updated_at = datetime.now().isoformat()
    if not prompt.version:
        prompt.version = 1
    prompts_db[prompt_id] = prompt
    return prompt


@app.put("/api/prompts/{prompt_id}", response_model=Prompt)
async def update_prompt(prompt_id: str, prompt: Prompt):
    """Update an existing prompt"""
    if prompt_id not in prompts_db:
        raise HTTPException(status_code=404, detail="Prompt not found")
    old_prompt = prompts_db[prompt_id]
    prompt.id = prompt_id
    prompt.version = (old_prompt.version or 1) + 1
    prompt.updated_at = datetime.now().isoformat()
    if not prompt.parent_id:
        prompt.parent_id = prompt_id
    prompts_db[prompt_id] = prompt
    return prompt


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt"""
    if prompt_id not in prompts_db:
        raise HTTPException(status_code=404, detail="Prompt not found")
    del prompts_db[prompt_id]
    return {"message": "Prompt deleted successfully"}


# LLM Provider Management
@app.get("/api/providers", response_model=List[LLMProvider])
async def get_providers():
    """Get all LLM providers"""
    return list(providers_db.values())


@app.post("/api/providers", response_model=LLMProvider)
async def create_provider(provider: LLMProvider):
    """Add a new LLM provider"""
    if provider.id in providers_db:
        raise HTTPException(status_code=400, detail="Provider ID already exists")
    providers_db[provider.id] = provider
    return provider


# Testing Endpoints
@app.post("/api/tests", response_model=PromptTest)
async def create_test(test: PromptTest):
    """Create a new prompt test"""
    test_id = str(uuid.uuid4())
    test.id = test_id
    test.created_at = datetime.now().isoformat()
    
    # TODO: Actually call LLM API here
    # For now, just store the test configuration
    tests_db[test_id] = test
    return test


@app.get("/api/tests", response_model=List[PromptTest])
async def get_tests(prompt_id: Optional[str] = None):
    """Get all tests, optionally filtered by prompt_id"""
    tests = list(tests_db.values())
    if prompt_id:
        tests = [t for t in tests if t.prompt_id == prompt_id]
    return tests


@app.get("/api/tests/{test_id}", response_model=PromptTest)
async def get_test(test_id: str):
    """Get a specific test by ID"""
    if test_id not in tests_db:
        raise HTTPException(status_code=404, detail="Test not found")
    return tests_db[test_id]


# Comparison Endpoints
@app.post("/api/comparisons", response_model=PromptComparison)
async def create_comparison(comparison: PromptComparison):
    """Create a new comparison between multiple test results"""
    comparison_id = str(uuid.uuid4())
    comparison.id = comparison_id
    comparison.created_at = datetime.now().isoformat()
    comparisons_db[comparison_id] = comparison
    return comparison


@app.get("/api/comparisons", response_model=List[PromptComparison])
async def get_comparisons():
    """Get all comparisons"""
    return list(comparisons_db.values())


# Quality Scoring Endpoints
@app.post("/api/quality-scores", response_model=QualityScore)
async def create_quality_score(score: QualityScore):
    """Add a quality score for a test"""
    quality_scores_db[score.test_id] = score
    return score


@app.get("/api/quality-scores/{test_id}", response_model=QualityScore)
async def get_quality_score(test_id: str):
    """Get quality score for a test"""
    if test_id not in quality_scores_db:
        raise HTTPException(status_code=404, detail="Quality score not found")
    return quality_scores_db[test_id]


# Hallucination Detection Endpoints
@app.post("/api/hallucination-checks", response_model=HallucinationCheck)
async def create_hallucination_check(check: HallucinationCheck):
    """Add a hallucination check result"""
    hallucination_checks_db[check.test_id] = check
    return check


@app.get("/api/hallucination-checks/{test_id}", response_model=HallucinationCheck)
async def get_hallucination_check(test_id: str):
    """Get hallucination check for a test"""
    if test_id not in hallucination_checks_db:
        raise HTTPException(status_code=404, detail="Hallucination check not found")
    return hallucination_checks_db[test_id]


# A/B Testing Endpoints
@app.post("/api/ab-tests", response_model=ABTest)
async def create_ab_test(ab_test: ABTest):
    """Create a new A/B test"""
    test_id = str(uuid.uuid4())
    ab_test.id = test_id
    ab_test.created_at = datetime.now().isoformat()
    ab_tests_db[test_id] = ab_test
    return ab_test


@app.get("/api/ab-tests", response_model=List[ABTest])
async def get_ab_tests():
    """Get all A/B tests"""
    return list(ab_tests_db.values())


@app.get("/api/ab-tests/{test_id}", response_model=ABTest)
async def get_ab_test(test_id: str):
    """Get a specific A/B test"""
    if test_id not in ab_tests_db:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return ab_tests_db[test_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

