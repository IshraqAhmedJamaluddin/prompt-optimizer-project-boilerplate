# Prompt Optimizer Tool

A web application for testing, comparing, and optimizing prompts across multiple LLMs (Claude, ChatGPT, Gemini, DeepSeek, etc.).

## Project Structure

```
prompt-optimizer-tool/
├── backend/
│   ├── main.py          # FastAPI backend application
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── index.html       # Main HTML file
│   ├── styles.css       # Custom styles
│   └── app.js           # Frontend JavaScript
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Open `frontend/index.html` in a web browser, or use a local server:

```bash
cd frontend
python -m http.server 8080  # Python 3
# or
npx http-server -p 8080     # Node.js
```

2. Open `http://localhost:8080` in your browser

### API Documentation

Once the backend is running, you can access:

- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Features

### Core Functionality

- **Prompt Library Management**: Create, edit, delete, and organize prompts with categories and tags
- **Multi-LLM Testing**: Test prompts across multiple LLM providers (Claude, ChatGPT, Gemini, DeepSeek)
- **Quality Scoring**: Automated scoring for clarity, completeness, and effectiveness
- **Hallucination Detection**: Check for factual inaccuracies in outputs
- **A/B Testing**: Compare multiple prompt variations systematically
- **Comparison Tools**: Side-by-side comparison of outputs from different prompts or providers
- **Temperature Testing**: Test prompts across different temperature settings
- **Version Control**: Track prompt iterations and improvements

## API Endpoints

### Prompts

- `GET /api/prompts` - Get all prompts (optional `?category=writing`)
- `GET /api/prompts/{id}` - Get a specific prompt
- `POST /api/prompts` - Create a new prompt
- `PUT /api/prompts/{id}` - Update a prompt (creates new version)
- `DELETE /api/prompts/{id}` - Delete a prompt

### LLM Providers

- `GET /api/providers` - Get all configured LLM providers
- `POST /api/providers` - Add a new LLM provider

### Testing

- `GET /api/tests` - Get all tests (optional `?prompt_id={id}`)
- `GET /api/tests/{id}` - Get a specific test
- `POST /api/tests` - Run a new prompt test

### Comparisons

- `GET /api/comparisons` - Get all comparisons
- `POST /api/comparisons` - Create a new comparison

### Quality Scores

- `GET /api/quality-scores/{test_id}` - Get quality score for a test
- `POST /api/quality-scores` - Add a quality score

### Hallucination Detection

- `GET /api/hallucination-checks/{test_id}` - Get hallucination check
- `POST /api/hallucination-checks` - Add hallucination check result

### A/B Testing

- `GET /api/ab-tests` - Get all A/B tests
- `GET /api/ab-tests/{id}` - Get a specific A/B test
- `POST /api/ab-tests` - Create a new A/B test

## Next Steps

### Module 1 - Setup & Foundation

- [ ] Configure free API access (DeepSeek, OpenRouter, etc.) (Lesson 1.7)
- [ ] Implement actual LLM API calls in testing endpoint (Lesson 1.7)
- [ ] Add response time tracking (Lesson 1.7)
- [ ] Implement token counting (Lesson 1.7)

### Module 2 - Quality Scoring & Detection

- [ ] Implement automated clarity scoring (Lesson 2.8)
- [ ] Implement completeness scoring (Lesson 2.8)
- [ ] Implement effectiveness scoring (Lesson 2.8)
- [ ] Add few-shot example management (Lesson 2.2)
- [ ] Implement hallucination detection algorithms (Lesson 2.7)

### Module 3 - Advanced Testing Features

- [ ] Add structured output parsing (JSON validation) (Lesson 3.2)
- [ ] Implement temperature range testing (Lesson 3.4)
- [ ] Add prompt chaining functionality (Lesson 3.6)
- [ ] Implement context window management tracking (Lesson 3.8)
- [ ] Add iteration history logging (Lesson 3.8)

### Module 4 - Optimization & Meta-Prompting

- [ ] Implement A/B testing automation (Lesson 4.1)
- [ ] Add defensive prompt templates (injection prevention) (Lesson 4.2)
- [ ] Implement meta-prompt generation (Lesson 4.3)
- [ ] Add cost optimization recommendations (Lesson 4.7)
- [ ] Implement export functionality (CSV, JSON, Markdown) (Lesson 4.7)

### General Improvements

- [ ] Add database persistence (SQLite, PostgreSQL)
- [ ] Implement authentication and user management
- [ ] Add export/import functionality
- [ ] Implement batch processing
- [ ] Add comprehensive error handling
- [ ] Add logging and monitoring

## Configuration

### Setting Up LLM Providers

To configure LLM providers, you'll need to add them via the API:

```python
# Example: Add DeepSeek provider
POST /api/providers
{
    "id": "deepseek",
    "name": "DeepSeek",
    "api_key": "your-api-key",
    "endpoint": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-chat",
    "temperature": 0.7
}
```

## Development Notes

- Currently uses in-memory storage. For production, integrate with a database.
- LLM API integration is a placeholder - implement actual API calls based on provider documentation.
- Quality scoring and hallucination detection algorithms need to be implemented.
