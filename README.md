# Prompt Helper Chat - Starter Code

A simple chat application using Gemini API. This is the starter code for building a Prompt Helper Chat with a Prompt Critic character that helps optimize prompts using techniques from the Prompt Engineering Foundations course.

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
├── .env.example         # Environment variables template
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Get Your Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Backend Setup

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

4. Create a `.env` file (copy from `.env.example`):

```bash
# From backend directory
cp ../.env.example .env
```

5. Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your-actual-api-key-here
```

6. Run the FastAPI server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup

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

### Current (Starter Code)

- **Simple Chat Interface**: Basic chat with Gemini API
- **Message History**: Stores last 20 messages in memory
- **Error Handling**: Basic error messages for API failures

## API Endpoints

### Chat

- `POST /api/chat` - Send a message and get a response

  ```json
  {
    "message": "Hello, how are you?"
  }
  ```

- `GET /api/history` - Get chat history
- `DELETE /api/history` - Clear chat history

## Next Steps

### Module 1 - Setup & Foundation (Lesson 1.7: Project Kickoff)

- [ ] Create comprehensive Prompt Critic system prompt (500+ tokens) with character identity and course expertise
  - **Activate**: Update the `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py` with a detailed system prompt (500+ tokens)
- [ ] Enhance conversation history to maintain context across messages
  - **Activate**: Change `ENABLE_CONVERSATION_HISTORY = False` to `True` in `backend/main.py`
- [ ] Add token counting to track usage and prevent context window overflow
  - **Activate**: Change `ENABLE_TOKEN_COUNTING = False` to `True` in both `backend/main.py` and `frontend/app.js`

### Module 2 - Frameworks & Best Practices (Lesson 2.8: Project Extension)

- [ ] Enhance system prompt with role prompting: define Prompt Critic as expert consultant
  - **Activate**: Update `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py` to include role prompting sections
- [ ] Add few-shot examples to system prompt showing good vs bad prompt feedback
  - **Activate**: Add few-shot examples section to `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py`
- [ ] Implement structured feedback format: use markdown, lists, and tables in responses
  - **Activate**: Add structured output formatting instructions to `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py`
- [ ] Add chain-of-thought reasoning: Prompt Critic explains step-by-step analysis
  - **Activate**: Add chain-of-thought reasoning instructions to `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py`
- [ ] Create prompt quality scoring: analyze clarity, completeness, and effectiveness
  - **Activate**: Add quality scoring framework to `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py`
- [ ] Add iterative refinement: track prompt versions through conversation
  - **Activate**: Change `ENABLE_PROMPT_VERSION_TRACKING = False` to `True` in both `backend/main.py` and `frontend/app.js`

### Module 3 - Advanced Techniques (Lesson 3.8: Project Extension)

- [ ] Refine system prompt structure: separate identity, expertise, and behavior sections
  - **Activate**: Reorganize `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py` into clear sections
- [ ] Add structured output option: Prompt Critic can return JSON-formatted suggestions
  - **Activate**: Change `ENABLE_JSON_OUTPUT = False` to `True` in both `backend/main.py` and `frontend/app.js`
- [ ] Implement temperature control: allow users to adjust feedback style (creative vs precise)
  - **Activate**: Change `ENABLE_TEMPERATURE_CONTROL = False` to `True` in both `backend/main.py` and `frontend/app.js`
- [ ] Add reasoning strategies: Prompt Critic uses step-by-step or ReAct for complex analysis
  - **Activate**: Update `PROMPT_CRITIC_SYSTEM_PROMPT` in `backend/main.py` to include reasoning strategy instructions
- [ ] Create prompt chaining: multi-step optimization workflow (analyze → suggest → refine → final)
  - **Activate**: Change `ENABLE_PROMPT_CHAINING = False` to `True` in both `backend/main.py` and `frontend/app.js`
- [ ] Implement context window management: summarize old messages when conversation gets long
  - **Activate**: Change `ENABLE_CONTEXT_WINDOW_MANAGEMENT = False` to `True` in both `backend/main.py` and `frontend/app.js`

### Module 4 - Business Applications & Optimization (Lesson 4.7: Project Extension)

- [ ] Add defensive prompting: protect system prompt from injection attacks
  - **Activate**: Change `ENABLE_DEFENSIVE_PROMPTING = False` to `True` in `backend/main.py` (basic sanitization is always active)
- [ ] Add feedback evaluation: track which suggestions users find most helpful
  - **Activate**: Change `ENABLE_FEEDBACK_EVALUATION = False` to `True` in both `backend/main.py` and `frontend/app.js`
- [ ] Create prompt library: save and organize optimized prompts with tags/categories using SQLite
  - **Activate**: Change `ENABLE_PROMPT_LIBRARY = False` to `True` in both `backend/main.py` and `frontend/app.js`
- [ ] Add conversation export: save chat history as markdown or JSON
  - **Activate**: Change `ENABLE_CONVERSATION_EXPORT = False` to `True` in both `backend/main.py` and `frontend/app.js`

## Suggested Features (Optional Enhancements)

These features can be added as extensions to the project:

### LLM Provider Selection

- Add UI controls to switch between different LLM providers (Gemini, DeepSeek, OpenRouter)
- Implement provider-specific configuration and handling
- Add provider comparison functionality

## Feature Activation

All code features are controlled by boolean flags. To activate any feature:

1. **Backend features**: Edit `backend/main.py` and change the corresponding `ENABLE_*` flag from `False` to `True`
2. **Frontend features**: Edit `frontend/app.js` and change the corresponding `ENABLE_*` flag from `False` to `True`
3. **Restart the backend server** after making changes to `backend/main.py`
4. **Refresh the browser** after making changes to `frontend/app.js`

**Important**: Some features require both backend AND frontend flags to be enabled to work properly. The checklist items above indicate which files need to be updated for each feature.

## Development Notes

- Currently uses in-memory storage for chat history
- Prompt library uses SQLite database (`prompt_library.db`) - created automatically on first use
- The `.env` file should be in the `backend/` directory (created via `cp ../.env.example .env` from the backend directory)
- The `.env` file is gitignored - never commit your API key!
