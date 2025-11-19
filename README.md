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

### Module 1 - Setup & Foundation

- [ ] Create comprehensive Prompt Critic system prompt (500+ tokens) with character identity and course expertise (Lesson 1.6)
- [ ] Enhance conversation history to maintain context across messages (Lesson 1.3)
- [ ] Add token counting to track usage and prevent context window overflow (Lesson 1.4, 1.7)
- [ ] Add error handling for API failures and rate limits (Lesson 1.7)

### Module 2 - Frameworks & Best Practices

- [ ] Enhance system prompt with role prompting: define Prompt Critic as expert consultant (Lesson 2.3)
- [ ] Add few-shot examples to system prompt showing good vs bad prompt feedback (Lesson 2.2)
- [ ] Implement structured feedback format: use markdown, lists, and tables in responses (Lesson 2.4)
- [ ] Add chain-of-thought reasoning: Prompt Critic explains step-by-step analysis (Lesson 2.6)
- [ ] Create prompt quality scoring: analyze clarity, completeness, and effectiveness (Lesson 2.8)
- [ ] Add iterative refinement: track prompt versions through conversation (Lesson 2.5)

### Module 3 - Advanced Techniques

- [ ] Refine system prompt structure: separate identity, expertise, and behavior sections (Lesson 3.1)
- [ ] Add structured output option: Prompt Critic can return JSON-formatted suggestions (Lesson 3.2)
- [ ] Implement temperature control: allow users to adjust feedback style (creative vs precise) (Lesson 3.4)
- [ ] Add reasoning strategies: Prompt Critic uses step-by-step or ReAct for complex analysis (Lesson 3.5)
- [ ] Create prompt chaining: multi-step optimization workflow (analyze → suggest → refine → final) (Lesson 3.6)
- [ ] Implement context window management: summarize old messages when conversation gets long (Lesson 3.8)

### Module 4 - Business Applications & Optimization

- [ ] Add defensive prompting: protect system prompt from injection attacks (Lesson 4.2)
- [ ] Implement meta-prompting: create endpoint to optimize Prompt Critic's own system prompt (Lesson 4.3)
- [ ] Add conversation export: save chat history as markdown or JSON (Lesson 4.7)
- [ ] Create prompt library: save and organize optimized prompts with tags/categories using SQLite (Lesson 4.6)
- [ ] Add feedback evaluation: track which suggestions users find most helpful (Lesson 4.1)

## Suggested Features (Optional Enhancements)

These features can be added as extensions to the project:

### LLM Provider Selection

- Add UI controls to switch between different LLM providers (Gemini, DeepSeek, OpenRouter)
- Implement provider-specific configuration and handling
- Add provider comparison functionality

## Development Notes

- Currently uses in-memory storage for chat history
- The `.env` file should be in the `backend/` directory (created via `cp ../.env.example .env` from the backend directory)
- The `.env` file is gitignored - never commit your API key!
