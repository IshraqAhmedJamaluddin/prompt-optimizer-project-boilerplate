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

This is the starter code with a simple chat interface. Throughout the course, you'll extend this to build a **Prompt Helper Chat** with a **Prompt Critic** character that helps users optimize their prompts using prompt engineering techniques.

The complete solution includes:
- **Prompt Critic Character**: An expert AI assistant with a comprehensive system prompt
- **Educational Guidance**: Helps users improve prompts using course techniques
- **Chat Interface**: Interactive conversation for iterative prompt refinement
- **Constructive Feedback**: Provides specific, actionable suggestions with examples

See the `solutions` branch for the complete implementation.

## Development Notes

- Currently uses in-memory storage for chat history
- The `.env` file should be in the `backend/` directory (created via `cp ../.env.example .env` from the backend directory)
- The `.env` file is gitignored - never commit your API key!
