// API Base URL
const API_BASE_URL = 'http://localhost:8000/api';

// Chat Functions
async function sendMessage(event) {
    event.preventDefault();
    
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }
    
    // Display user message
    addMessage(message, 'user');
    input.value = '';
    
    // Show loading indicator
    const loadingId = addMessage('Thinking...', 'assistant', true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove loading indicator
        document.getElementById(loadingId).remove();
        
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'assistant', false, true);
        } else {
            addMessage(data.response, 'assistant');
        }
        
    } catch (error) {
        document.getElementById(loadingId).remove();
        addMessage(`Error: ${error.message}`, 'assistant', false, true);
        console.error('Error sending message:', error);
    }
}

function addMessage(message, sender, isLoading = false, isError = false) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    const messageId = 'msg-' + Date.now();
    messageDiv.id = messageId;
    
    messageDiv.className = `message ${sender === 'user' ? 'user-message' : 'assistant-message'} mb-3`;
    
    if (isError) {
        messageDiv.className += ' text-danger';
    } else if (isLoading) {
        messageDiv.className += ' text-muted';
    }
    
    messageDiv.innerHTML = `
        <div class="d-flex ${sender === 'user' ? 'justify-content-end' : 'justify-content-start'}">
            <div class="message-bubble ${sender === 'user' ? 'bg-primary text-white' : 'bg-light'} p-3 rounded" style="max-width: 80%;">
                ${isLoading ? '<span class="spinner-border spinner-border-sm me-2"></span>' : ''}
                ${escapeHtml(message)}
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return messageId;
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear the chat history?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/history`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = '<div class="alert alert-info">Chat history cleared. Start a new conversation!</div>';
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Error clearing history');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
