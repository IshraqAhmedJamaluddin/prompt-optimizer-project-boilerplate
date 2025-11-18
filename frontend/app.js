// API Base URL
const API_BASE_URL = 'http://localhost:8000/api';

// View Management
function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.view-section').forEach(view => {
        view.style.display = 'none';
    });

    // Show selected view
    document.getElementById(`${viewName}-view`).style.display = 'block';

    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');

    // Load data for the view
    if (viewName === 'prompts') {
        loadPrompts();
    } else if (viewName === 'testing') {
        loadPromptsForTesting();
        loadProviders();
    } else if (viewName === 'comparison') {
        loadComparisons();
    } else if (viewName === 'ab-testing') {
        loadABTests();
    }
}

// Prompt Management
async function loadPrompts() {
    try {
        const category = document.getElementById('categoryFilter')?.value || '';
        const url = category 
            ? `${API_BASE_URL}/prompts?category=${category}`
            : `${API_BASE_URL}/prompts`;
        
        const response = await fetch(url);
        const prompts = await response.json();
        
        const promptsList = document.getElementById('prompts-list');
        promptsList.innerHTML = '';

        if (prompts.length === 0) {
            promptsList.innerHTML = '<div class="col-12"><p class="text-muted">No prompts yet. Create your first prompt!</p></div>';
            return;
        }

        prompts.forEach(prompt => {
            const card = createPromptCard(prompt);
            promptsList.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading prompts:', error);
        showAlert('Error loading prompts', 'danger');
    }
}

function createPromptCard(prompt) {
    const col = document.createElement('div');
    col.className = 'col-md-6';
    
    const tags = prompt.tags && prompt.tags.length > 0 
        ? prompt.tags.map(tag => `<span class="badge bg-secondary me-1">${tag}</span>`).join('')
        : '';

    col.innerHTML = `
        <div class="card prompt-card">
            <div class="card-body">
                <h5 class="card-title">${prompt.title}</h5>
                ${prompt.category ? `<span class="badge bg-primary mb-2">${prompt.category}</span>` : ''}
                <p class="card-text">${truncateText(prompt.content, 150)}</p>
                ${tags ? `<div class="mb-2">${tags}</div>` : ''}
                <p class="card-text"><small class="text-muted">Version: ${prompt.version || 1}</small></p>
                <div class="btn-group-actions">
                    <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); editPrompt('${prompt.id}')">Edit</button>
                    <button class="btn btn-sm btn-success" onclick="event.stopPropagation(); testPrompt('${prompt.id}')">Test</button>
                    <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deletePrompt('${prompt.id}')">Delete</button>
                </div>
            </div>
        </div>
    `;
    
    return col;
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function showPromptForm(promptId = null) {
    const modal = new bootstrap.Modal(document.getElementById('promptModal'));
    const form = document.getElementById('promptForm');
    form.reset();

    if (promptId) {
        document.getElementById('promptModalTitle').textContent = 'Edit Prompt';
        loadPromptForEdit(promptId);
    } else {
        document.getElementById('promptModalTitle').textContent = 'Create Prompt';
        document.getElementById('promptId').value = '';
    }

    modal.show();
}

async function loadPromptForEdit(promptId) {
    try {
        const response = await fetch(`${API_BASE_URL}/prompts/${promptId}`);
        const prompt = await response.json();

        document.getElementById('promptId').value = prompt.id;
        document.getElementById('promptTitle').value = prompt.title;
        document.getElementById('promptContent').value = prompt.content;
        document.getElementById('promptCategory').value = prompt.category || '';
        document.getElementById('promptTags').value = prompt.tags ? prompt.tags.join(', ') : '';
    } catch (error) {
        console.error('Error loading prompt:', error);
        showAlert('Error loading prompt', 'danger');
    }
}

async function savePrompt() {
    const form = document.getElementById('promptForm');
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const promptId = document.getElementById('promptId').value;
    const promptData = {
        title: document.getElementById('promptTitle').value,
        content: document.getElementById('promptContent').value,
        category: document.getElementById('promptCategory').value || null,
        tags: document.getElementById('promptTags').value
            ? document.getElementById('promptTags').value.split(',').map(t => t.trim()).filter(t => t)
            : []
    };

    try {
        let response;
        if (promptId) {
            response = await fetch(`${API_BASE_URL}/prompts/${promptId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(promptData)
            });
        } else {
            response = await fetch(`${API_BASE_URL}/prompts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(promptData)
            });
        }

        if (response.ok) {
            const modal = bootstrap.Modal.getInstance(document.getElementById('promptModal'));
            modal.hide();
            loadPrompts();
            showAlert('Prompt saved successfully!', 'success');
        } else {
            throw new Error('Failed to save prompt');
        }
    } catch (error) {
        console.error('Error saving prompt:', error);
        showAlert('Error saving prompt', 'danger');
    }
}

async function deletePrompt(promptId) {
    if (!confirm('Are you sure you want to delete this prompt?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/prompts/${promptId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadPrompts();
            showAlert('Prompt deleted successfully!', 'success');
        } else {
            throw new Error('Failed to delete prompt');
        }
    } catch (error) {
        console.error('Error deleting prompt:', error);
        showAlert('Error deleting prompt', 'danger');
    }
}

function editPrompt(promptId) {
    showPromptForm(promptId);
}

function testPrompt(promptId) {
    showView('testing');
    document.getElementById('testPromptSelect').value = promptId;
}

// Testing
async function loadPromptsForTesting() {
    try {
        const response = await fetch(`${API_BASE_URL}/prompts`);
        const prompts = await response.json();
        
        const select = document.getElementById('testPromptSelect');
        select.innerHTML = '<option value="">Choose a prompt...</option>';
        
        prompts.forEach(prompt => {
            const option = document.createElement('option');
            option.value = prompt.id;
            option.textContent = prompt.title;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading prompts:', error);
    }
}

async function loadProviders() {
    try {
        const response = await fetch(`${API_BASE_URL}/providers`);
        const providers = await response.json();
        
        const select = document.getElementById('testProviderSelect');
        select.innerHTML = '<option value="">Choose a provider...</option>';
        
        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = provider.name;
            select.appendChild(option);
        });

        // If no providers, show message
        if (providers.length === 0) {
            select.innerHTML = '<option value="">No providers configured</option>';
        }
    } catch (error) {
        console.error('Error loading providers:', error);
    }
}

async function runTest() {
    const promptId = document.getElementById('testPromptSelect').value;
    const providerId = document.getElementById('testProviderSelect').value;
    const inputText = document.getElementById('testInput').value;
    const temperature = parseFloat(document.getElementById('testTemperature').value);

    if (!promptId || !providerId || !inputText) {
        showAlert('Please fill in all required fields', 'warning');
        return;
    }

    try {
        const testData = {
            prompt_id: promptId,
            provider_id: providerId,
            input_text: inputText,
            temperature: temperature
        };

        const response = await fetch(`${API_BASE_URL}/tests`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });

        if (response.ok) {
            const test = await response.json();
            displayTestResult(test);
            showAlert('Test completed!', 'success');
        } else {
            throw new Error('Test failed');
        }
    } catch (error) {
        console.error('Error running test:', error);
        showAlert('Error running test', 'danger');
    }
}

function displayTestResult(test) {
    const resultsDiv = document.getElementById('test-results');
    const resultCard = document.createElement('div');
    resultCard.className = 'test-result-card';
    
    resultCard.innerHTML = `
        <h5>Test Result</h5>
        <p><strong>Test ID:</strong> ${test.id}</p>
        <p><strong>Temperature:</strong> ${test.temperature}</p>
        ${test.response_time_ms ? `<p><strong>Response Time:</strong> ${test.response_time_ms}ms</p>` : ''}
        ${test.output_text ? `
            <div class="mt-3">
                <strong>Output:</strong>
                <div class="code-block">${escapeHtml(test.output_text)}</div>
            </div>
        ` : ''}
        ${test.error ? `
            <div class="alert alert-danger mt-3">Error: ${escapeHtml(test.error)}</div>
        ` : ''}
    `;

    resultsDiv.insertBefore(resultCard, resultsDiv.firstChild);
}

// Comparison
async function loadComparisons() {
    try {
        const response = await fetch(`${API_BASE_URL}/comparisons`);
        const comparisons = await response.json();

        const resultsDiv = document.getElementById('comparison-results');
        resultsDiv.innerHTML = '';

        if (comparisons.length === 0) {
            resultsDiv.innerHTML = '<p class="text-muted">No comparisons yet.</p>';
            return;
        }

        // TODO: Display comparison results in a table format
        comparisons.forEach(comparison => {
            const div = document.createElement('div');
            div.className = 'card mb-3';
            div.innerHTML = `
                <div class="card-body">
                    <h5>Comparison: ${comparison.id}</h5>
                    <p>Test IDs: ${comparison.test_ids.join(', ')}</p>
                </div>
            `;
            resultsDiv.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading comparisons:', error);
    }
}

// A/B Testing
async function loadABTests() {
    try {
        const response = await fetch(`${API_BASE_URL}/ab-tests`);
        const abTests = await response.json();

        const listDiv = document.getElementById('ab-tests-list');
        listDiv.innerHTML = '';

        if (abTests.length === 0) {
            listDiv.innerHTML = '<p class="text-muted">No A/B tests yet.</p>';
            return;
        }

        abTests.forEach(test => {
            const div = document.createElement('div');
            div.className = 'card mb-3';
            div.innerHTML = `
                <div class="card-body">
                    <h5>${test.name}</h5>
                    <p>Variations: ${test.prompt_variations.length}</p>
                    <p>Test Cases: ${test.test_cases.length}</p>
                </div>
            `;
            listDiv.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading A/B tests:', error);
    }
}

function showABTestForm() {
    // TODO: Implement A/B test form
    showAlert('A/B test form coming soon!', 'info');
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadPrompts();
});

