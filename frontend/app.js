// API Base URL
const API_BASE_URL = "http://localhost:8000/api";

// Feature flags - Mirror backend feature flags
// Students can enable/disable these features by changing the flags below
const ENABLE_TOKEN_COUNTING = true; // Set to true to show token usage (Lesson 1.4, 1.7)
const ENABLE_PROMPT_VERSION_TRACKING = true; // Set to true to show prompt versions (Lesson 2.5)
const ENABLE_JSON_OUTPUT = true; // Set to true to enable JSON output format option (Lesson 3.2)
const ENABLE_TEMPERATURE_CONTROL = true; // Set to true to enable temperature slider (Lesson 3.4)
const ENABLE_PROMPT_CHAINING = true; // Set to true to enable prompt chaining workflow (Lesson 3.6)
const ENABLE_CONVERSATION_EXPORT = true; // Set to true to enable conversation export (Lesson 4.7)
const ENABLE_PROMPT_LIBRARY = true; // Set to true to enable prompt library features (Lesson 4.6)
const ENABLE_FEEDBACK_EVALUATION = true; // Set to true to enable feedback evaluation (Lesson 4.1)
const ENABLE_META_PROMPTING = true; // Set to true to enable meta-prompting endpoint (Lesson 4.3)

// Chat state
let conversationHistory = [];
let characterInfo = null;
let sessionId = "default"; // Session ID for conversation tracking

// DOM elements
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatMessages = document.getElementById("chat-messages");
const sendBtn = document.getElementById("send-btn");
const characterDetailsContent = document.getElementById(
  "character-details-content"
);

// Load character information
async function loadCharacterInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/character`);
    characterInfo = await response.json();

    // Update character display
    document.getElementById("character-name").textContent = characterInfo.name;
    document.getElementById("character-role").textContent = characterInfo.role;
    document.getElementById("character-avatar").textContent =
      characterInfo.avatar;
    document.getElementById(
      "chat-header"
    ).textContent = `Chat with ${characterInfo.name} ${characterInfo.avatar}`;

    // Update character details
    characterDetailsContent.innerHTML = `
            <div class="mb-3">
                <h6>Personality Traits</h6>
                <p>${characterInfo.personality_traits.join(", ")}</p>
            </div>
            <div class="mb-3">
                <h6>Tone of Voice</h6>
                <p>${characterInfo.tone_of_voice}</p>
            </div>
            <div class="mb-3">
                <h6>How I Help</h6>
                <ul>
                    <li>Analyze your prompts and identify areas for improvement</li>
                    <li>Suggest specific prompt engineering techniques from the course</li>
                    <li>Provide before/after examples of improved prompts</li>
                    <li>Explain the reasoning behind my suggestions</li>
                    <li>Help you refine prompts through iterative improvement</li>
                </ul>
            </div>
            <div class="mb-3">
                <h6>System Prompt</h6>
                <pre class="bg-light p-3 rounded"><code>${escapeHtml(
                  characterInfo.system_prompt
                )}</code></pre>
            </div>
        `;
  } catch (error) {
    console.error("Error loading character info:", error);
    characterDetailsContent.innerHTML =
      '<p class="text-danger">Error loading character information.</p>';
  }
}

// Add user message to chat
function addUserMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message mb-3";
  messageDiv.innerHTML = `
        <div class="d-flex justify-content-end">
            <div class="message-bubble user-message">
                ${escapeHtml(message)}
            </div>
        </div>
    `;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();

  // Add to conversation history
  conversationHistory.push({
    role: "user",
    content: message,
    timestamp: new Date().toISOString(),
  });
}

// Add assistant message to chat
function addAssistantMessage(
  message,
  characterName = "Prompt Critic",
  metadata = {}
) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message mb-3";

  let metadataHtml = "";

  // Add token information if enabled
  if (ENABLE_TOKEN_COUNTING && metadata.tokens_used !== undefined) {
    metadataHtml += `<div class="metadata-info mt-2 text-muted small">
            <span>Tokens used: ${metadata.tokens_used}</span>`;
    if (metadata.tokens_remaining !== undefined) {
      metadataHtml += `<span class="ms-2">Tokens remaining: ${metadata.tokens_remaining}</span>`;
    }
    metadataHtml += `</div>`;
  }

  // Add prompt version if enabled
  if (ENABLE_PROMPT_VERSION_TRACKING && metadata.prompt_version) {
    metadataHtml += `<div class="metadata-info mt-2 text-muted small">
            <span>Prompt Version: ${metadata.prompt_version}</span>
        </div>`;
  }

  // Add quality score if available
  if (metadata.quality_score) {
    const scores = metadata.quality_score;
    metadataHtml += `<div class="metadata-info mt-2 text-muted small">
            Quality: Clarity ${(scores.clarity * 10).toFixed(1)}/10 | 
            Completeness ${(scores.completeness * 10).toFixed(1)}/10 | 
            Effectiveness ${(scores.effectiveness * 10).toFixed(1)}/10
        </div>`;
  }

  messageDiv.innerHTML = `
        <div class="d-flex justify-content-start">
            <div class="message-bubble assistant-message">
                <strong>${characterName}:</strong> ${formatMessage(message)}
                ${metadataHtml}
            </div>
        </div>
    `;
  chatMessages.appendChild(messageDiv);

  scrollToBottom();

  // Add to conversation history
  conversationHistory.push({
    role: "assistant",
    content: message,
    timestamp: new Date().toISOString(),
  });
}

// Format message with markdown-like formatting
function formatMessage(text) {
  // Escape HTML first
  let formatted = escapeHtml(text);

  // Convert code blocks (```code```)
  formatted = formatted.replace(
    /```([\s\S]*?)```/g,
    '<pre class="code-inline"><code>$1</code></pre>'
  );

  // Convert inline code (`code`)
  formatted = formatted.replace(
    /`([^`]+)`/g,
    '<code class="code-inline">$1</code>'
  );

  // Convert bold (**text**)
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Convert italic (*text*)
  formatted = formatted.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // Convert line breaks
  formatted = formatted.replace(/\n/g, "<br>");

  return formatted;
}

// Add error message
function addErrorMessage(message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "alert alert-danger mb-3";
  messageDiv.textContent = message;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Scroll to bottom of chat
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Set loading state
function setLoading(loading) {
  messageInput.disabled = loading;
  sendBtn.disabled = loading;
  const spinner = sendBtn.querySelector(".spinner-border");
  const sendText = sendBtn.querySelector(".send-text");

  if (loading) {
    spinner.classList.remove("d-none");
    sendText.textContent = "Sending...";
  } else {
    spinner.classList.add("d-none");
    sendText.textContent = "Send";
  }
}

// Handle form submission
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const message = messageInput.value.trim();
  if (!message) return;

  // Remove welcome message if present
  const welcomeMsg = document.getElementById("welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }

  // Add user message to chat
  addUserMessage(message);
  messageInput.value = "";

  // Set loading state
  setLoading(true);

  try {
    // Build request body with optional parameters based on feature flags
    const requestBody = {
      message: message,
      conversation_history: conversationHistory.slice(0, -1), // Exclude the message we just added
      session_id: sessionId,
    };

    // Add optional parameters if features are enabled
    if (ENABLE_TEMPERATURE_CONTROL) {
      const tempValue = document.getElementById("temperature-slider")?.value;
      if (tempValue) {
        requestBody.temperature = parseFloat(tempValue);
      }
    }

    if (ENABLE_JSON_OUTPUT) {
      const outputFormat = document.getElementById(
        "output-format-select"
      )?.value;
      if (outputFormat) {
        requestBody.output_format = outputFormat;
      }
    }

    // Reasoning strategy is always available (prompt-based feature)
    const reasoningStrategy = document.getElementById(
      "reasoning-strategy-select"
    )?.value;
    if (reasoningStrategy) {
      requestBody.reasoning_strategy = reasoningStrategy;
    }

    if (ENABLE_PROMPT_CHAINING) {
      const enableChaining = document.getElementById(
        "enable-chaining-checkbox"
      )?.checked;
      requestBody.enable_chaining = enableChaining || false;
    }

    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (response.ok) {
      const data = await response.json();

      // Extract metadata for display
      const metadata = {
        tokens_used: data.tokens_used,
        tokens_remaining: data.tokens_remaining,
        prompt_version: data.prompt_version,
        quality_score: data.quality_score,
      };

      addAssistantMessage(data.response, data.character_name, metadata);
    } else {
      const errorData = await response
        .json()
        .catch(() => ({ detail: "Failed to get response" }));
      addErrorMessage(
        `Error: ${errorData.detail || "Failed to get response from server"}`
      );
    }
  } catch (error) {
    console.error("Error:", error);
    addErrorMessage(
      "Error: Unable to connect to server. Make sure the backend is running on http://localhost:8000"
    );
  } finally {
    // Re-enable input
    setLoading(false);
    messageInput.focus();
  }
});

// Initialize UI based on feature flags
function initializeUI() {
  // Create advanced options container
  const chatBody = document.querySelector(".card-body");
  const form = document.getElementById("chat-form");

  if (!chatBody || !form) {
    console.error("Could not find chat-body or form element");
    return;
  }

  // Create advanced options section
  let advancedOptions = document.getElementById("advanced-options");
  let toggleBtn = document.getElementById("advanced-options-toggle");

  if (!advancedOptions) {
    advancedOptions = document.createElement("div");
    advancedOptions.id = "advanced-options";
    advancedOptions.className = "advanced-options mb-3 p-3 bg-light rounded";
    advancedOptions.style.display = "none"; // Hidden by default
    form.parentNode.insertBefore(advancedOptions, form);

    // Add toggle button
    toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.id = "advanced-options-toggle";
    toggleBtn.className = "btn btn-sm btn-outline-secondary mb-2";
    toggleBtn.textContent = "⚙️ Advanced Options";
    toggleBtn.onclick = () => {
      const opts = document.getElementById("advanced-options");
      if (opts) {
        opts.style.display = opts.style.display === "none" ? "block" : "none";
      }
    };
    form.parentNode.insertBefore(toggleBtn, advancedOptions);
  }

  // Make sure we have a reference to the actual DOM element
  advancedOptions = document.getElementById("advanced-options");
  if (!advancedOptions) {
    console.error("Advanced options container not found");
    return;
  }

  // Temperature control
  if (ENABLE_TEMPERATURE_CONTROL) {
    if (!document.getElementById("temperature-slider")) {
      const tempGroup = document.createElement("div");
      tempGroup.className = "mb-2";
      tempGroup.innerHTML = `
                <label class="form-label small">Temperature: <span id="temp-value">0.7</span></label>
                <input type="range" id="temperature-slider" class="form-range" min="0" max="2" step="0.1" value="0.7"
                    oninput="document.getElementById('temp-value').textContent = this.value">
            `;
      advancedOptions.appendChild(tempGroup);
      advancedOptions.style.display = "block";
    }
  }

  // Output format selector
  if (ENABLE_JSON_OUTPUT) {
    if (!document.getElementById("output-format-select")) {
      const formatGroup = document.createElement("div");
      formatGroup.className = "mb-2";
      formatGroup.innerHTML = `
                <label class="form-label small">Output Format:</label>
                <select id="output-format-select" class="form-select form-select-sm">
                    <option value="text">Text</option>
                    <option value="json">JSON</option>
                </select>
            `;
      advancedOptions.appendChild(formatGroup);
      advancedOptions.style.display = "block";
    }
  }

  // Reasoning strategy selector (always available)
  if (!document.getElementById("reasoning-strategy-select")) {
    const strategyGroup = document.createElement("div");
    strategyGroup.className = "mb-2";
    strategyGroup.innerHTML = `
            <label class="form-label small">Reasoning Strategy:</label>
            <select id="reasoning-strategy-select" class="form-select form-select-sm">
                <option value="direct">Direct</option>
                <option value="step_by_step">Step-by-Step</option>
                <option value="react">ReAct</option>
            </select>
        `;
    advancedOptions.appendChild(strategyGroup);
    advancedOptions.style.display = "block";
  }

  // Prompt chaining checkbox
  if (ENABLE_PROMPT_CHAINING) {
    if (!document.getElementById("enable-chaining-checkbox")) {
      const chainingGroup = document.createElement("div");
      chainingGroup.className = "mb-2 form-check";
      chainingGroup.innerHTML = `
                <input type="checkbox" id="enable-chaining-checkbox" class="form-check-input">
                <label class="form-check-label small" for="enable-chaining-checkbox">
                    Enable Prompt Chaining Workflow
                </label>
            `;
      advancedOptions.appendChild(chainingGroup);
      advancedOptions.style.display = "block";
    }
  }

  // Export button
  if (ENABLE_CONVERSATION_EXPORT) {
    if (!document.getElementById("export-btn")) {
      const exportBtn = document.createElement("button");
      exportBtn.type = "button";
      exportBtn.id = "export-btn";
      exportBtn.className = "btn btn-sm btn-outline-primary mt-2";
      exportBtn.textContent = "📥 Export Conversation";
      exportBtn.onclick = exportConversation;
      advancedOptions.appendChild(exportBtn);
      advancedOptions.style.display = "block";
    }
  }

  // Prompt library buttons
  if (ENABLE_PROMPT_LIBRARY) {
    if (!document.getElementById("library-buttons")) {
      const libraryGroup = document.createElement("div");
      libraryGroup.id = "library-buttons";
      libraryGroup.className = "mt-2";
      libraryGroup.innerHTML = `
                <button type="button" class="btn btn-sm btn-outline-info" onclick="showPromptLibrary()">📚 View Library</button>
                <button type="button" class="btn btn-sm btn-outline-success" onclick="saveToLibrary()">💾 Save to Library</button>
            `;
      advancedOptions.appendChild(libraryGroup);
      advancedOptions.style.display = "block";
    }
  }

  // Feedback evaluation buttons
  if (ENABLE_FEEDBACK_EVALUATION) {
    // This can be added as inline feedback buttons on messages
    // For now, we'll just enable the feature flag
  }

  // Show the toggle button if we have any options
  toggleBtn = document.getElementById("advanced-options-toggle");
  if (toggleBtn && advancedOptions.children.length > 0) {
    toggleBtn.style.display = "inline-block";
    // If we have options, show the advanced options panel by default
    advancedOptions.style.display = "block";
  } else if (advancedOptions.children.length === 0) {
    // Hide toggle button if no options
    if (toggleBtn) {
      toggleBtn.style.display = "none";
    }
  }
}

// Export conversation function
async function exportConversation() {
  if (!ENABLE_CONVERSATION_EXPORT) return;

  try {
    const response = await fetch(
      `${API_BASE_URL}/export/${sessionId}?format=markdown`
    );
    if (response.ok) {
      const text = await response.text();
      const blob = new Blob([text], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation-${sessionId}-${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      addErrorMessage("Error exporting conversation");
    }
  } catch (error) {
    console.error("Export error:", error);
    addErrorMessage("Error exporting conversation");
  }
}

// Prompt library functions
async function showPromptLibrary() {
  if (!ENABLE_PROMPT_LIBRARY) return;

  try {
    const response = await fetch(`${API_BASE_URL}/library`);
    if (response.ok) {
      const data = await response.json();
      if (data.entries && data.entries.length > 0) {
        const libraryHtml = data.entries
          .map(
            (entry) => `
          <div class="card mb-2">
            <div class="card-body">
              <h6>${entry.prompt.substring(0, 50)}...</h6>
              <p class="small text-muted">Category: ${
                entry.category || "N/A"
              }</p>
              <p class="small">Tags: ${entry.tags.join(", ") || "None"}</p>
            </div>
          </div>
        `
          )
          .join("");

        const modal = document.createElement("div");
        modal.className = "modal fade show";
        modal.style.display = "block";
        modal.innerHTML = `
          <div class="modal-dialog modal-lg">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title">Prompt Library (${data.count} entries)</h5>
                <button type="button" class="btn-close" onclick="this.closest('.modal').remove()"></button>
              </div>
              <div class="modal-body">
                ${libraryHtml}
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" onclick="this.closest('.modal').remove()">Close</button>
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(modal);
      } else {
        alert("Prompt library is empty. Save some prompts first!");
      }
    } else {
      addErrorMessage("Error loading prompt library");
    }
  } catch (error) {
    console.error("Library error:", error);
    addErrorMessage("Error loading prompt library");
  }
}

async function saveToLibrary() {
  if (!ENABLE_PROMPT_LIBRARY) return;

  const promptText =
    messageInput.value.trim() || window.prompt("Enter the prompt to save:");
  if (!promptText) return;

  const optimizedPrompt =
    window.prompt("Enter the optimized version (or leave empty):") ||
    promptText;
  const category = window.prompt("Enter category (optional):") || null;
  const tagsInput =
    window.prompt("Enter tags (comma-separated, optional):") || "";
  const tags = tagsInput
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t);

  try {
    const entry = {
      id: `prompt-${Date.now()}`,
      prompt: promptText,
      optimized_prompt: optimizedPrompt,
      tags: tags,
      category: category,
      timestamp: new Date().toISOString(),
    };

    const response = await fetch(`${API_BASE_URL}/library`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(entry),
    });

    if (response.ok) {
      const successMsg = document.createElement("div");
      successMsg.className =
        "alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3";
      successMsg.style.zIndex = "9999";
      successMsg.innerHTML = `
        <strong>Saved!</strong> Prompt saved to library.
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
      `;
      document.body.appendChild(successMsg);
      setTimeout(() => successMsg.remove(), 3000);
    } else {
      addErrorMessage("Error saving to library");
    }
  } catch (error) {
    console.error("Save error:", error);
    addErrorMessage("Error saving to library");
  }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadCharacterInfo();
  initializeUI();
  messageInput.focus();
});
