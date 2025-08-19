class ChatUI {
    constructor() {
        this.initialized = false;
        this.chatHistory = [];
        this.pendingRequests = new Map();
        this.currentAudio = null;
        this.requestTimeout = 30000; // 30 seconds timeout
        this.currentSessionId = null; // Add session tracking
        this.sessions = []; // Store all sessions
        this.initializeElements();
        this.setupEventListeners();
        this.setupThreeJS();
        this.loadAvailableModels();
        this.loadSessions(); // Load existing sessions
        this.createNewSession(); // Start with a new session
    }

    initializeElements() {
        this.inputContainer = document.querySelector('.input-container');
        this.chatContainer = document.querySelector('.chat-container');
        this.inputBox = document.querySelector('.input-box');
        this.fileInput = document.querySelector('#file-input');
        this.modelSelect = document.querySelector('#model-select');
        this.avatarSelect = document.querySelector('#avatar-select');
        // Session management elements
        this.sidebar = document.querySelector('.sessions-sidebar');
        this.sessionsList = document.querySelector('.sessions-list');
        this.newChatBtn = document.querySelector('.new-chat-btn');
        this.sidebarToggle = document.querySelector('.sidebar-toggle');
    }

    setupEventListeners() {
        // Input box events
        this.inputBox.addEventListener('focus', () => this.transitionToChat());
        this.inputBox.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File upload
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // Model selection
        this.modelSelect.addEventListener('change', (e) => this.handleModelChange(e));

        // Avatar selection
        this.avatarSelect.addEventListener('change', (e) => this.handleAvatarChange(e));

        // Session management events
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.createNewSession());
        }

        if (this.sidebarToggle) {
            this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }

        // Avatar control events
        this.setupAvatarControls();

        // Send button click handler
        const sendBtn = document.querySelector('.send-btn');
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Mic button click handler
        const micBtn = document.querySelector('.mic-btn');
        if (micBtn) {
            micBtn.addEventListener('click', () => this.handleVoiceInput());
        }

        // Upload button click handler - fix for CSP compliance
        const uploadBtn = document.querySelector('.upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => {
                document.getElementById('file-input').click();
            });
        }
    }

    // Setup avatar control functionality
    setupAvatarControls() {
        const muteBtn = document.getElementById('mute-btn');
        const fullscreenBtn = document.getElementById('fullscreen-btn');

        if (muteBtn) {
            muteBtn.addEventListener('click', () => this.toggleMute());
        }

        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }
    }

    // Toggle mute functionality
    toggleMute() {
        const muteBtn = document.getElementById('mute-btn');
        const isCurrentlyMuted = muteBtn.classList.contains('active');

        if (isCurrentlyMuted) {
            muteBtn.classList.remove('active');
            muteBtn.title = 'Mute';
            muteBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                </svg>
            `;
            console.log('Audio unmuted');
        } else {
            muteBtn.classList.add('active');
            muteBtn.title = 'Unmute';
            muteBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                    <line x1="23" y1="9" x2="17" y2="15"></line>
                    <line x1="17" y1="9" x2="23" y2="15"></line>
                </svg>
            `;
            console.log('Audio muted');
        }
    }

    // Toggle fullscreen functionality
    toggleFullscreen() {
        const avatarCanvas = document.getElementById('avatar-canvas');
        const fullscreenBtn = document.getElementById('fullscreen-btn');

        if (!document.fullscreenElement) {
            if (avatarCanvas.requestFullscreen) {
                avatarCanvas.requestFullscreen();
                fullscreenBtn.classList.add('active');
                fullscreenBtn.title = 'Exit Fullscreen';
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                fullscreenBtn.classList.remove('active');
                fullscreenBtn.title = 'Fullscreen';
            }
        }
    }

    // Handle voice input (placeholder for future implementation)
    handleVoiceInput() {
        const micBtn = document.querySelector('.mic-btn');
        micBtn.classList.toggle('active');

        if (micBtn.classList.contains('active')) {
            this.updateAvatarStatus('listening');
            console.log('Voice recording started');
            // TODO: Implement actual voice recording
            micBtn.style.background = 'linear-gradient(135deg, #ff4757, #ff3742)';
            micBtn.title = 'Stop Recording';
        } else {
            this.updateAvatarStatus('idle');
            console.log('Voice recording stopped');
            micBtn.style.background = '';
            micBtn.title = 'Record voice message';
        }
    }

    // Session Management Methods
    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            // API returns an array of sessions
            this.sessions = Array.isArray(data) ? data : (data.sessions || []);
            this.renderSessionsList();
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    }

    async createNewSession() {
        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: `New Chat ${new Date().toLocaleString()}`
                })
            });

            const newSession = await response.json(); // API returns session directly, not wrapped

            // Add to sessions list
            this.sessions.unshift(newSession);

            // Switch to new session
            this.switchToSession(newSession.session_id);

            // Update UI
            this.renderSessionsList();

            console.log('Created new session:', newSession.session_id);
        } catch (error) {
            console.error('Error creating new session:', error);
        }
    }

    async switchToSession(sessionId) {
        if (this.currentSessionId === sessionId) return;

        this.currentSessionId = sessionId;

        // Clear current chat
        this.chatHistory = [];
        this.clearChatContainer();

        // Load session messages
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();

            // Display session messages
            if (data.messages) {
                for (const message of data.messages) {
                    this.addMessageToChat(message.user_message, 'user');
                    this.addMessageToChat(message.ai_response, 'bot');
                }
            }

            // Update active session in UI
            this.updateActiveSession(sessionId);

        } catch (error) {
            console.error('Error loading session:', error);
        }
    }

    async deleteSession(sessionId) {
        if (confirm('Are you sure you want to delete this conversation?')) {
            try {
                await fetch(`/api/sessions/${sessionId}`, {
                    method: 'DELETE'
                });

                // Remove from sessions list
                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                // If deleting current session, create a new one
                if (this.currentSessionId === sessionId) {
                    await this.createNewSession();
                } else {
                    this.renderSessionsList();
                }

            } catch (error) {
                console.error('Error deleting session:', error);
            }
        }
    }

    renderSessionsList() {
        if (!this.sessionsList) return;

        this.sessionsList.innerHTML = '';

        this.sessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = `session-item ${session.session_id === this.currentSessionId ? 'active' : ''}`;

            sessionItem.innerHTML = `
                <div class="session-content" data-session-id="${session.session_id}">
                    <div class="session-title">${session.title}</div>
                    <div class="session-info">
                        <span class="message-count">${session.message_count} messages</span>
                        <span class="last-updated">${new Date(session.last_updated).toLocaleDateString()}</span>
                    </div>
                </div>
                <div class="session-actions">
                    <button class="delete-session-btn" data-session-id="${session.session_id}" title="Delete conversation">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3,6 5,6 21,6"></polyline>
                            <path d="M19,6V20a2,2,0,0,1-2,2H7a2,2,0,0,1-2-2V6M8,6V4a2,2,0,0,1,2-2h4a2,2,0,0,1,2,2V6"></path>
                        </svg>
                    </button>
                </div>
            `;

            // Add event listeners instead of inline handlers
            const sessionContent = sessionItem.querySelector('.session-content');
            const deleteBtn = sessionItem.querySelector('.delete-session-btn');

            sessionContent.addEventListener('click', () => {
                this.switchToSession(session.session_id);
            });

            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering session switch
                this.deleteSession(session.session_id);
            });

            this.sessionsList.appendChild(sessionItem);
        });
    }

    updateActiveSession(sessionId) {
        const sessionItems = this.sessionsList.querySelectorAll('.session-item');
        sessionItems.forEach(item => {
            item.classList.remove('active');
            const sessionContent = item.querySelector('.session-content');
            if (sessionContent && sessionContent.dataset.sessionId === sessionId) {
                item.classList.add('active');
            }
        });
    }

    clearChatContainer() {
        if (this.chatContainer) {
            this.chatContainer.innerHTML = '';
        }
    }

    toggleSidebar() {
        if (this.sidebar) {
            this.sidebar.classList.toggle('collapsed');
        }
    }

    transitionToChat() {
        if (!this.initialized) {
            this.inputContainer.classList.remove('initial');
            this.inputContainer.classList.add('chat');
            this.initialized = true;
        }
    }

    async sendMessage() {
        const text = this.inputBox.value.trim();
        if (!text) return;

        // Create a session if we don't have one
        if (!this.currentSessionId) {
            await this.createNewSession();
        }

        const selectedModel = this.modelSelect.value;
        if (!selectedModel) {
            this.addMessageToChat('Please select a model first.', 'bot', true);
            return;
        }

        // Clear input and add user message to chat
        this.inputBox.value = '';
        this.addMessageToChat(text, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    model: selectedModel,
                    session_id: this.currentSessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Remove typing indicator
            this.removeTypingIndicator();

            if (data.reply) {
                this.addMessageToChat(data.reply, 'bot');

                // Update session after message
                await this.updateSessionAfterMessage();

                // Play audio if available
                if (data.audio_url) {
                    await this.playAudio(data.audio_url, data.viseme_url);
                }
            } else {
                throw new Error('No reply received from server');
            }
        } catch (error) {
            console.error('Error:', error);
            this.removeTypingIndicator();
            this.addMessageToChat(`Error: ${error.message}`, 'bot', true);
        }
    }

    async updateSessionAfterMessage() {
        // Refresh session info to update message count and last updated time
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();

            const sessions = Array.isArray(data) ? data : (data.sessions || []);
            if (sessions && sessions.length) {
                // Find and update the current session
                const currentSession = sessions.find(s => s.session_id === this.currentSessionId);
                if (currentSession) {
                    // Update sessions array
                    const sessionIndex = this.sessions.findIndex(s => s.session_id === this.currentSessionId);
                    if (sessionIndex !== -1) {
                        this.sessions[sessionIndex] = currentSession;
                    } else {
                        this.sessions.unshift(currentSession);
                    }

                    // Re-render sessions list
                    this.renderSessionsList();
                }
            }
        } catch (error) {
            console.error('Error updating session after message:', error);
        }
    }

    // Enhanced message handling with typing indicator
    addMessageToChat(text, type, isError = false) {
        // Remove any existing typing indicator
        this.removeTypingIndicator();

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type} ${isError ? 'error' : ''}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.textContent = text;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();

        messageContent.appendChild(messageText);
        messageContent.appendChild(timeDiv);
        messageDiv.appendChild(messageContent);

        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;

        // Update avatar status if it's a bot message
        if (type === 'bot') {
            this.updateAvatarStatus('speaking');
        }
    }

    // Add typing indicator
    showTypingIndicator() {
        this.removeTypingIndicator(); // Remove existing one first

        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';

        typingDiv.innerHTML = `
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span>Zhāra is thinking...</span>
        `;

        this.chatContainer.appendChild(typingDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;

        // Update avatar status
        this.updateAvatarStatus('thinking');
    }

    // Remove typing indicator
    removeTypingIndicator() {
        const existing = document.getElementById('typing-indicator');
        if (existing) {
            existing.remove();
        }
    }

    // Update avatar status
    updateAvatarStatus(status) {
        const avatarStatus = document.getElementById('avatar-status');
        const statusText = avatarStatus?.querySelector('span');

        if (avatarStatus && statusText) {
            // Remove all status classes
            avatarStatus.classList.remove('speaking', 'thinking');

            switch (status) {
                case 'speaking':
                    avatarStatus.classList.add('speaking');
                    statusText.textContent = 'Speaking';
                    break;
                case 'thinking':
                    statusText.textContent = 'Thinking';
                    break;
                case 'listening':
                    statusText.textContent = 'Listening';
                    break;
                default:
                    statusText.textContent = 'Idle';
            }
        }
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/stt', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.text) {
                this.inputBox.value = data.text;
                this.inputBox.focus();
            }
        } catch (error) {
            console.error('Error:', error);
            this.addMessageToChat('Error processing audio file.', 'bot', true);
        }
    }

    async loadAvailableModels() {
        try {
            console.log('Fetching available models...');
            this.modelSelect.innerHTML = '<option value="" disabled selected>Loading models...</option>';
            this.modelSelect.disabled = true;  // Disable while loading

            const response = await fetch('/api/models');
            let data;

            try {
                data = await response.json();
            } catch (parseError) {
                console.error('Error parsing models response:', parseError);
                throw new Error('Invalid response from server');
            }

            console.log('Received models:', data);

            // Clear existing options
            this.modelSelect.innerHTML = '';

            if (!data.models || data.models.length === 0) {
                throw new Error('No models available');
            }

            // Add models to dropdown
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;

                // Create display text with size if available
                let displayText = model.name;
                if (model.size) {
                    displayText += ` (${model.size})`;
                }
                option.textContent = displayText;

                // Add tooltip with more info if available
                if (model.modified_at) {
                    const modifiedDate = new Date(model.modified_at).toLocaleString();
                    option.title = `Last modified: ${modifiedDate}`;
                }

                this.modelSelect.appendChild(option);
                console.log(`Added model option: ${displayText}`);
            });

            // Try to restore last selected model or select the first one
            const lastSelectedModel = localStorage.getItem('lastSelectedModel');
            if (lastSelectedModel && [...this.modelSelect.options].some(opt => opt.value === lastSelectedModel)) {
                this.modelSelect.value = lastSelectedModel;
                console.log(`Restored last selected model: ${lastSelectedModel}`);
            } else if (this.modelSelect.options.length > 0) {
                this.modelSelect.value = this.modelSelect.options[0].value;
                console.log(`Selected default model: ${this.modelSelect.value}`);
            }

        } catch (error) {
            console.error('Error loading models:', error);
            // Add a default offline option if models can't be loaded
            this.modelSelect.innerHTML = '';
            const option = document.createElement('option');
            option.value = 'offline';
            option.textContent = 'Offline Mode';
            this.modelSelect.appendChild(option);

            // Show error message to user
            const errorMessage = document.createElement('div');
            errorMessage.className = 'message bot error';
            errorMessage.textContent = 'Unable to load models. Running in offline mode.';
            this.chatContainer.appendChild(errorMessage);
        } finally {
            this.modelSelect.disabled = false;  // Re-enable after loading/error
        }
    }

    handleModelChange(e) {
        const selectedModel = e.target.value;
        console.log('Selected model:', selectedModel);
        localStorage.setItem('lastSelectedModel', selectedModel);
        // Additional model-specific logic can be added here
    }

    handleAvatarChange(event) {
        // Handle avatar selection
        console.log('Avatar changed:', event.target.value);
        this.updateAvatar(event.target.value);
    }

    setupThreeJS() {
        // Initialize Three.js scene
        this.initThreeJS();
    }

    initThreeJS() {
        // Basic Three.js setup code here
        // This is a placeholder for the actual 3D avatar implementation
    }

    updateAvatar(avatarId) {
        // Update the 3D avatar model
        // This is a placeholder for the actual avatar update logic
    }

    async animateAvatar(visemeUrl) {
        try {
            const response = await fetch(visemeUrl);
            if (!response.ok) {
                throw new Error(`Failed to load viseme data: ${response.status}`);
            }
            const visemeData = await response.json();
            
            // Animation logic here
            if (this.avatarMesh) {
                // Implement viseme animation
                console.log('Animating avatar with viseme data:', visemeData);
            }
        } catch (error) {
            console.error('Error loading viseme data:', error);
            // Don't throw - avatar animation is non-critical
        }
    }

    async playAudioResponse(audioUrl) {
        try {
            // Stop any currently playing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            const audio = new Audio(audioUrl);
            this.currentAudio = audio;

            return new Promise((resolve, reject) => {
                audio.addEventListener('ended', () => {
                    this.currentAudio = null;
                    resolve();
                });
                audio.addEventListener('error', (e) => {
                    console.error('Audio playback error:', e);
                    this.currentAudio = null;
                    reject(e);
                });
                audio.play().catch(reject);
            });
        } catch (error) {
            console.error('Error playing audio:', error);
            throw error;
        }
    }

    async playAudio(audioUrl, visemeUrl) {
        try {
            // Check if audio is muted
            const muteBtn = document.getElementById('mute-btn');
            if (muteBtn && muteBtn.classList.contains('active')) {
                console.log('Audio is muted, skipping playback');
                return;
            }

            // Stop any currently playing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            const audio = new Audio(audioUrl);
            this.currentAudio = audio;

            // Start avatar animation if viseme data is available
            if (visemeUrl) {
                this.animateAvatar(visemeUrl);
            }

            // Update avatar status
            this.updateAvatarStatus('speaking');

            return new Promise((resolve, reject) => {
                audio.addEventListener('ended', () => {
                    this.currentAudio = null;
                    this.updateAvatarStatus('idle');
                    resolve();
                });
                audio.addEventListener('error', (e) => {
                    console.error('Audio playback error:', e);
                    this.currentAudio = null;
                    this.updateAvatarStatus('idle');
                    reject(e);
                });
                audio.play().catch(reject);
            });
        } catch (error) {
            console.error('Error playing audio:', error);
            this.updateAvatarStatus('idle');
            // Don't throw - audio playback is non-critical
        }
    }

    // Add cleanup method
    cleanup() {
        // Abort any pending requests
        for (const controller of this.pendingRequests.values()) {
            controller.abort();
        }
        this.pendingRequests.clear();

        // Stop any playing audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }
    }
}

// Initialize the UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatUI = new ChatUI();
});
