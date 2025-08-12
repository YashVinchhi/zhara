class ChatUI {
    constructor() {
        this.initialized = false;
        this.chatHistory = [];
        this.initializeElements();
        this.setupEventListeners();
        this.setupThreeJS();
    }

    initializeElements() {
        this.inputContainer = document.querySelector('.input-container');
        this.chatContainer = document.querySelector('.chat-container');
        this.inputBox = document.querySelector('.input-box');
        this.fileInput = document.querySelector('#file-input');
        this.modelSelect = document.querySelector('#model-select');
        this.avatarSelect = document.querySelector('#avatar-select');
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

        // Get selected model
        const model = this.modelSelect.value;

        try {
            // Add user message to chat
            this.addMessageToChat(text, 'user');
            this.inputBox.value = '';

            console.log('Sending request:', { text, model });
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ 
                    text: text,
                    model: model
                })
            });

            const data = await response.json();
            
            // Add bot response to chat
            this.addMessageToChat(data.reply, 'bot');

            // Handle audio response
            if (data.audio_url) {
                this.playAudioResponse(data.audio_url);
            }

            // Handle viseme data for avatar animation
            if (data.viseme_url) {
                this.animateAvatar(data.viseme_url);
            }
        } catch (error) {
            console.error('Error:', error);
            this.addMessageToChat('Sorry, there was an error processing your request.', 'bot', true);
        }
    }

    addMessageToChat(text, type, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type} ${isError ? 'error' : ''}`;

        const content = document.createElement('div');
        content.className = 'message-content';

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.textContent = text;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();

        content.appendChild(messageText);
        content.appendChild(timeDiv);
        messageDiv.appendChild(content);

        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
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

    handleModelChange(event) {
        // Handle model selection
        console.log('Model changed:', event.target.value);
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
            const visemeData = await response.json();
            // Animate the avatar based on viseme data
            // This is a placeholder for the actual animation logic
        } catch (error) {
            console.error('Error loading viseme data:', error);
        }
    }

    playAudioResponse(audioUrl) {
        const audio = new Audio(audioUrl);
        audio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }
}

// Initialize the UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatUI = new ChatUI();
});
