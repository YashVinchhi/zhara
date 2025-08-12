# UI Documentation

This document outlines the structure, components, and styling of the user interface.

## Overall Layout

The application features a two-column layout on larger screens, adapting to a single-column view on smaller devices.

- **Left Column (Larger Screens):** Houses the interactive 3D avatar viewport and settings panel.
- **Right Column (Larger Screens):** Contains the chat panel for user interaction with the AI.
- **Mobile Layout:** The chat panel is displayed below the 3D avatar viewport and settings panel.

## Interactive UI Components

### 3D Avatar Viewport

- Displays a 3D model of the AI avatar.
- Allows for camera interaction (zoom, pan, rotate).
- The avatar can exhibit subtle animations based on AI responses (e.g., blinking, head turns).

### Chat Panel

- Provides an interface for users to input text prompts and receive AI responses.
- Displays a history of the conversation.
- Features a text input field and a send button.

## Settings Customization

The settings panel allows users to personalize their experience.

- **Voice Selection:** Choose from a list of available AI voices for spoken responses.
- **Avatar Customization:** Options to change the appearance or model of the 3D avatar (if available).
- **LLM Model Selection:** Ability to select different underlying Large Language Models for the AI responses.

## Style Guidelines

### Colors

- **Primary Color:** `#007bff`
- **Background Color:** `#f8f9fa`
- **Accent Color:** `#28a745`

### Fonts

- **Body Font:** `Arial, sans-serif`
- **Headline Font:** `Georgia, serif`
- **Code Font:** `Courier New, monospace`

### Icons

- Icons should follow a clean, minimalist style.
- Prefer outline icons over filled icons.

### Animations

- Subtle animations should be used to enhance user experience without being distracting.
- Examples include smooth transitions, hover effects, and subtle avatar movements.