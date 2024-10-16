/*
 * Copyright (c) 2024, RTE (https://www.rte-france.com)
 * See AUTHORS.txt
 * SPDX-License-Identifier: MPL-2.0
 * This file is part of Wattelse, a NLP application suite.
 */

// variables related to the page
const chatHistory = document.querySelector('.chat-history');
const userInput = document.querySelector('.input-field');
const sendButton = document.querySelector('.send-button');

// variables related to Django templates
const userName =  JSON.parse(document.getElementById('user_name').textContent);
const csrfmiddlewaretoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

// Create a UUID to identify conversations
function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

//  Display a welcome message
function createWelcomeMessage(message) {
    chatHistory.innerHTML = `
    <div class="welcome-container">
        <div class="welcome-message">${message}</div>
    </div>
    `;
}

// Remove a welcome message
function removeWelcomeMessage() {
    const welcomeMessage = document.querySelector(".welcome-container");
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
}

// Display a user message
function createUserMessage(message) {
    const userDiv = document.createElement('div');
    userDiv.classList.add('user-message');
    userDiv.innerHTML = message;
    chatHistory.appendChild(userDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

// Display an error message
function createErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.innerHTML = message;
    chatHistory.appendChild(errorDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

// Display a warning message
function createWarningMessage(message) {
    const warningDiv = document.createElement('div');
    warningDiv.classList.add('warning-message');
    warningDiv.innerHTML = message;
    chatHistory.appendChild(warningDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

// Display bot message
function createBotMessage(message) {
    const botDiv = document.createElement('div');
    botDiv.classList.add('bot-message');
    botDiv.innerHTML = message;
    chatHistory.appendChild(botDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message

    return botDiv;
}

// Create a new conversation
function newConversation() {
    chatHistory.id = uuid4();
    createWelcomeMessage(WELCOME_MSG);
}