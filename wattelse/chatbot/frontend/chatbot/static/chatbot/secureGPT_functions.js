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

// initialize layout
initializeLayout();

// ####################### Definition of functions ########################################
function initializeLayout(){

    sendButton.addEventListener('click', () => {
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage)
            }
    });

    userInput.addEventListener('keydown', (event) => {
        if (event.key === "Enter") { // Check if Enter key is pressed
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage)
            }
        }
    });

    //  Create welcome message
    createWelcomeMessage();
}

function getChatHistory() {
    // Get the current chat history
    // Returns an array of messages in the format:
    // [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    let history = [];
    chatHistory.querySelectorAll(':scope > div').forEach(element => {
        if (element.classList.contains("user-message")) {
            history.push({
                "role": "user",
                "content": element.innerText.trim()
            });
        }
        else if (element.classList.contains("bot-message")) {
            history.push({
                "role": "assistant",
                "content": element.innerText.trim()
            });
        }
    });

    return history;
}

function handleUserMessage(userMessage) {
    // Remove welcome message if it exists
    removeWelcomeMessage();

    // Diplsay user message
    createUserMessage(userMessage);

    // Post Message to RAG
    const history = getChatHistory();
    postUserMessageToRAG(history);

    // Clean user input field
    userInput.value = '';
}


async function postUserMessageToRAG(history) {
    // Create bot waiting div
    const botDiv = createBotMessage('<i class="fa-solid fa-ellipsis fa-fade"></i>');
    botDiv.classList.add("waiting-div", "animate");
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Fetch response
    const response = await fetch('', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify(history)
    });

    const decoder = new TextDecoder();
    let streamResponse = "";

    for await (const chunk of response.body) {
        // Remove wainting div
        if (botDiv.classList.contains("waiting-div")) {
            botDiv.innerHTML = "";
            botDiv.classList.remove("waiting-div");
        }
        isFirstChunk = false;

        streamResponse += decoder.decode(chunk);
        botDiv.innerHTML = marked.parse(streamResponse);
    }

    // When streaming is done, show feedback section and save interaction
    botDiv.classList.remove("animate"); // remove generation animation
    chatHistory.scrollTop = chatHistory.scrollHeight;
}


function createUserMessage(message) {
    const userDiv = document.createElement('div');
    userDiv.classList.add('user-message');
    userDiv.innerHTML = message;
    chatHistory.appendChild(userDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function createBotMessage(message, nextTab="extracts") {
    const botDiv = document.createElement('div');
    botDiv.classList.add('bot-message');
    botDiv.innerHTML = message;
    chatHistory.appendChild(botDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message

    return botDiv;
}

function createWelcomeMessage() {
    chatHistory.innerHTML = `
    <div class="welcome-container">
        <div class="welcome-message">Bonjour <span class="username">${userName}</span> !</div>
    </div>
    `;
}

function removeWelcomeMessage() {
    const welcomeMessage = document.querySelector(".welcome-container");
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
}

function createErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.innerHTML = message;
    chatHistory.appendChild(errorDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function createWarningMessage(message) {
    const warningDiv = document.createElement('div');
    warningDiv.classList.add('warning-message');
    warningDiv.innerHTML = message;
    chatHistory.appendChild(warningDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

// Create a new conversation
function newConversation() {
    chatHistory.id = uuid4();
    createWelcomeMessage();
}

function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}