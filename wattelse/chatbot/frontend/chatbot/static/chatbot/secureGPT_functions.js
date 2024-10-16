/*
 * Copyright (c) 2024, RTE (https://www.rte-france.com)
 * See AUTHORS.txt
 * SPDX-License-Identifier: MPL-2.0
 * This file is part of Wattelse, a NLP application suite.
 */


// variables
const WELCOME_MSG = "Bonjour <span class='username'>${userName}</span> !"

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
    createWelcomeMessage(WELCOME_MSG);
}

async function postUserMessageToRAG(history) {
    // Handle too long response from backend
    const startTime = Date.now();

    // Question timestamp
    const queryStartTimestamp = new Date();

    // Get conversation id
    const conversationId = chatHistory.id;

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
    const queryEndTimestamp = new Date();
    const answerDelay = queryEndTimestamp - queryStartTimestamp;

    // When streaming is done, show feedback section and save interaction
    botDiv.classList.remove("animate"); // remove generation animation
    provideFeedback(userMessage, streamResponse);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    saveInteraction(conversationId, userMessage, streamResponse, queryStartTimestamp.toISOString(), answerDelay)
}


