/*
 * Copyright (c) 2024, RTE (https://www.rte-france.com)
 * See AUTHORS.txt
 * SPDX-License-Identifier: MPL-2.0
 * This file is part of Wattelse, a NLP application suite.
 */


// variables
const WELCOME_MSG = "Bonjour <span class='username'>"+userName+"</span> !";
const DISCLAIMER = "<i class=\"fa-solid fa-user-tie\"></i> WattElse GPT est reservé à un usage professionnel.<br>" +
    "<i class=\"fa-solid fa-sack-dollar\"></i> Chaque utilisation génère des coûts, merci de l'employer de façon responsable.<br>" +
    "<i class=\"fa-solid fa-lock\"></i> Veillez à ne pas divulguer d'informations sensibles (C3, C4).<br>" +
    "<i class=\"fa-solid fa-eye\"></i> Toutes les questions/réponses sont enregistrées à des fins d'analyse."

// initialize layout
initializeLayout();

// change default color scheme
setGPTColors(); // Jungle Green

// ####################### Definition of functions ########################################
function initializeLayout(){

    sendButton.addEventListener('click', () => {
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage)
            }
    });

    userInput.addEventListener('keydown', (event) => {
        if (event.key === "Enter" && !event.shiftKey) { // Check if Enter key is pressed
            event.preventDefault();
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage);
            }
        }
    });

    //  Create welcome message
    createWelcomeMessage(WELCOME_MSG, DISCLAIMER);
}

async function postUserMessageToChatBot(userMessage) {
    // Question timestamp
    const queryStartTimestamp = new Date();

    // Get conversation id
    const conversationId = chatHistory.id;

    // If new conversation: add conversation to history
    if (chatHistory.childElementCount < 2) {
        let todayListHistory = document.getElementById("today-history");
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = `<li class="active" id="${conversationId}" onclick="getConversationHistory(this, '${conversationId}')"></li>`;
        const newListItem = tempDiv.firstChild;
        newListItem.textContent = userMessage;
        todayListHistory.insertBefore(newListItem, todayListHistory.firstChild);

        // Remove old active conversation
        removeActiveConversation();
    }

    
    // Create bot waiting div
    const botDiv = createBotMessage('<i class="fa-solid fa-ellipsis fa-fade"></i>');
    botDiv.classList.add("waiting-div", "animate");
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Fetch response
    const response = await fetch('/query_gpt/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
            'message': userMessage,
            'conversation_id': conversationId,
        })
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
        botDiv.innerHTML = md.render(streamResponse);
    }

    // When streaming is done, show feedback section and save interaction
    const queryEndTimestamp = new Date();
    const answerDelay = queryEndTimestamp - queryStartTimestamp;

    // When streaming is done, show feedback section and save interaction
    botDiv.classList.remove("animate"); // remove generation animation
    provideFeedback(userMessage, streamResponse);

    saveInteraction(conversationId, userMessage, streamResponse, queryStartTimestamp.toISOString(), answerDelay)
}