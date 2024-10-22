/*
 * Copyright (c) 2024, RTE (https://www.rte-france.com)
 * See AUTHORS.txt
 * SPDX-License-Identifier: MPL-2.0
 * This file is part of Wattelse, a NLP application suite.
 */


// variables
const WELCOME_MSG = "Bonjour <span class='username'>"+userName+"</span> !";

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

async function postUserMessageToChatBot(userMessage) {
    // Question timestamp
    const queryStartTimestamp = new Date();

    // Get conversation id
    const conversationId = chatHistory.id;

    // If new conversation: add conversation to history
    if (chatHistory.childElementCount < 2) {
        let todayListHistory = document.getElementById("today-history");
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = `<li class="active" id="${conversationId}" onclick="getConversationHistory(this, '${conversationId}')">${userMessage}</li>`;
        const newListItem = tempDiv.firstChild;
        todayListHistory.insertBefore(newListItem, todayListHistory.firstChild);
    }
    // Remove old active conversation
    removeActiveConversation()


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


function getConversationHistory(button, id) {
    fetch("/get_conversation_messages/", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
              'id': id,
          })
      })
      .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                // Write messages in user interface
                setConversationHistory(id, data.history);
                // Remove old active conversation
                removeActiveConversation();
                
                // Set current conversation as active
                button.classList.add("active");
            })
        }
        else {
            // Handle errors caught in python backend
            response.json().then(data => {
                showPopup(data.message, error=true);
            })
            // Handle uncaught errors
            .catch(error => {
                showPopup("Erreur non interceptée", error=true);
            })
        }
    })
}

function setConversationHistory(id, history) {
    // Set new id to chatHisory element and empty it
    chatHistory.id = id;
    chatHistory.innerHTML = "";

    // Save the current scroll position
    const scrollTop = chatHistory.scrollTop;

    // Create a message for each turn in history
    history.forEach((message) => {
        if (message.role=="user") {
            createUserMessage(message.content);
        }
        else {
            createBotMessage(marked.parse(message.content));
        }
    });

    // Restore the scroll position
    chatHistory.scrollTop = scrollTop;
}