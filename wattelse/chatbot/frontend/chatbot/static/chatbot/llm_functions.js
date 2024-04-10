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

const selectAll = document.getElementById('select-all');

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


    // Welcome message
    createBotMessage("Bonjour <b><span style='font-weight:bold;color:" +
        getComputedStyle(document.documentElement).getPropertyValue('--main-color')+";'>"+userName +
        "</span></b> !",
        false, "documents");
}


function handleUserMessage(userMessage) {

    // Remove last feedback div
    const lastFeedbackDiv = document.querySelector('.feedback-section');
    if (lastFeedbackDiv) {
        lastFeedbackDiv.remove()
    }

    createUserMessage(userMessage);

    // Simulate bot response with a delay
    const waitDiv = document.createElement('div');
    waitDiv.id = 'wait-div';
    const botDiv = document.createElement('p');
    botDiv.classList.add('bot-message');
    botDiv.innerHTML='<i class="fa-solid fa-ellipsis fa-fade"></i>'
    waitDiv.appendChild(botDiv);
    chatHistory.appendChild(waitDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Post Message to RAG
    postUserMessageToLLM(userMessage);

    userInput.value = '';
}

function postUserMessageToLLM(userMessage) {
    if (userMessage === '') {
        return;
    }
    console.log("Posting user message: "+userMessage)

    fetch('', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': csrfmiddlewaretoken,
            'message': userMessage,
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('wait-div').remove();
        createBotMessage(data.answer);
    })
    .catch(error => {
        createErrorMessage(error.message);
        console.error('There was a problem with the Fetch operation:', error);
    });
}

function createUserMessage(message) {
    const userDiv = document.createElement('div');
    userDiv.classList.add('user-message');
    userDiv.innerHTML = message;
    chatHistory.appendChild(userDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function createBotMessage(message, showFeedbackSection = true, nextTab="extracts") {
    const botDiv = document.createElement('div');
    botDiv.classList.add('bot-message');
    botDiv.innerHTML = message
    chatHistory.appendChild(botDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function createErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.innerHTML = message;
    chatHistory.appendChild(errorDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}
