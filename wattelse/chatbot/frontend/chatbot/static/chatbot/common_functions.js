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

// Feedback vars
MAX_QUESTIONS_WITHOUT_FEEDBACK = 5
FEEDBACK_TOLERANCE = 5  // random value for reminder messages

// management of display timeout
let popupTimeout; // needed to avoid removing popup that is already removed

// initialization
chatHistory.id = uuid4();

///////////////////////// GENERIC FUNCTIONS ///////////////////////////////

// Create a UUID to identify conversations
function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

///////////////////////// DISPLAY FUNCTIONS ///////////////////////////////
// Display a pop-up
function showPopup(message, error = false) {
    // Delete any existing popups
    const existingPopups = document.querySelector('.popup');
    if (existingPopups) {
        existingPopups.remove(); // Remove the old popup immediately
        clearTimeout(popupTimeout); // Cancel any timeout if a new popup is created
    }

    // Create a new popup element
    const popup = document.createElement('div');
    popup.className = 'popup';
    popup.innerHTML = message;
    if (error) {
        popup.style.backgroundColor = "crimson";
    } else {
        popup.style.backgroundColor = "mediumseagreen";
    }
    document.body.appendChild(popup);

    // Show the popup with a fade-in effect
    setTimeout(() => {
        popup.style.display = 'block';
        popup.style.opacity = '1';
    }, 100); // Slight delay for smoother transition

    // Set a timeout to hide and remove the popup
    popupTimeout = setTimeout(() => {
        popup.style.opacity = '0'; // Fade out
        setTimeout(() => {
            popup.style.display = 'none';
            if (popup.parentNode) {
                document.body.removeChild(popup); // Remove the popup from DOM after fading out
            }
        }, 500); // Wait for fade-out to finish
    }, 3000);
}

// Display a welcome message
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

// Manage user input message
function handleUserMessage(userMessage) {
    // Remove last feedback div
    const lastFeedbackDiv = document.querySelector('.feedback-section');
    if (lastFeedbackDiv) {
        lastFeedbackDiv.remove()
    }

    // Remove welcome message if it exists
    removeWelcomeMessage();

    // Display user message
    createUserMessage(userMessage);

    // Post Message to RAG
    postUserMessageToChatBot(userMessage);

    userInput.value = '';
}

///////////////////////// LOG FUNCTIONS ///////////////////////////////

// Store logs into the database
function saveInteraction(conversationId, userMessage, botResponse, queryStartTimestamp, answerDelay) {
    fetch('/save_interaction/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
            'conversation_id': conversationId,
            'message': userMessage,
            'answer': botResponse,
            'question_timestamp': queryStartTimestamp,
            'answer_delay': answerDelay,
            'source_path': window.location.pathname, // allows to know from with page (RAG/GPT) the feedback comes from
        })
    })
    .then((response) => {
        // Handle successful requests
        if (response.ok) {
            // pass
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


// Common function that sends feedback message
function sendFeedback(endpoint, feedback, user_message, bot_message){
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
              'feedback': feedback,
              'user_message': user_message,
              'bot_message': bot_message,
              'source_path': window.location.pathname, // allows to know from with page (RAG/GPT) the feedback comes from
          })
      })
      .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                showPopup(data.message);
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

// Display user feedback collection buttons
function provideFeedback(userMessage, botMessage) {
    checkFeedbackCountSinceLastFeedback();

    const feedbackSection = document.createElement('div');
    feedbackSection.classList.add('feedback-section');

    feedbackSection.innerHTML = `
          <span class="emoji-rating"><span class="rating-great" title="Réponse parfaite"><i class="fa-solid fa-circle-check fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-ok" title="Réponse acceptable"><i class="fa-solid fa-circle-half-stroke fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-missing" title="Réponse incomplète"><i class="fa-solid fa-circle-minus fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-wrong" title="Réponse fausse"><i class="fa-solid fa-circle-exclamation fa-xl"></i></span></span>
          <button id="open-text-feedback">Réponse attendue</button> 
        `;

    chatHistory.appendChild(feedbackSection);

    // Add click event listeners to each emoji rating element
    const emojiRatings = document.querySelectorAll('.emoji-rating');
    emojiRatings.forEach(emojiRating => emojiRating.addEventListener('click', (event) => handleEmojiRatingClick(event, userMessage, botMessage)));

    const textFeedbackButton = document.getElementById('open-text-feedback');
    textFeedbackButton.addEventListener('click', (event) => handleTextFeedbackClick(event, userMessage, botMessage));
}


///////////////////////// FEEDBACK FUNCTIONS ///////////////////////////////

// counts the number of recent feedback for the current user
function checkFeedbackCountSinceLastFeedback() {
    fetch('/get_questions_count_since_last_feedback/?source_path=' + window.location.pathname,{ // allows to know from with page (RAG/GPT) the feedback comes from,
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
    })
    .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                if (data.count  > MAX_QUESTIONS_WITHOUT_FEEDBACK +  Math.floor(Math.random() * FEEDBACK_TOLERANCE)) {
                    createWarningMessage("N'oubliez pas de donner du feedback svp !")
                }
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

// Function to handle short feedback
function handleEmojiRatingClick(event, userMessage, botMessage) {
  // Get the parent element (emoji-rating) of the clicked element
  const ratingElement = event.currentTarget;

  // Highlight the clicked element and remove highlight from siblings
  const feedbackName = ratingElement.querySelector('span').className;
  ratingElement.querySelector('span').classList.add('selected');

  const ratings = ratingElement.parentElement.querySelectorAll('.emoji-rating');
  for (const r of ratings){
      if (r !== ratingElement){
          r.querySelector('span').classList.remove('selected');
      }
  }

  // send feedback for processing
  if (feedbackName) {
    sendFeedback("/send_short_feedback/", feedbackName, userMessage, botMessage);
  }
}

// Function to handle click on the text feedback button (optional)
function handleTextFeedbackClick(event, userMessage, botMessage) {
    const feedbackButton = event.currentTarget;
    feedbackButton.classList.add('selected');

    // Implement your logic for opening a text feedback form or modal here
    let feedback = prompt("Veuillez saisir la réponse attendue. \nVotre réponse sera ajoutée à la FAQ du groupe.", "");

    // send back answer
    if (feedback){
        sendFeedback("/send_long_feedback/", feedback, userMessage, botMessage);
    }
}