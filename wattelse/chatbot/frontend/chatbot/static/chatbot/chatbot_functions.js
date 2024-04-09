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

const documentPanel = document.querySelector('.panel-container');
const tabs = documentPanel.querySelectorAll('.tab');
const contentSections = documentPanel.querySelectorAll('.tab-content');

const extractList = document.getElementById('extract-list');
const availableDocumentList = document.querySelector('.available-list');
const removalDocumentList = document.querySelector('.removal-list');
const conversationsList = document.querySelector('.conversations-list');

const selectAll = document.getElementById('select-all');

// variables related to the upload area
const dropArea = document.querySelector('.drop-section')
const uploadedSection = document.querySelector('.uploaded-section')
const uploadedListContainer = document.querySelector('.uploaded-list')
const fileSelector = document.querySelector('.file-selector')
const fileSelectorInput = document.querySelector('.file-selector-input')

// Functions for adding/removing users from group
const groupUsernamesList = document.getElementById("group-usernames-list");
const addUsersInputField = document.getElementById("add-users-input-field");

// variables related to Django templates
const userName =  JSON.parse(document.getElementById('user_name').textContent);
const conversationsIds = JSON.parse(document.getElementById('conversations_ids').textContent);
let availableDocs = JSON.parse(document.getElementById('available_docs').textContent);
const csrfmiddlewaretoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

// initialize layout
initializeLayout();

if (dropArea) {
    initializeUploadArea();
}


// ####################### Definition of functions ########################################
function initializeLayout(){

    // Initialization of listeners
    selectAll.addEventListener('click', handleSelectAll);

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

    // Initialization of chat history
    if (conversationsIds.length === 0) {
        newConversation();
    }
    else {
        selectConversation(conversationsIds[0]);
    };

    // Initialization of tabs
    tabs.forEach((tab) => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach((tab) => tab.classList.remove('active'));

            // Add active class to the clicked tab
            tab.classList.add('active');

            // Hide all content sections
            contentSections.forEach((content) => content.style.display = 'none');

            // Show the content section corresponding to the clicked tab
            const targetContent = documentPanel.querySelector(`.tab-content.${tab.dataset.content}`);
            targetContent.style.display = 'block';
        });
    });

    // Display available documents
    updateAvailableDocuments();

    // Initiliaze conversations list
    initializeConversationsList()

    // Select all documents by default
    selectAll.click();

    // Adding users
    if (addUsersInputField) {
        addUsersInputField.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                const newUsername = addUsersInputField.value;
                if (newUsername !== "") {
                    fetch('add_user_to_group/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: new URLSearchParams({
                            'csrfmiddlewaretoken': csrfmiddlewaretoken,
                            'new_username': newUsername,
                        })
                    })
                    .then(response => {
                        if (response.ok) {
                            addUserToUserList(newUsername);
                        }
                        else {
                            response.json().then(data => {
                                window.alert(data.error_message);
                            });
                        }
                    });
                    addUsersInputField.value = "";
                }
            }
        });
    }

    // Welcome message
    createBotMessage("Bonjour <b><span style='font-weight:bold;color:" +
        getComputedStyle(document.documentElement).getPropertyValue('--main-color')+";'>"+userName +
        "</span></b> ! Posez-moi des questions en lien avec les documents sélectionnés...",
        false, "documents");
}

function updateAvailableDocuments(){
    // Update the visible list of documents, after and before removal
    // gather the list of active documents before the change
    let previously_selected = getSelectedFileNames("available-list");

    availableDocumentList.innerHTML="";
    if (removalDocumentList) {
        removalDocumentList.innerHTML="";
    }
    
    // Display documents to be selected
    availableDocs.forEach((document) =>{
        const listItem = createDocumentListItem(document);
        availableDocumentList.appendChild(listItem);
    });

    // Display documents that can be removed
    availableDocs.forEach((document) =>{
        const listItem = createDocumentListItem(document);
        if (removalDocumentList) {
            removalDocumentList.appendChild(listItem);
        }
    });

    // intersection of previous selection with  new available docs
    newly_selected =  previously_selected.filter(x => availableDocs.includes(x));
    // reset previously checked docs
    setSelectedFileNames("available-list", newly_selected);
}

function handleUserMessage(userMessage) {
    if (getSelectedFileNames("available-list").length ===0){
        createErrorMessage("Aucun document sélectionné.")
        return
    }
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
    postUserMessageToRAG(userMessage);

    userInput.value = '';
}

function postUserMessageToRAG(userMessage) {
    if (userMessage === '') {
        return;
    }
    console.log("Posting user message: "+userMessage)
    // Remove "new" attribute of conversation
    document.getElementById(getSelectedConversationId()).classList.remove("conversation-button-new");
    fetch('query_rag/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': csrfmiddlewaretoken,
            'message': userMessage,
            'selected_docs': JSON.stringify(getSelectedFileNames("available-list")),
            'conversation_id': getSelectedConversationId(),
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('wait-div').remove();
        createBotMessage(data.answer);
        updateRelevantExtracts(data.relevant_extracts);
    })
    .catch(error => {
        createErrorMessage(error.message);
        console.error('There was a problem with the Fetch operation:', error);
    });
}

function deleteDocumentsInCollection(){
    const selectedFileNames = getSelectedFileNames("removal-list")
    if (selectedFileNames.length === 0  ) {
        alert("Aucun document à supprimer.");
    } else {
        // TODO: improve confirm popup look & feel
        if (confirm("Confirmer la suppression des fichiers sélectionnés?")){
            fetch('delete/', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: new URLSearchParams({
                    'csrfmiddlewaretoken': csrfmiddlewaretoken,
                    'selected_docs': JSON.stringify(selectedFileNames),
                })
            })
                .then(response => response.json())
                .then(data => {
                    // update available docs
                    availableDocs = data.available_docs
                    updateAvailableDocuments();
                })
                .catch(error => {
                    createErrorMessage(error.message);
                    console.error('There was a problem with the Fetch operation:', error);
                });
        }
    }
}

function updateRelevantExtracts(relevant_extracts){
    extractList.innerHTML = ""
    relevant_extracts.forEach((extract) => {
        const listItem = createExtract(extract.content, extract.metadata.url, extract.metadata.file_name);
        extractList.appendChild(listItem);
    });
}

function createWelcomeMessage() {
    chatHistory.innerHTML = `<div class="bot-message">Bonjour <span class="username">${userName}</span> ! Posez-moi une question sur les documents.</div>`
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

    // Feedback section (modify based on your chosen approach)
    if (showFeedbackSection) {
        // Call the function to activate the "Extracts" tab
        activateTab(nextTab);
        provideFeedback();
    }
}

function createErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-message');
    errorDiv.innerHTML = message;
    chatHistory.appendChild(errorDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function activateTab(tabName) {
  // Find the tab with the data-content attribute set to "extracts"
  const tabToBeActivated = Array.from(tabs).find(tab => tab.dataset.content === tabName);
  tabToBeActivated.click();
}

function createExtract(text, sourceUrl, fileName) {
    const listItem = document.createElement('li');
    const paragraph = document.createElement('p');
    const link = document.createElement('a');
    const horizontalLine = document.createElement('hr');

    paragraph.textContent = text;
    link.href = sourceUrl;
    link.target = '_blank';
    link.textContent = 'Source : ' + fileName;
    link.classList.add('source-link'); // Optional styling class

    listItem.appendChild(paragraph);
    listItem.appendChild(horizontalLine);
    listItem.appendChild(link);

    return listItem;
}

function createDocumentListItem(title) {
  const listItem = document.createElement('li');
  listItem.innerHTML = `
        <div class="col">
            ${iconSelector(title)}
        </div>
        <div class="col">
            <div class="file-name">
                <div title="${title}" class="name">${title}</div>
            </div>
        </div>
        <div class="col">
            <svg xmlns="http://www.w3.org/2000/svg" class="tick" height="20" width="20"><path d="m8.229 14.438-3.896-3.917 1.438-1.438 2.458 2.459 6-6L15.667 7Z"/></svg>
        </div>
  `
  // Add click event listener
  listItem.addEventListener('click', () => {
    listItem.classList.toggle('selected');
  });
  return listItem;
}

// Function to handle selecting/deselecting all documents
function handleSelectAll(event) {
  const available = availableDocumentList.querySelectorAll('li');
  let unselect = false;
  if (available.length === availableDocumentList.querySelectorAll('li.selected').length){
      unselect = true;
  }
  available.forEach((item) => {
      if (unselect)
          item.classList.remove('selected');
      else
          item.classList.add('selected') ;
  });
}

function getSelectedFileNames(listName) {
  const selectedItems = document.querySelectorAll("." + listName + " li.selected .col .file-name .name");
  const names = [];
  for (const item of selectedItems) {
    names.push(item.textContent.trim());
  }
  return names;
}

function setSelectedFileNames(listName, textsToSelect) {
  const selectedItems = document.querySelectorAll("." + listName + " li .col .file-name .name");
  for (const item of selectedItems) {
      const textItem = item.textContent.trim();
      const selected = textsToSelect.includes(textItem);
      const liElement = item.parentElement.parentElement.parentElement
      if (selected)
          liElement.classList.add('selected');
      else
          liElement.classList.remove('selected') ;
  }
}


function initializeUploadArea(){
    // upload files with browse button
    fileSelector.onclick = () => fileSelectorInput.click()
    fileSelectorInput.onchange = () => {
        [...fileSelectorInput.files].forEach((file) => {
            if (typeValidation(file.name)) {
                uploadFile(file)
            }
        })
    }

    // when file is over the drag area
    dropArea.ondragover = (e) => {
        e.preventDefault();
        [...e.dataTransfer.items].forEach((item) => {
            if (typeValidation(item.name)) {
                dropArea.classList.add('drag-over-effect')
            }
        })
    }
    // when file leave the drag area
    dropArea.ondragleave = () => {
        dropArea.classList.remove('drag-over-effect')
    }
    // when file drop on the drag area
    dropArea.ondrop = (e) => {
        e.preventDefault();
        dropArea.classList.remove('drag-over-effect')
        if (e.dataTransfer.items) {
            [...e.dataTransfer.items].forEach((item) => {
                if (item.kind === 'file') {
                    const file = item.getAsFile();
                    if (typeValidation(file.name)) {
                        uploadFile(file)
                    }
                }
            })
        } else {
            [...e.dataTransfer.files].forEach((file) => {
                if (typeValidation(file.name)) {
                    uploadFile(file)
                }
            })
        }
    }
}

// check the file type
function typeValidation(filename) {
    const fileExt = filename.split('.').pop().toLowerCase();
    return fileExt === "pdf" || fileExt === "docx" || fileExt === "pptx" || fileExt === "xlsx" || fileExt === "html"
        || fileExt === "htm" || fileExt === "md" || fileExt === "csv" || fileExt === "txt";
}

// upload file function
function uploadFile(file) {
    uploadedSection.style.display = 'block'
    const li = document.createElement('li');
    li.classList.add('in-prog')
    li.innerHTML = `
        <div class="col">
            ${iconSelector(file.name)}
        </div>
        <div class="col">
            <div class="file-name">
                <div class="name">${file.name}</div>
                <span>0%</span>
            </div>
            <div class="file-progress">
                <span></span>
            </div>
            <div class="file-size">${(file.size / (1024 * 1024)).toFixed(2)} MB</div>
        </div>
        <div class="col">
            <svg xmlns="http://www.w3.org/2000/svg" class="cross" height="20" width="20"><path d="m5.979 14.917-.854-.896 4-4.021-4-4.062.854-.896 4.042 4.062 4-4.062.854.896-4 4.062 4 4.021-.854.896-4-4.063Z"/></svg>
            <svg xmlns="http://www.w3.org/2000/svg" class="tick" height="20" width="20"><path d="m8.229 14.438-3.896-3.917 1.438-1.438 2.458 2.459 6-6L15.667 7Z"/></svg>
        </div>
    `
    uploadedListContainer.prepend(li)
    const http = new XMLHttpRequest();
    const data = new FormData();
    data.append('file', file)
    data.append('csrfmiddlewaretoken', csrfmiddlewaretoken)
    http.onload = () => {
        li.classList.add('complete')
        li.classList.remove('in-prog')
    }
    http.upload.onprogress = (e) => {
        const percent_complete = (e.loaded / e.total) * 100;
        li.querySelectorAll('span')[0].innerHTML = Math.round(percent_complete) + '%'
        li.querySelectorAll('span')[1].style.width = percent_complete + '%'
    }
    http.onreadystatechange = function() {
        if (http.readyState === XMLHttpRequest.DONE) {
            if (http.status >= 200 && http.status < 300) {
                availableDocs = JSON.parse(http.responseText)["available_docs"]
                updateAvailableDocuments();
            } else {
                alert(JSON.parse(http.responseText)["error_message"])
                li.remove()
            }
        }
    }
    http.open('POST', '/upload/', true)
    http.send(data)
    li.querySelector('.cross').onclick = () => http.abort()
    http.onabort = () => li.remove()
}

// find icon for file
function iconSelector(filename) {
    const pdfIcon = '<i class="fa-solid fa-file-pdf fa-xl" style="color: #ff0000;"></i>'
    const docxIcon = '<i class="fa-solid fa-file-word fa-xl" style="color: #2d2dff;"></i>'
    const xlsxIcon = '<i class="fa-solid fa-file-excel fa-xl" style="color: #008040;"></i>'
    const pptxIcon = '<i class="fa-solid fa-file-powerpoint fa-xl" style="color: #df0000;"></i>'
    const csvIcon = '<i class="fa-solid fa-file-csv fa-xl" style="color: #63E6BE;"></i>'
    const htmlIcon = '<i class="fa-solid fa-file-code fa-xl" style="color: #400080;"></i>'
    const defaultIcon = '<i class="fa-solid fa-file fa-xl" style="color: #000000;"></i>'
    const fileExt = filename.split('.').pop().toLowerCase();
    switch (fileExt) {
        case 'pdf':
            return pdfIcon
        case 'docx':
            return docxIcon
        case 'xlsx':
            return xlsxIcon
        case 'pptx':
            return pptxIcon
        case 'csv':
            return csvIcon
        case 'html':
            return htmlIcon
        case 'md':
            return htmlIcon
        default:
            return defaultIcon
    }
}


// Add new user to user list in users management tab
function addUserToUserList(username) {
    const listItem = document.createElement('li');
    listItem.id = `group_user_${username}`
    listItem.innerHTML = `
    <div class="col">
        <i class="fa-solid fa-user-secret fa-xl"></i>
    </div>
    <div class="col">
        ${username}
    </div>
    <div class="col">
        <button class="remove-user-button" onclick="removeUserFromGroup('${username}')"><i class="fa-solid fa-xmark fa-xl"></i></button>
    </div>
    `;
    groupUsernamesList.appendChild(listItem);
}


// Delete users
function removeUserFromGroup(userNameToDelete) {
    if (confirm(`Voulez-vous vraiment supprimer ${userNameToDelete} ?`)) {
        fetch('remove_user_from_group/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                'csrfmiddlewaretoken': csrfmiddlewaretoken,
                'username_to_delete': userNameToDelete,
            })
        })
        .then(response => {
            if (response.ok) {
                const userToDeleteItem = document.getElementById(`group_user_${userNameToDelete}`);
                userToDeleteItem.remove();
            }
            else {
                response.json().then(data => {
                    window.alert(data.error_message);
                });
            }
        });
    }
}

// Delete chat history
function deleteConversation() {
    const selectedConversationId = getSelectedConversationId()
    fetch('reset_history/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': csrfmiddlewaretoken,
            'conversation_id': selectedConversationId,
        })
    })
    .then(response => {
        document.getElementById(selectedConversationId).remove();
        newConversation();
    })
}

// Functions to collect user feedback
function provideFeedback() {
    const feedbackSection = document.createElement('div');
    feedbackSection.classList.add('feedback-section');

    feedbackSection.innerHTML = `
          <span class="emoji-rating"><span class="rating-great" title="Réponse parfaite"><i class="fa-solid fa-circle-check fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-ok" title="Réponse acceptable"><i class="fa-solid fa-circle-plus fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-missing" title="Réponse incomplète"><i class="fa-solid fa-circle-minus fa-xl"></i></span></span>
          <span class="emoji-rating"><span class="rating-wrong" title="Réponse fausse"><i class="fa-solid fa-circle-exclamation fa-xl"></i></span></span>
          <button id="open-text-feedback">Text Feedback</button> 
        `;

    chatHistory.appendChild(feedbackSection);

    // Add click event listeners to each emoji rating element
    const emojiRatings = document.querySelectorAll('.emoji-rating');
    emojiRatings.forEach(emojiRating => emojiRating.addEventListener('click', handleEmojiRatingClick));

    const textFeedbackButton = document.getElementById('open-text-feedback');
    textFeedbackButton.addEventListener('click', handleTextFeedbackClick);
}

function handleEmojiRatingClick(event) {
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

  let previousElement =  ratingElement.parentElement.previousSibling;
  while (previousElement && !previousElement.classList.contains('bot-message')) {
    previousElement = previousElement.previousSibling;
  }
  let bot_answer = "";
  if (previousElement) {
      bot_answer = previousElement.textContent;
  }

  previousElement =  ratingElement.parentElement.previousSibling;
  while (previousElement && !previousElement.classList.contains('user-message')) {
    previousElement = previousElement.previousSibling;
  }
  let user_question = "";
  if (previousElement) {
      user_question = previousElement.textContent;
  }

  // send feedback for processing
  if (feedbackName) {
    sendFeedback("send_short_feedback/", feedbackName, user_question, bot_answer);
  }
}

// Function to handle click on the text feedback button (optional)
function handleTextFeedbackClick(event) {
  const feedbackButton = event.currentTarget;
  feedbackButton.classList.add('selected');

  // Implement your logic for opening a text feedback form or modal here
  let feedback = prompt("Veuillez saisir la réponse attendue :", "");

  // Search for related messages (bot and user)
  let previousElement =  feedbackButton.parentElement.previousSibling;
  while (previousElement && !previousElement.classList.contains('bot-message')) {
    previousElement = previousElement.previousSibling;
  }
  let bot_answer = "";
  if (previousElement) {
      bot_answer = previousElement.textContent;
  }

  previousElement =  feedbackButton.parentElement.previousSibling;
  while (previousElement && !previousElement.classList.contains('user-message')) {
    previousElement = previousElement.previousSibling;
  }
  let user_question = "";
  if (previousElement) {
      user_question = previousElement.textContent;
  }

  // send back answer
  if (feedback){
      sendFeedback("send_long_feedback/", feedback, user_question, bot_answer);
  }
}


// Common function that sends feedback message
function sendFeedback(endpoint, feedback, user_message, bot_message){
    fetch(endpoint, {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({
              'csrfmiddlewaretoken': csrfmiddlewaretoken,
              'feedback': feedback,
              'user_message': user_message,
              'bot_message': bot_message,
          })
      })
          .then(response => {
              if (response.ok) {
                  console.info("Feedback reçu!");
              } else {
                  response.json().then(data => {
                      window.alert(data.error_message);
                  });
              }
          });
}

function selectConversation(conversationId) {
    fetch('select_conversation/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': csrfmiddlewaretoken,
            'conversation_id': conversationId,
        })
    })
    .then(response => response.json())
    .then(data => {
        const conversationItems = document.querySelectorAll(".conversation-button");
        conversationItems.forEach((button) => button.classList.remove("conversation-button-selected"));
        document.getElementById(conversationId).classList.add("conversation-button-selected");
        chatHistory.innerHTML = "";
        if (data.history) {
            data.history.forEach((message) => {
                if (message.role === "user") {
                    createUserMessage(message.content)
                }
                if (message.role === "assistant") {
                    createBotMessage(message.content, showFeedbackSection=false)
                }
            })
        }
        else {
            createWelcomeMessage();
        }
    });
}

function createConversationsListItem(conversationId) {
    const listItem = document.createElement('li');
    listItem.innerHTML = `
          <div class="col">OK</div>
          <div class="col">
              <div class="file-name">
                  <div class="name">${conversationId}</div>
              </div>
          </div>
          <div class="col">
              <svg xmlns="http://www.w3.org/2000/svg" class="tick" height="20" width="20"><path d="m8.229 14.438-3.896-3.917 1.438-1.438 2.458 2.459 6-6L15.667 7Z"/></svg>
          </div>
    `;
    listItem.classList.add("conversation-button", "conversation-button-new");
    listItem.id = conversationId;
    listItem.addEventListener('click', () => {
        selectConversation(conversationId);
    });
    conversationsList.appendChild(listItem);
  }

function newConversation() {
    const newConversationsButtons = document.querySelector(".conversation-button-new");
    if (newConversationsButtons) {
        selectConversation(newConversationsButtons.id);
    }
    else {
        // var newConversationButton = document.createElement("button");
        // newConversationButton.classList.add("conversation-button", "conversation-button-new");
        // const newUUID4 = uuid4();
        // newConversationButton.onclick = function() {
        //     selectConversation(newUUID4);
        // };
        // newConversationButton.innerHTML = newUUID4;
        // newConversationButton.id = newUUID4;
        const newUUID4 = uuid4();
        createConversationsListItem(newUUID4);
        selectConversation(newUUID4);
    }
}

function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

function getSelectedConversationId() {
    return document.querySelector(".conversation-button-selected").id;
}

function initializeConversationsList() {
    conversationsIds.forEach((conversationId) => {
        createConversationsListItem(conversationId);
    })
}