/*
 * Copyright (c) 2024, RTE (https://www.rte-france.com)
 * See AUTHORS.txt
 * SPDX-License-Identifier: MPL-2.0
 * This file is part of Wattelse, a NLP application suite.
 */


const documentPanel = document.querySelector('.panel-container');
const tabs = documentPanel.querySelectorAll('.tab');
const contentSections = documentPanel.querySelectorAll('.tab-content');

const extractList = document.getElementById('extract-list');
const availableDocumentList = document.querySelector('.available-list');
const removalDocumentList = document.querySelector('.removal-list');

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
let availableDocs = JSON.parse(document.getElementById('available_docs').textContent);

// separator used for streaming
const SPECIAL_SEPARATOR = '¤¤¤¤¤';

// messages
const NO_EXTRACT_MSG = "Pas d'extraits pertinents dans les documents, le texte généré peut contenir des erreurs."
const WELCOME_MSG = "Bonjour <span class='username'>"+userName+"</span> !"

// Empty disclaimer for WattElse Doc
const DISCLAIMER = "";

// Message timeout
const timeout = 120000; // 120 seconds timeout

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
                if (getSelectedFileNames("available-list").length ===0){
                    createWarningMessage("Aucun document sélectionné.")
                } else {
                    handleUserMessage(userMessage)
                }
            }
    });

    userInput.addEventListener('keydown', (event) => {
        if (event.key === "Enter"&& !event.shiftKey) {
            event.preventDefault();
            const userMessage = userInput.value.trim();
            if (userMessage) {
                if (getSelectedFileNames("available-list").length ===0){
                    createWarningMessage("Aucun document sélectionné.")
                } else {
                    handleUserMessage(userMessage)
                }
            }
        }
    });

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
    activateTab("documents");

    // Display available documents
    updateAvailableDocuments();

    // Select all documents by default
    selectAll.click();

    // Adding users
    if (addUsersInputField) {
        addUsersInputField.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                const newUsername = addUsersInputField.value;
                if (newUsername !== "") {
                    addUserToGroup(newUsername);
                }
            }
        });
    }
    //  Create welcome message
    createWelcomeMessage(WELCOME_MSG);
}

async function postUserMessageToChatBot(userMessage) {
    // Handle too long response from backend
    const startTime = Date.now();

    // Question timestamp
    const queryStartTimestamp = new Date();

    // Get conversation id
    const conversationId = chatHistory.id;

    // Create bot waiting div
    // - Call first the function to activate the "Extracts" tab
    activateTab("extracts");
    const botDiv = createBotMessage('<i class="fa-solid fa-ellipsis fa-fade"></i>');
    botDiv.classList.add("waiting-div", "animate");
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Get selected files
    let selectedFiles = getSelectedFileNames("available-list");
    if (selectedFiles.length === availableDocs.length) {
        selectedFiles = [];
    }

    // Fetch response
    const response = await fetch('query_rag/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
            'message': userMessage,
            'selected_docs': selectedFiles,
            'conversation_id': conversationId,
        })
    });

    let isFirstChunk = true; // Track for first chunk processing containing relevant extracts
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let accumulatedData = "";
    let chunk;
    let noExtract = false;
    let streamResponse = "";

    do {
        // Handle too long response from backend
        if (Date.now() - startTime > timeout) {
            reader.cancel();
            botDiv.classList.remove("animate"); // remove generation animation
            showPopup("Réponse du serveur trop lente", error=true);
            return;
        }
        // Read streaming chunks
        chunk = await reader.read();

        // Handle last chunk
        if (chunk.done) {
            break;
        }

        // Must handle cases where multiple chunks are received in one read and
        // also the case in which long json chunks are received in multiple (incomplete json chunks)
        accumulatedData += decoder.decode(chunk.value);
        if (! isCompleteJSON(accumulatedData.split(SPECIAL_SEPARATOR)[0]))
        {
            continue;
        }
        const strJsonObjects = accumulatedData.split(SPECIAL_SEPARATOR).filter(elm => elm);
        accumulatedData="";
        const dataChunks = strJsonObjects.map(JSON.parse);

        // Handle other chunks
        dataChunks.forEach((json_chunk) => {
            if (isFirstChunk) {
                updateRelevantExtracts(json_chunk.relevant_extracts);
                if (json_chunk.relevant_extracts.length===0){
                    createWarningMessage(NO_EXTRACT_MSG);
                    noExtract = true;
                }
                isFirstChunk = false;
            } else {
                // Remove wainting div
                if (botDiv.classList.contains("waiting-div")) {
                    botDiv.innerHTML = "";
                    botDiv.classList.remove("waiting-div");
                }
                streamResponse += json_chunk.answer;
                botDiv.innerHTML = marked.parse(streamResponse);
            }
        });

    } while (true);

    // When streaming is done, show feedback section and save interaction
    const queryEndTimestamp = new Date();
    const answerDelay = queryEndTimestamp - queryStartTimestamp;

    botDiv.classList.remove("animate"); // remove generation animation
    if (!noExtract){
        provideFeedback(userMessage, streamResponse);
    }
    chatHistory.scrollTop = chatHistory.scrollHeight;

    saveInteraction(conversationId, userMessage, streamResponse, queryStartTimestamp.toISOString(), answerDelay)
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

// Helper function to check if the buffer contains a complete JSON object
function isCompleteJSON(buffer) {
  try {
    JSON.parse(buffer);
    return true;
  } catch (error) {
    return false;
  }
}

function deleteDocumentsInCollection(){
    const selectedFileNames = getSelectedFileNames("removal-list")
    if (selectedFileNames.length === 0  ) {
        showPopup("Aucun document sélectionné", error=true);
    }
    else {
        // TODO: improve confirm popup look & feel
        if (confirm("Confirmer la suppression des fichiers sélectionnés?")){
            fetch('delete/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfmiddlewaretoken,
                },
                body: JSON.stringify({
                    'selected_docs': selectedFileNames,
                })
            })
            .then((response) => {
                // Handle successful requests
                if (response.ok) {
                    response.json().then(data => {
                        // Update available docs
                        availableDocs = data.available_docs
                        updateAvailableDocuments();
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
    }
}

function updateRelevantExtracts(relevantExtracts){
    extractList.innerHTML = "";
    if (relevantExtracts.length>0) {
        relevantExtracts.forEach((extract) => {
            const url = `file_viewer/${extract["metadata"]["file_name"]}#page=${parseInt(extract["metadata"]["page"] ?? 0)+1}`
            const listItem = createExtract(extract.content, url, extract.metadata.file_name);
            extractList.appendChild(listItem);
        });
    } else {
         const listItem = createExtract("Aucun extrait trouvé !", "", "");
         extractList.appendChild(listItem);
    }
}

function typeWriter(botDiv, message, typingSpeed, callback) {
    let index = 0;

    const type = () => {
        if (index < message.length) {
            botDiv.innerHTML += message.charAt(index);
            index++;
            setTimeout(type, typingSpeed);
        } else if (callback) {
            callback();
        }
    };

    type();
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
    if (sourceUrl) {
        listItem.appendChild(horizontalLine);
        listItem.appendChild(link);
    }

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
        <button class="user-icon" title="Augmenter les permissions de l'utilisateur" onclick="manageUserPermissions('${username}', true)">
            <i class="fa-solid fa-user-plus"></i>
        </button>
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

// Upgrade or downgrade user permissions
function manageUserPermissions(username, upgrade) {
    // Define message to be displayed in the confirmation dialog
    let confirmMessage;
    if (upgrade) {
        confirmMessage = `Voulez-vous vraiment donner les droits administrateurs à ${username} ?`;
    }
    else {
        confirmMessage = `Voulez-vous vraiment enlever les droits administrateurs à ${username} ?`;
    }

    // Call back function if user confirms the action
    if (confirm(confirmMessage)) {
        fetch('/manage_superuser_permission/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfmiddlewaretoken,
            },
            body: JSON.stringify({
                'username': username,
                'upgrade': upgrade,
            })
        })
        .then(response => {
            // Handle successful requests
            if (response.ok) {
                response.json().then(data => {
                    // Change icon to match the new permissions
                    const userItem = document.getElementById(`group_user_${username}`);
                    if (upgrade) {
                        const userIcon = userItem.querySelector('.user-icon').parentElement;
                        userIcon.innerHTML = `
                        <button class="superuser-icon" title="Diminuer les permissions de l'utilisateur" onclick="manageUserPermissions('${username}', false)">
                            <i class="fa-solid fa-user-secret"></i>
                        </button>
                        `;
                    }
                    else {
                        const userIcon = userItem.querySelector('.superuser-icon').parentElement;
                        userIcon.innerHTML = `
                        <button class="user-icon" title="Augmenter les permissions de l'utilisateur" onclick="manageUserPermissions('${username}', true)">
                            <i class="fa-solid fa-user-plus"></i>
                        </button>
                        `;
                    }
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
}

// Add users to group
function addUserToGroup(newUsername) {
    fetch('add_user_to_group/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
            'new_username': newUsername,
        })
    })
    // Handle successful requests
    .then(response => {
        if (response.ok) {
            response.json().then(data => {
                addUserToUserList(newUsername);
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
    addUsersInputField.value = "";
}

// Delete users
function removeUserFromGroup(userNameToDelete) {
    if (confirm(`Voulez-vous vraiment supprimer ${userNameToDelete} ?`)) {
        fetch('remove_user_from_group/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfmiddlewaretoken,
            },
            body: JSON.stringify({
                'username_to_delete': userNameToDelete,
            })
        })
        .then((response) => {
            // Handle successful requests
            if (response.ok) {
                response.json().then(data => {
                    const userToDeleteItem = document.getElementById(`group_user_${userNameToDelete}`);
                    userToDeleteItem.remove();
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
}

function addFallingPumpkins() {
    const pumpkinCount = 15; // Number of pumpkins to display

    for (let i = 0; i < pumpkinCount; i++) {
      const pumpkin = document.createElement("div");
      pumpkin.className = "pumpkin";

      // Random horizontal start position for each pumpkin
      pumpkin.style.left = 10+Math.random() * 80 + "vw";
      pumpkin.style.animationDelay = `${Math.random() * 1}s`; // Random delay for falling

      document.body.appendChild(pumpkin);

      // Optional: Remove pumpkin after it reaches the bottom
      pumpkin.addEventListener('animationend', () => pumpkin.remove());
    }
  }

// Call the function on page load
window.onload = addFallingPumpkins;
