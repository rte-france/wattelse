// variables related to the page
const chatHistory = document.querySelector('.chat-history');
const userInput = document.querySelector('.input-field');
const sendButton = document.querySelector('.send-button');

const documentPanel = document.querySelector('.document-panel');
const tabs = documentPanel.querySelectorAll('.tab');
const contentSections = documentPanel.querySelectorAll('.content');

const extractList = document.getElementById('extract-list');
const documentList = document.querySelector('.document-list');
const selectAllCheckbox = document.getElementById('select-all');

// variables related to Django templates
const userName =  JSON.parse(document.getElementById('user_name').textContent);
const sessionId =  JSON.parse(document.getElementById('session_id').textContent);
const availableDocs = JSON.parse(document.getElementById('available_docs').textContent);

// initialize layout
initializeLayout()

function initializeLayout(){

    // Initialization of listeners
    selectAllCheckbox.addEventListener('change', handleSelectAll);

    sendButton.addEventListener('click', () => {
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage)
            }
    });

    userInput.addEventListener('keydown', (event) => {
        if (event.keyCode === 13) { // Check if Enter key is pressed
            const userMessage = userInput.value.trim();
            if (userMessage) {
                handleUserMessage(userMessage)
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
            const targetContent = documentPanel.querySelector(`.content.${tab.dataset.content}`);
            targetContent.style.display = 'block';
        });
    });

    // Display of available documents
    availableDocs.forEach((document) =>{
        const listItem = createDocumentListItem(document, "description");
        documentList.appendChild(listItem);
    });

    // Select all documents by default
    selectAllCheckbox.click()

    // Welcome message
    createBotMessage("Bonjour "+userName+ "! Posez-moi des questions en lien avec les documents sélectionnés...", false);
}


function handleUserMessage(userMessage) {
    if (getSelectedDocuments().length === 0) {
        createErrorMessage("Merci de sélectionner au moins un document!")
        return
    }
    createUserMessage(userMessage);

    // Simulate bot response with a delay
    const waitDiv = document.createElement('div');
    waitDiv.id = 'wait-div';
    const botDiv = document.createElement('p');
    botDiv.classList.add('bot-message');
    botDiv.innerHTML='<i class="fa-solid fa-comment-dots fa-beat-fade">'
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

    fetch('', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            'message': userMessage,
            'selected_docs': JSON.stringify(getSelectedDocuments()),
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

function updateRelevantExtracts(relevant_extracts){
    extractList.innerHTML = ""
    relevant_extracts.forEach((extract) => {
        const listItem = createExtract(extract.content, extract.url);
        extractList.appendChild(listItem);
    });
}

function createUserMessage(message) {
    const userDiv = document.createElement('p');
    userDiv.classList.add('user-message');
    userDiv.textContent = message;
    chatHistory.appendChild(userDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}


function createBotMessage(message, showFeedbackSection = true) {
    const botDiv = document.createElement('p');
    botDiv.classList.add('bot-message');
    botDiv.textContent = message;
    chatHistory.appendChild(botDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message

    // Call the function to activate the "Extracts" tab
    activateExtractsTab();

    // Feedback section (modify based on your chosen approach)
    if (showFeedbackSection) {
        provideFeedback();
    }
}


function createErrorMessage(message) {
    const errorDiv = document.createElement('p');
    errorDiv.classList.add('error-message');
    errorDiv.textContent = message;
    chatHistory.appendChild(errorDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the latest message
}

function activateExtractsTab() {
  // Select all tabs using the documentPanel and the class 'tab'
  const tabs = documentPanel.querySelectorAll('.tab');

  // Find the tab with the data-content attribute set to "extracts"
  const extractsTab = Array.from(tabs).find(tab => tab.dataset.content === 'extracts');

  // If the extractsTab is found, remove the 'active' class from all tabs
  // and add the 'active' class to the extractsTab
  if (extractsTab) {
    tabs.forEach(tab => tab.classList.remove('active'));
    extractsTab.classList.add('active');

    // Show the content section corresponding to the extractsTab
    const targetContent = documentPanel.querySelector(`.content.${extractsTab.dataset.content}`);
    targetContent.style.display = 'block';

    // Hide any other content sections that might be visible
    const otherContents = documentPanel.querySelectorAll('.content:not(.extracts)');
    otherContents.forEach(content => content.style.display = 'none');
  }
}

function createExtract(text, sourceUrl) {
    const listItem = document.createElement('li');
    const paragraph = document.createElement('p');
    const link = document.createElement('a');

    paragraph.textContent = text;
    link.href = sourceUrl;
    link.target = '_blank';
    link.textContent = 'Source';
    link.classList.add('source-link'); // Optional styling class

    listItem.appendChild(paragraph);
    listItem.appendChild(link);

    return listItem;
}

function createDocumentListItem(title, description) {
  const listItem = document.createElement('li');
  const checkbox = document.createElement('input');
  const titleSpan = document.createElement('span');

  checkbox.type = 'checkbox';
  titleSpan.textContent = title;
  listItem.appendChild(checkbox);
  listItem.appendChild(titleSpan);
  // Add space using margin or padding
  titleSpan.style.marginRight = '1rem'; // Using margin for spacing

  if (description) {
    const descriptionElement = document.createElement('p');
    descriptionElement.textContent = description;
    listItem.appendChild(descriptionElement);
  }

  return listItem;
}

// Function to handle selecting/deselecting all documents
function handleSelectAll(event) {
  const isChecked = event.target.checked;
  documentList.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.checked = isChecked;
  });
}

function getSelectedDocuments() {
  const checkboxes = documentList.querySelectorAll('input[type="checkbox"]')
  const selectedDocNames = [];
  for (const checkbox of checkboxes) {
    if (checkbox.checked) {
      // Find the associated label element
      text = checkbox.closest('li').querySelector('span').textContent;
      if (text) {
          selectedDocNames.push(text);
      }
    }
  }
  return selectedDocNames;
}

//TODO!
function provideFeedback() {
    const feedbackSection = document.createElement('div');
    feedbackSection.classList.add('feedback-section');

    feedbackSection.innerHTML = `
          <button id="open-text-feedback">Text Feedback</button> 
          <span class="emoji-rating" style="margin-right: 1.5rem">👍</span>
          <span class="emoji-rating" style="margin-right: 1.5rem">👎</span>
        `;

    chatHistory.appendChild(feedbackSection);

    /*
    emojiRatings.forEach((emoji) => {
        emoji.addEventListener('click', (event) => {
            const feedback = event.target.textContent; // Get the emoji text
            // TODO
            // Process emoji feedback (implement logic as needed)
        });
    });
    */
}
