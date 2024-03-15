// variables related to the page
const chatHistory = document.querySelector('.chat-history');
const userInput = document.querySelector('.input-field');
const sendButton = document.querySelector('.send-button');

const documentPanel = document.querySelector('.document-panel');
const tabs = documentPanel.querySelectorAll('.tab');
const contentSections = documentPanel.querySelectorAll('.content');

const extractList = document.getElementById('extract-list');
const documentList = document.querySelector('.document-list');
const selectAllCheckbox = document.getElementById('select-all'); // Assuming an element with ID 'select-all' exists

// variables related to Django templates
const user_name =  JSON.parse(document.getElementById('user_name').textContent);
const session_id =  JSON.parse(document.getElementById('session_id').textContent);

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

// Initialisation
createBotMessage("Bonjour "+user_name+ "!", false);
createBotMessage("Session ID "+ session_id, false);


//TODO: A ADAPTER
// Simulate chat messages (replace with actual logic to communicate with your chatbot)
function handleUserMessage(userMessage) {
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
    postUserMessageToRAG(userMessage)

    //FIXME: remove waitDiv

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
        })
    })
    .then(response => response.json())
    .then(data => {
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

const fakeDocuments = [
  { title: 'NMT.pdf' },
  { title: 'Nouveau modes 2.docx', description: 'Un document important...' },
  // Add more documents as needed
];

fakeDocuments.forEach((document) => {
  const listItem = createDocumentListItem(document.title, document.description);
  documentList.appendChild(listItem);
});

// Javascript functions to fill dynamically divs
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

/*
function provideFeedback() {
  const feedbackSection = document.createElement('div');
  feedbackSection.classList.add('feedback-section');

  const textFeedbackButton = document.createElement('button');
  textFeedbackButton.id = "open-text-feedback";
  textFeedbackButton.textContent = "Text Feedback";

  // Add click event listener for text feedback button
  textFeedbackButton.addEventListener('click', openTextPopup);

  feedbackSection.appendChild(textFeedbackButton);

  const emojiRatingContainer = document.createElement('span');
  emojiRatingContainer.classList.add('emoji-rating-container');  // New container for emojis

  emojiRatingContainer.innerHTML = `
        <span class="emoji-rating" style="margin-right: 1.5rem"></span>
        <span class="emoji-rating" style="margin-right: 1.5rem"></span>
      `;

  feedbackSection.appendChild(emojiRatingContainer);

  chatHistory.appendChild(feedbackSection);

  // Function to open text popup (not defined in original code)
  function openTextPopup() {
    // Implement logic to create and display a text input popup here
    // This could involve creating a new element (e.g., div) with:
    // - A text area for user input
    // - A submit button to capture feedback
    // - Logic to close the popup after submit or cancellation
    alert('Text Feedback popup opens here!');  // Placeholder alert for now
  }

  emojiRatings.forEach((emoji) => {
    emoji.addEventListener('click', (event) => {
      const feedback = event.target.textContent;
      // TODO: Implement logic to handle emoji feedback
    });
  });
}
*/