import {
  showPopup,
  uuid4,
  unsecuredCopyToClipboard,
} from "../../../static/js/utils.js";

// Global variables
export const chatInput = document.querySelector(".input-field");
export const chatConversation = document.querySelector(".chat-conversation");

export const csrfmiddlewaretoken = document.querySelector(
  "[name=csrfmiddlewaretoken]"
).value;
export const md = markdownit({
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        // Highlight the code with language specified
        const highlightedCode = hljs.highlight(str, { language: lang }).value;
        // Wrap it in a <pre><code> block with hljs class to apply theme styles
        return `<pre class="hljs"><code class="${lang}">${highlightedCode}</code></pre>`;
      } catch (__) {}
    }

    // If no language is specified or highlighting fails, wrap code in hljs class anyway
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`;
  },
});

export const newConversationButton = document.querySelector(
  ".new-conversation-button"
);

const welcomeMessage = "Comment puis-je vous aider ?";

// Create a new conversation
export function newConversation() {
  // Remove active conversation
  removeActiveConversation();

  // Empty conversation
  chatConversation.innerHTML = "";

  // Set new conversation id
  chatConversation.id = uuid4();

  // Create welcome message
  createWelcomeMessage(welcomeMessage);

  // Focus on chat input
  chatInput.focus();
}

export function createWelcomeMessage() {
  chatConversation.innerHTML = `
    <div class="welcome-container">
        <div class="welcome-message">${welcomeMessage}</div>
    </div>
    `;
}

export function removeActiveConversation() {
  let activeButton = document.querySelector(".history-container li.active");
  if (activeButton) {
    activeButton.classList.remove("active");
  }
}

// Remove a welcome message
export function removeWelcomeMessage() {
  const welcomeMessage = document.querySelector(".welcome-container");
  if (welcomeMessage) {
    welcomeMessage.remove();
  }
}

export function createUserMessage(messageContent, messageId = null) {
  // Generate message ID if not provided
  if (!messageId) {
    messageId = uuid4();
  }
  const message = document.createElement("div");
  message.id = messageId;
  message.classList.add("user-message");
  message.innerHTML = messageContent;
  chatConversation.appendChild(message);
  return messageId;
}

export function addNewConversationToHistory(conversationId, messageContent) {
  let todayListHistory = document.getElementById("today-history");
  const tempDiv = document.createElement("div");
  tempDiv.innerHTML = `<li class="active" id="${conversationId}"></li>`;
  const newConversationItem = tempDiv.firstChild;
  newConversationItem.textContent = messageContent;
  newConversationItem.addEventListener("click", () => {
    handleConversationClick(newConversationItem);
  });
  todayListHistory.insertBefore(
    newConversationItem,
    todayListHistory.firstChild
  );
}

export function handleConversationClick(conversationHistoryElement) {
  // Get conversation id
  const conversationId = conversationHistoryElement.id;

  // Remove active class from current active conversation
  removeActiveConversation();

  // Add active class to clicked conversation
  conversationHistoryElement.classList.add("active");

  // Set conversation messages
  setConversationMessages(conversationId);

  // Scroll to top of conversation
  chatConversation.scrollTop = chatConversation.scrollHeight;
}

export function setConversationMessages(conversationId) {
  // Set new id to chatHisory element and empty it
  chatConversation.id = conversationId;
  chatConversation.innerHTML = "";

  // Get conversation messages
  fetch(`get_conversation_messages/?id=${conversationId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
  }).then((response) => {
    // Handle successful requests
    if (response.ok) {
      return response.json().then((data) => {
        const messages = data.messages;
        // Create a message for each turn in messages
        messages.forEach((message) => {
          if (message.role == "user") {
            createUserMessage(md.render(message.content), message.id);
          } else {
            createAssistantMessage(md.render(message.content), message.id);
          }
        });
      });
    } else {
      // Handle errors caught in python backend
      return (
        response
          .json()
          .then((data) => {
            showPopup(data.message, true);
          })
          // Handle uncaught errors
          .catch((error) => {
            showPopup("Erreur non interceptée", true);
          })
      );
    }
  });
}

export function createAssistantMessage(messageContent, messageId = null) {
  // Generate message ID if not provided
  if (!messageId) {
    messageId = uuid4();
  }

  // Assistant message container
  const assistantMessageContainer = document.createElement("div");
  assistantMessageContainer.id = messageId;
  assistantMessageContainer.classList.add("assistant-message-container");

  // Assistant message div
  const assistantMessageDiv = document.createElement("div");
  assistantMessageDiv.classList.add("assistant-message");
  assistantMessageDiv.innerHTML = messageContent;

  // Assistant message actions div
  const assistantMessageActions = createAssistantMessageActions(messageId);

  // Append divs to container
  assistantMessageContainer.appendChild(assistantMessageDiv);
  assistantMessageContainer.appendChild(assistantMessageActions);

  // Append container to conversation
  chatConversation.appendChild(assistantMessageContainer);
  return [messageId, assistantMessageDiv];
}

export function createAssistantMessageActions(messageId) {
  // Main div
  const assistantMessageActions = document.createElement("div");

  // Rating up button
  const ratingUpButton = document.createElement("button");
  ratingUpButton.classList.add("rating-button");
  const ratingUpIcon = document.createElement("img");
  ratingUpIcon.src = `/static/icons/thumbs-up.svg`;
  ratingUpIcon.classList.add("icon");
  ratingUpButton.appendChild(ratingUpIcon);
  ratingUpButton.addEventListener("click", () => {
    handleVote(true, messageId);
  });
  assistantMessageActions.appendChild(ratingUpButton);

  // Rating down button
  const ratingDownButton = document.createElement("button");
  ratingDownButton.classList.add("rating-button");
  const ratingDownIcon = document.createElement("img");
  ratingDownIcon.src = `/static/icons/thumbs-down.svg`;
  ratingDownIcon.classList.add("icon");
  ratingDownButton.appendChild(ratingDownIcon);
  ratingDownButton.addEventListener("click", () => {
    handleVote(false, messageId);
  });
  assistantMessageActions.appendChild(ratingDownButton);

  // Copy / paste button
  const copyButton = document.createElement("button");
  copyButton.classList.add("copy-button");
  const copyIcon = document.createElement("img");
  copyIcon.src = `/static/icons/copy.svg`;
  copyIcon.classList.add("icon");
  copyButton.appendChild(copyIcon);
  copyButton.addEventListener("click", () => {
    const messageElement = document.getElementById(messageId).firstChild;
    if (messageElement) {
      const messageText = messageElement.innerText;
      unsecuredCopyToClipboard(messageText);
      showPopup("Texte copié dans le presse-papier");
    }
  });
  assistantMessageActions.appendChild(copyButton);

  // Add FAQ button if in doc app
  if (window.location.pathname.includes("/doc")) {
    const faqButton = document.createElement("button");
    faqButton.classList.add("rating-button");
    const faqIcon = document.createElement("img");
    faqIcon.src = `/static/icons/faq.svg`;
    faqIcon.classList.add("icon");
    faqButton.appendChild(faqIcon);
    faqButton.addEventListener("click", () => {
      console.log("CLICK");
      handleFAQ(messageId);
    });
    assistantMessageActions.appendChild(faqButton);
  }

  assistantMessageActions.classList.add("assistant-message-actions");

  return assistantMessageActions;
}

function handleVote(rating, messageId) {
  fetch("handle_vote/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      message_id: messageId,
      rating: rating,
    }),
  }).then((response) => {
    // Handle successful requests
    if (response.ok) {
      showPopup("Merci pour votre vote !");
    } else {
      // Handle errors caught in python backend
      response
        .json()
        .then((data) => {
          showPopup(data.message, (error = true));
        })
        // Handle uncaught errors
        .catch((error) => {
          showPopup("Erreur non interceptée", (error = true));
        });
    }
  });
}

function handleFAQ(assistantMessageId) {
  // Get message just above assistant message
  let userMessageId =
    document.getElementById(assistantMessageId).previousElementSibling.id;
  let faqAnswer = prompt(
    "Veuillez saisir la réponse attendue. \nVotre réponse sera ajoutée à la FAQ du groupe.",
    ""
  );

  // send back answer
  if (faqAnswer) {
    fetch("handle_faq/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfmiddlewaretoken,
      },
      body: JSON.stringify({
        user_message_id: userMessageId,
        assistant_message_id: assistantMessageId,
        faq_answer: faqAnswer,
      }),
    }).then((response) => {
      // Handle successful requests
      if (response.ok) {
        showPopup("Réponse ajoutée à la FAQ");
      } else {
        // Handle errors caught in python backend
        response
          .json()
          .then((data) => {
            showPopup(data.message, (error = true));
          })
          // Handle uncaught errors
          .catch((error) => {
            showPopup("Erreur non interceptée", (error = true));
          });
      }
    });
  }
}
