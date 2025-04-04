import {
  showPopup,
  uuid4,
  unsecuredCopyToClipboard,
} from "../../../static/js/utils.js";
import {
  chatInput,
  chatConversation,
  md,
  csrfmiddlewaretoken,
  removeActiveConversation,
  removeWelcomeMessage,
  createUserMessage,
  createAssistantMessage,
  addNewConversationToHistory,
} from "../../../static/js/gpt_doc_common.js";

// Functions
export function handleUserMessage(messageContent) {
  // Remove welcome message
  removeWelcomeMessage();

  // Reinitialize text area height
  chatInput.style.height = "auto";

  // Display user message
  const userMessageId = createUserMessage(md.render(messageContent));

  // Post Message to RAG
  postUserMessageToGPT(messageContent, userMessageId);

  chatInput.value = "";
}

async function postUserMessageToGPT(userMessage, userMessageId) {
  // Get conversation id
  const conversationId = chatConversation.id;

  // If new conversation: add conversation to history
  if (chatConversation.childElementCount < 2) {
    // Remove old active conversation
    removeActiveConversation();
    addNewConversationToHistory(conversationId, userMessage);
  }

  // Create assistant message with waiting div
  const [assistantMessageId, assistantMessageDiv] = createAssistantMessage(
    md.render("...")
  );
  assistantMessageDiv.classList.add("waiting");
  assistantMessageDiv.scrollIntoView({ behavior: "smooth" });

  // Fetch response
  const response = await fetch("query_gpt/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      message_id: userMessageId,
      content: userMessage,
    }),
  });

  const decoder = new TextDecoder();
  let streamResponse = "";

  for await (const chunk of response.body) {
    // Remove wainting div
    if (assistantMessageDiv.classList.contains("waiting")) {
      assistantMessageDiv.innerHTML = "";
      assistantMessageDiv.classList.remove("waiting");
    }

    streamResponse += decoder.decode(chunk);
    assistantMessageDiv.innerHTML = md.render(streamResponse);
  }

  // provideFeedback(messageContent, streamResponse);
  saveAssistantMessage(conversationId, assistantMessageId, streamResponse);

  // saveInteraction(conversationId, messageContent, streamResponse, queryStartTimestamp.toISOString(), answerDelay)
}

function saveAssistantMessage(conversationId, messageId, content) {
  fetch("save_assistant_message/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      message_id: messageId,
      content: content,
    }),
  }).then((response) => {
    // Handle successful requests
    if (response.ok) {
      // pass
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
