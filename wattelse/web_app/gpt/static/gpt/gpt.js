import { handleUserMessage } from "./utils.js";
import {
  sendButton,
  chatInput,
  chatConversation,
  newConversation,
  createWelcomeMessage,
  newConversationButton,
  handleConversationClick,
} from "../../../static/js/gpt_doc_common.js";
import { uuid4 } from "../../../static/js/utils.js";

// Add conversation id to chat conversation element on page load
chatConversation.id = uuid4();

// Show welcome message on page load
createWelcomeMessage();

// Make the chat input grow if input text is multiline
chatInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
});

// Handle user message on enter key press
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const messageContent = chatInput.value.trim();
    if (messageContent) {
      handleUserMessage(messageContent);
    }
  }
});

// Handle user message on send button click
sendButton.addEventListener("click", () => {
  const messageContent = chatInput.value.trim();
  if (messageContent) {
    handleUserMessage(messageContent);
  }
});

// Set conversation on conversation history click
document.querySelectorAll(".history-container li").forEach((conversation) => {
  conversation.addEventListener("click", (event) => {
    handleConversationClick(event.target);
  });
});

// New conversation button
newConversationButton.addEventListener("click", () => {
  newConversation();
});
