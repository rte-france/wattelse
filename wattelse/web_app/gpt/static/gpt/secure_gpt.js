import {
  handleUserMessage,
  modelSelect,
  selectedModel,
  modeloptions,
} from "./utils.js";
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

// Model selection
// Toggle dropdown when clicking the selected area
selectedModel.addEventListener("click", (e) => {
  e.stopPropagation(); // Prevent from triggering document click
  modelSelect.classList.toggle("open");
});

// Handle option click
modeloptions.forEach((item) => {
  item.addEventListener("click", (e) => {
    e.stopPropagation(); // Prevent bubbling

    // Get the content from the clicked item
    const imgSrc = item.querySelector("img").src;
    const imgAlt = item.querySelector("img").alt;
    const labelText = item.querySelector("span").textContent;
    const value = item.dataset.value;

    // Update the selected element
    selectedModel.querySelector("img").src = imgSrc;
    selectedModel.querySelector("img").alt = imgAlt;
    selectedModel.querySelector("span").textContent = labelText;
    selectedModel.dataset.value = value;

    modelSelect.classList.remove("open");
  });
});

// Close if clicking outside
document.addEventListener("click", (e) => {
  if (!modelSelect.contains(e.target)) {
    modelSelect.classList.remove("open");
  }
});
