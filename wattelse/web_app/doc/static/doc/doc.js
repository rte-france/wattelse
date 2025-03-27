import {
  sendButton,
  chatInput,
  chatConversation,
  newConversation,
  createWelcomeMessage,
  newConversationButton,
  handleConversationClick,
} from "../../../static/js/gpt_doc_common.js";
import {
  handleUserMessage,
  getSelectedFileNames,
  tabs,
  contentSections,
  activateTab,
  updateAvailableDocuments,
  selectAll,
  addUsersInputField,
  addUserToGroup,
  sideBar,
  initializeUploadArea,
  handleSelectAll,
  trashButton,
  deleteDocumentsInCollection,
  removeUserFromGroup,
  manageUserPermissions,
  updateGroupSystemPrompt,
} from "./utils.js";
import { showPopup, uuid4 } from "../../../static/js/utils.js";

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
      if (getSelectedFileNames("available-list").length === 0) {
        showPopup("Aucun document sélectionné.", true);
      } else {
        handleUserMessage(messageContent);
      }
    }
  }
});

// Handle user message on send button click
sendButton.addEventListener("click", () => {
  const messageContent = chatInput.value.trim();
  if (messageContent) {
    if (getSelectedFileNames("available-list").length === 0) {
      showPopup("Aucun document sélectionné.", true);
    } else {
      handleUserMessage(messageContent);
    }
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

// Initialization of tabs
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    // Remove active class from all tabs
    tabs.forEach((tab) => tab.classList.remove("active"));

    // Add active class to the clicked tab
    tab.classList.add("active");

    // Hide all content sections
    contentSections.forEach((content) => (content.style.display = "none"));

    // Show the content section corresponding to the clicked tab
    const targetContent = sideBar.querySelector(
      `.tab-content.${tab.dataset.content}`
    );
    targetContent.style.display = "";
  });
});

// Activate documents tab by default
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

// Initialize upload area
initializeUploadArea();

// Handle select all button
selectAll.addEventListener("click", handleSelectAll);
selectAll.click();

// Trash button
trashButton.addEventListener("click", deleteDocumentsInCollection);

// Functions used in 'onClick' button parameters
window.removeUserFromGroup = removeUserFromGroup;
window.manageUserPermissions = manageUserPermissions;
window.updateGroupSystemPrompt = updateGroupSystemPrompt;
