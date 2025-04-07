import {
  chatInput,
  removeWelcomeMessage,
  createUserMessage,
  createAssistantMessage,
  md,
  chatConversation,
  addNewConversationToHistory,
  removeActiveConversation,
  csrfmiddlewaretoken,
} from "../../../static/js/gpt_doc_common.js";
import { showPopup } from "../../../static/js/utils.js";

/// Global variables ///

export const sideBar = document.querySelector(".side-bar");
export const tabs = sideBar.querySelectorAll(".tab");

export const contentSections = sideBar.querySelectorAll(".tab-content");

const extractList = document.getElementById("extract-list");
const availableDocumentList = document.querySelector(".available-list");
const removalDocumentList = document.querySelector(".removal-list");

export const selectAll = document.getElementById("select-all");

const groupSystemPromptArea = document.getElementById("group-system-prompt");

// Variables related to the upload area
const dropArea = document.querySelector(".drop-section");
const uploadedSection = document.querySelector(".uploaded-section");
const uploadedListContainer = document.querySelector(".uploaded-list");
const fileSelector = document.querySelector(".file-selector");
const fileSelectorInput = document.querySelector(".file-selector-input");

// Trash button
export const trashButton = document.querySelector(".trash-button");

// Functions for adding/removing users from group
const groupUsernamesList = document.getElementById("group-usernames-list");
export const addUsersInputField = document.getElementById(
  "add-users-input-field"
);

// Message timeout
const timeout = 120000; // 120 seconds timeout

// variables related to Django templates
let availableDocs = JSON.parse(
  document.getElementById("available_docs").textContent
);

/// Functions ///

// Handle user message
export function handleUserMessage(messageContent) {
  // Remove welcome message
  removeWelcomeMessage();

  // Reinitialize text area height
  chatInput.style.height = "auto";

  // Display user message
  const userMessageId = createUserMessage(md.render(messageContent));

  // Post Message to RAG
  postUserMessageToRAG(messageContent, userMessageId);

  chatInput.value = "";
}

// Post user message to RAG handle streaming response and display it
async function postUserMessageToRAG(userMessage, userMessageId) {
  // Handle too long response from backend
  const startTime = Date.now();

  // Get conversation id
  const conversationId = chatConversation.id;

  // If new conversation: add conversation to history
  if (chatConversation.childElementCount < 2) {
    removeActiveConversation();
    addNewConversationToHistory(conversationId, userMessage);
  }

  // Go to extracts tab
  activateTab("extracts");

  // Create assistant message with waiting div
  const [assistantMessageId, assistantMessageDiv] = createAssistantMessage(
    md.render("...")
  );
  assistantMessageDiv.classList.add("waiting");
  assistantMessageDiv.scrollIntoView({ behavior: "smooth" });

  // Get selected files
  let selectedFiles = getSelectedFileNames("available-list");
  if (selectedFiles.length === availableDocs.length) {
    selectedFiles = [];
  }

  // Fetch response
  const response = await fetch("query_rag/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      message_id: userMessageId,
      content: userMessage,
      selected_docs: selectedFiles,
    }),
  });

  // Initialize variables to handle streaming response
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let isFirstChunk = true;
  let jsonBuffer = "";
  let streamResponse = "";
  let relevantExtracts;

  // Iterate over the streamed response
  try {
    while (true) {
      if (Date.now() - startTime > timeout) {
        throw new Error("Server response timed out");
      }

      const { value, done } = await reader.read();
      if (done) break;

      const chunkText = decoder.decode(value);

      if (isFirstChunk) {
        // Accumulate chunks for the JSON object containing relevant extracts
        jsonBuffer += chunkText;

        try {
          const firstChunk = JSON.parse(jsonBuffer);
          relevantExtracts = firstChunk.relevant_extracts;
          updateRelevantExtracts(relevantExtracts);

          isFirstChunk = false; // JSON is fully received, switch to plain text mode
          jsonBuffer = ""; // Clear buffer as JSON is processed
        } catch (err) {
          // Wait for more data if JSON parsing fails
          continue;
        }
      } else {
        if (assistantMessageDiv.classList.contains("waiting")) {
          assistantMessageDiv.innerHTML = "";
          assistantMessageDiv.classList.remove("waiting");
        }
        streamResponse += chunkText;
        assistantMessageDiv.innerHTML = md.render(streamResponse);
      }
    }
  } catch (error) {
    console.error(error.message);
    showPopup("Réponse du serveur trop lente", true);
  } finally {
    // Save interaction
    saveAssistantMessage(
      conversationId,
      assistantMessageId,
      streamResponse,
      relevantExtracts
    );
  }
}

// Displays a tab to the user
export function activateTab(tabName) {
  // Find the tab with the data-content attribute set to "extracts"
  const tabToBeActivated = Array.from(tabs).find(
    (tab) => tab.dataset.content === tabName
  );
  tabToBeActivated.click();
}

// Return the list of selected file names
export function getSelectedFileNames(listName) {
  const selectedItems = document.querySelectorAll(
    "." + listName + " li.selected .col .file-name .name"
  );
  const names = [];
  for (const item of selectedItems) {
    names.push(item.textContent.trim());
  }
  return names;
}

// Update the visible list of documents, after and before removal
// gather the list of active documents before the change
export function updateAvailableDocuments() {
  let previously_selected = getSelectedFileNames("available-list");

  availableDocumentList.innerHTML = "";
  if (removalDocumentList) {
    removalDocumentList.innerHTML = "";
  }

  // Display documents to be selected
  availableDocs.forEach((document) => {
    const listItem = createDocumentListItem(document);
    availableDocumentList.appendChild(listItem);
  });

  // Display documents that can be removed
  availableDocs.forEach((document) => {
    const listItem = createDocumentListItem(document);
    if (removalDocumentList) {
      removalDocumentList.appendChild(listItem);
    }
  });

  // intersection of previous selection with  new available docs
  const newly_selected = previously_selected.filter((x) =>
    availableDocs.includes(x)
  );
  // reset previously checked docs
  setSelectedFileNames("available-list", newly_selected);
}

// Add new user to user list in users management tab
function addUserToUserList(username) {
  const listItem = document.createElement("li");
  listItem.id = `group_user_${username}`;
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
export function manageUserPermissions(username, upgrade) {
  // Define message to be displayed in the confirmation dialog
  let confirmMessage;
  if (upgrade) {
    confirmMessage = `Voulez-vous vraiment donner les droits administrateurs à ${username} ?`;
  } else {
    confirmMessage = `Voulez-vous vraiment enlever les droits administrateurs à ${username} ?`;
  }

  // Call back function if user confirms the action
  if (confirm(confirmMessage)) {
    fetch("manage_superuser_permission/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfmiddlewaretoken,
      },
      body: JSON.stringify({
        username: username,
        upgrade: upgrade,
      }),
    }).then((response) => {
      // Handle successful requests
      if (response.ok) {
        response.json().then((data) => {
          // Change icon to match the new permissions
          const userItem = document.getElementById(`group_user_${username}`);
          if (upgrade) {
            const userIcon = userItem.querySelector(".user-icon").parentElement;
            userIcon.innerHTML = `
                        <button class="superuser-icon" title="Diminuer les permissions de l'utilisateur" onclick="manageUserPermissions('${username}', false)">
                            <i class="fa-solid fa-user-secret"></i>
                        </button>
                        `;
          } else {
            const userIcon =
              userItem.querySelector(".superuser-icon").parentElement;
            userIcon.innerHTML = `
                        <button class="user-icon" title="Augmenter les permissions de l'utilisateur" onclick="manageUserPermissions('${username}', true)">
                            <i class="fa-solid fa-user-plus"></i>
                        </button>
                        `;
          }
          showPopup(data.message);
        });
      } else {
        // Handle errors caught in python backend
        response
          .json()
          .then((data) => {
            showPopup(data.message, true);
          })
          // Handle uncaught errors
          .catch((error) => {
            showPopup("Erreur non interceptée", true);
          });
      }
    });
  }
}

// Add users to group
export function addUserToGroup(newUsername) {
  fetch("add_user_to_group/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      new_username: newUsername,
    }),
  })
    // Handle successful requests
    .then((response) => {
      if (response.ok) {
        response.json().then((data) => {
          if (!data.new_user_already_in_group) {
            addUserToUserList(newUsername);
          }
          showPopup(data.message);
        });
      } else {
        // Handle errors caught in python backend
        response
          .json()
          .then((data) => {
            showPopup(data.message, true);
          })
          // Handle uncaught errors
          .catch((error) => {
            showPopup("Erreur non interceptée", true);
          });
      }
    });
  addUsersInputField.value = "";
}

// Delete users
export function removeUserFromGroup(userNameToDelete) {
  if (confirm(`Voulez-vous vraiment supprimer ${userNameToDelete} ?`)) {
    fetch("remove_user_from_group/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfmiddlewaretoken,
      },
      body: JSON.stringify({
        username_to_delete: userNameToDelete,
      }),
    }).then((response) => {
      // Handle successful requests
      if (response.ok) {
        response.json().then((data) => {
          const userToDeleteItem = document.getElementById(
            `group_user_${userNameToDelete}`
          );
          userToDeleteItem.remove();
          showPopup(data.message);
        });
      } else {
        // Handle errors caught in python backend
        response
          .json()
          .then((data) => {
            showPopup(data.message, true);
          })
          // Handle uncaught errors
          .catch((error) => {
            showPopup("Erreur non interceptée", true);
          });
      }
    });
  }
}

// Upload area with drag and drop and browse button
export function initializeUploadArea() {
  // upload files with browse button
  fileSelector.onclick = () => fileSelectorInput.click();
  fileSelectorInput.onchange = () => {
    [...fileSelectorInput.files].forEach((file) => {
      if (typeValidation(file.name)) {
        uploadFile(file);
      }
    });
  };

  // when file is over the drag area
  dropArea.ondragover = (e) => {
    e.preventDefault();
    [...e.dataTransfer.items].forEach((item) => {
      if (typeValidation(item.name)) {
        dropArea.classList.add("drag-over-effect");
      }
    });
  };
  // when file leave the drag area
  dropArea.ondragleave = () => {
    dropArea.classList.remove("drag-over-effect");
  };
  // when file drop on the drag area
  dropArea.ondrop = (e) => {
    e.preventDefault();
    dropArea.classList.remove("drag-over-effect");
    if (e.dataTransfer.items) {
      [...e.dataTransfer.items].forEach((item) => {
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (typeValidation(file.name)) {
            uploadFile(file);
          }
        }
      });
    } else {
      [...e.dataTransfer.files].forEach((file) => {
        if (typeValidation(file.name)) {
          uploadFile(file);
        }
      });
    }
  };
}

// Create list item for the file
function createDocumentListItem(title) {
  const listItem = document.createElement("li");
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
  `;
  // Add click event listener
  listItem.addEventListener("click", () => {
    listItem.classList.toggle("selected");
  });
  return listItem;
}

// Find icon for file
function iconSelector(filename) {
  const pdfIcon =
    '<i class="fa-solid fa-file-pdf fa-xl" style="color: #ff0000;"></i>';
  const docxIcon =
    '<i class="fa-solid fa-file-word fa-xl" style="color: #2d2dff;"></i>';
  const xlsxIcon =
    '<i class="fa-solid fa-file-excel fa-xl" style="color: #008040;"></i>';
  const pptxIcon =
    '<i class="fa-solid fa-file-powerpoint fa-xl" style="color: #df0000;"></i>';
  const csvIcon =
    '<i class="fa-solid fa-file-csv fa-xl" style="color: #63E6BE;"></i>';
  const htmlIcon =
    '<i class="fa-solid fa-file-code fa-xl" style="color: #400080;"></i>';
  const pyIcon =
    '<i class="fa-brands fa-python fa-xl" style="color:  #3776ab;"></i>';
  const javaIcon =
    '<i class="fa-brands fa-java fa-xl" style="color: #4169e1;"></i>';
  const defaultIcon =
    '<i class="fa-solid fa-file fa-xl" style="color: #000000;"></i>';
  const fileExt = filename.split(".").pop().toLowerCase();
  switch (fileExt) {
    case "pdf":
      return pdfIcon;
    case "docx":
      return docxIcon;
    case "xlsx":
      return xlsxIcon;
    case "pptx":
      return pptxIcon;
    case "csv":
      return csvIcon;
    case "html":
      return htmlIcon;
    case "md":
      return htmlIcon;
    case "py":
      return pyIcon;
    case "java":
      return javaIcon;
    default:
      return defaultIcon;
  }
}

function setSelectedFileNames(listName, textsToSelect) {
  const selectedItems = document.querySelectorAll(
    "." + listName + " li .col .file-name .name"
  );
  for (const item of selectedItems) {
    const textItem = item.textContent.trim();
    const selected = textsToSelect.includes(textItem);
    const liElement = item.parentElement.parentElement.parentElement;
    if (selected) liElement.classList.add("selected");
    else liElement.classList.remove("selected");
  }
}

// check the file type
function typeValidation(filename) {
  const fileExt = filename.split(".").pop().toLowerCase();
  const validExtensions = [
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "htm",
    "md",
    "csv",
    "txt",
  ];
  const validConfigExtensions = ["toml", "cfg"];
  const validCodeExtensions = [
    "py",
    "cpp",
    "cc",
    "h",
    "java",
    "kotlin",
    "js",
    "php",
    "ts",
    "c",
    "sql",
    "latex",
  ];
  return (
    validExtensions.includes(fileExt) ||
    validConfigExtensions.includes(fileExt) ||
    validCodeExtensions.includes(fileExt)
  );
}

// upload file function
function uploadFile(file) {
  uploadedSection.style.display = "block";
  const li = document.createElement("li");
  li.classList.add("in-prog");
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
            <div class="file-size">${(file.size / (1024 * 1024)).toFixed(
              2
            )} MB</div>
        </div>
        <div class="col">
            <svg xmlns="http://www.w3.org/2000/svg" class="cross" height="20" width="20"><path d="m5.979 14.917-.854-.896 4-4.021-4-4.062.854-.896 4.042 4.062 4-4.062.854.896-4 4.062 4 4.021-.854.896-4-4.063Z"/></svg>
            <svg xmlns="http://www.w3.org/2000/svg" class="tick" height="20" width="20"><path d="m8.229 14.438-3.896-3.917 1.438-1.438 2.458 2.459 6-6L15.667 7Z"/></svg>
        </div>
    `;
  uploadedListContainer.prepend(li);
  const http = new XMLHttpRequest();
  const data = new FormData();
  data.append("file", file);
  data.append("csrfmiddlewaretoken", csrfmiddlewaretoken);
  http.onload = () => {
    li.classList.add("complete");
    li.classList.remove("in-prog");
  };
  http.upload.onprogress = (e) => {
    const percent_complete = (e.loaded / e.total) * 100;
    li.querySelectorAll("span")[0].innerHTML =
      Math.round(percent_complete) + "%";
    li.querySelectorAll("span")[1].style.width = percent_complete + "%";
  };
  http.onreadystatechange = function () {
    if (http.readyState === XMLHttpRequest.DONE) {
      if (http.status >= 200 && http.status < 300) {
        availableDocs = JSON.parse(http.responseText)["available_docs"];
        updateAvailableDocuments();
      } else {
        alert(JSON.parse(http.responseText)["error_message"]);
        li.remove();
      }
    }
  };
  http.open("POST", "upload/", true);
  http.send(data);
  li.querySelector(".cross").onclick = () => http.abort();
  http.onabort = () => li.remove();
}

// Function to handle selecting/deselecting all documents
export function handleSelectAll(event) {
  const available = availableDocumentList.querySelectorAll("li");
  let unselect = false;
  if (
    available.length ===
    availableDocumentList.querySelectorAll("li.selected").length
  ) {
    unselect = true;
  }
  available.forEach((item) => {
    if (unselect) item.classList.remove("selected");
    else item.classList.add("selected");
  });
}

function updateRelevantExtracts(relevantExtracts) {
  extractList.innerHTML = "";
  if (relevantExtracts.length > 0) {
    relevantExtracts.forEach((extract) => {
      const url = `file_viewer/${extract["metadata"]["file_name"]}#page=${
        parseInt(extract["metadata"]["page"] ?? 0) + 1
      }`;
      const listItem = createExtract(
        extract.content,
        url,
        extract.metadata.file_name
      );
      extractList.appendChild(listItem);
    });
  } else {
    const listItem = createExtract("Aucun extrait trouvé !", "", "");
    extractList.appendChild(listItem);
  }
}

function createExtract(text, sourceUrl, fileName) {
  const listItem = document.createElement("li");
  const paragraph = document.createElement("p");
  const link = document.createElement("a");
  const horizontalLine = document.createElement("hr");

  paragraph.textContent = text;
  link.href = sourceUrl;
  link.target = "_blank";
  link.textContent = "Source : " + fileName;
  link.classList.add("source-link");

  listItem.appendChild(paragraph);
  if (sourceUrl) {
    listItem.appendChild(horizontalLine);
    listItem.appendChild(link);
  }

  return listItem;
}

function saveAssistantMessage(
  conversationId,
  messageId,
  content,
  relevantExtracts = null
) {
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
      relevant_extracts: relevantExtracts,
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
          showPopup(data.message, true);
        })
        // Handle uncaught errors
        .catch((error) => {
          showPopup("Erreur non interceptée", true);
        });
    }
  });
}

export function deleteDocumentsInCollection() {
  const selectedFileNames = getSelectedFileNames("removal-list");
  if (selectedFileNames.length === 0) {
    showPopup("Aucun document sélectionné", true);
  } else {
    // TODO: improve confirm popup look & feel
    if (confirm("Confirmer la suppression des fichiers sélectionnés?")) {
      fetch("delete/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfmiddlewaretoken,
        },
        body: JSON.stringify({
          selected_docs: selectedFileNames,
        }),
      }).then((response) => {
        // Handle successful requests
        if (response.ok) {
          response.json().then((data) => {
            // Update available docs
            availableDocs = data.available_docs;
            updateAvailableDocuments();
            showPopup(data.message);
          });
        } else {
          // Handle errors caught in python backend
          response
            .json()
            .then((data) => {
              showPopup(data.message, true);
            })
            // Handle uncaught errors
            .catch((error) => {
              showPopup("Erreur non interceptée", true);
            });
        }
      });
    }
  }
}

export function updateGroupSystemPrompt() {
  const newGroupSystemPrompt = groupSystemPromptArea.value;
  fetch("update_group_system_prompt/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    body: JSON.stringify({
      group_system_prompt: newGroupSystemPrompt,
    }),
  }).then((response) => {
    // Handle successful requests
    if (response.ok) {
      response.json().then((data) => {
        showPopup(data.message);
      });
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
