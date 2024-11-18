function showFaqPopup(id, question="", answer="", showDeleteButton=false, addToTable=true) {
    // Define popup HTML
    const popupHTML = `
    <div id="${id}" class="faq-popup-overlay">
        <div class="faq-popup-content">
            <div class="faq-popup-textareas-container">
                <textarea id="faqPopupQuestionArea" class="faq-popup-textarea">${question}</textarea>
                <textarea id="faqPopupAnswerArea" class="faq-popup-textarea">${answer}</textarea>
            </div>
        <button class="faq-popup-close-btn" id="closePopupBtn">X</button>
        <div class="faq-popup-bottom-btn-container">
            ${showDeleteButton ? '<button class="faq-popup-delete-button">Supprimer</button>' : ''}
            <button class="faq-popup-add-button">Envoyer</button>
        </div>
        </div>
    </div>
    `
    
    // Add it to document
    document.body.insertAdjacentHTML('beforeend', popupHTML);

    // Get the popup elements
    const popupOverlay = document.querySelector('.faq-popup-overlay');
    const closeBtn = document.querySelector('.faq-popup-close-btn');
    const questionTextarea = document.getElementById('faqPopupQuestionArea');
    const answerTextarea = document.getElementById('faqPopupAnswerArea');
    const addBtn = document.querySelector('.faq-popup-add-button');
    const deleteBtn = document.querySelector('.faq-popup-delete-button');

    // Function to close the popup and remove from DOM
    function closePopup() {
        popupOverlay.remove();
    }

    // Event listeners to close the popup
    closeBtn.addEventListener('click', closePopup);
    popupOverlay.addEventListener('click', (e) => {
        if (e.target === popupOverlay) closePopup();
    });

    // Event listener for add button
    addBtn.addEventListener('click', function(){
        addToFaq(popupOverlay.id, questionTextarea.value, answerTextarea.value, addToTable);
    });

    // Event listener for delete button
    if (deleteBtn) {
        deleteBtn.addEventListener('click', function(){
            // Confirm box
            if (confirm("Voulez-vous vraiment supprimer cet élément ?")) {
                deleteFromFaq(popupOverlay.id);
            };
        });
    }
}

function addToFaq(id, question, answer, addToTable=true){
    fetch("/add_to_faq/", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
              'id': id,
              'question': question,
              'answer': answer,
          })
      })
      .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                showPopup(data.message);
                // If a new object was created, add it to the FAQ else edit the existing one
                if (addToTable) {
                    if (data.created) {
                        // Add new line to FAQ table
                        const newLine = document.createElement("tr");
                        newLine.id = id;
                        const questionCol = document.createElement("td");
                        questionCol.innerHTML = question;
                        const answerCol = document.createElement("td");
                        answerCol.innerHTML = answer;
                        newLine.appendChild(questionCol);
                        newLine.appendChild(answerCol);
                        faqTable.appendChild(newLine);
                        // Add event listener for popup creation
                        newLine.addEventListener("click", () => {
                            showFaqPopup(id, question, answer, showDeleteButton=true);
                        });
                    } else {
                        // Edit existing line in FAQ
                        const existingLine = document.getElementById(id);
                        const fieldsToEdit = existingLine.querySelectorAll("td");
                        fieldsToEdit[0].innerHTML = question;
                        fieldsToEdit[1].innerHTML = answer;
    
                    }
                }
                // Close FAQ popup
                const popupOverlay = document.querySelector('.faq-popup-overlay');
                popupOverlay.remove();
                // Update FAQ
                updateFaq();
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

function deleteFromFaq(id){
    fetch("/delete_from_faq/", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
        body: JSON.stringify({
              'id': id,
          })
      })
      .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                showPopup(data.message);
                // Delete associated table row
                const lineToDelete = document.getElementById(id);
                lineToDelete.remove();
                // Close FAQ popup
                const popupOverlay = document.querySelector('.faq-popup-overlay');
                popupOverlay.remove();
                // Update FAQ
                updateFaq();
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

function updateFaq() {
    fetch("/update_faq/", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfmiddlewaretoken,
        },
      })
      .then((response) => {
        // Handle successful requests
        if (response.ok) {
            response.json().then(data => {
                showPopup(data.message);
            });
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


function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

function showPopup(message, error = false) {
    // Delete any existing popups
    const existingPopups = document.querySelector('.popup');
    if (existingPopups) {
        existingPopups.remove(); // Remove the old popup immediately
        clearTimeout(popupTimeout); // Cancel any timeout if a new popup is created
    }

    // Create a new popup element
    const popup = document.createElement('div');
    popup.className = 'popup';
    popup.innerHTML = message;
    if (error) {
        popup.style.backgroundColor = "crimson";
    } else {
        popup.style.backgroundColor = "mediumseagreen";
    }
    document.body.appendChild(popup);

    // Show the popup with a fade-in effect
    setTimeout(() => {
        popup.style.display = 'block';
        popup.style.opacity = '1';
    }, 100); // Slight delay for smoother transition

    // Set a timeout to hide and remove the popup
    popupTimeout = setTimeout(() => {
        popup.style.opacity = '0'; // Fade out
        setTimeout(() => {
            popup.style.display = 'none';
            if (popup.parentNode) {
                document.body.removeChild(popup); // Remove the popup from DOM after fading out
            }
        }, 500); // Wait for fade-out to finish
    }, 3000);
}