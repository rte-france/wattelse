const messagesList = document.querySelector('#messagesList');
const messageForm = document.querySelector('#messageForm');
const messageInput = document.querySelector('#messageInput');

// Function to send message to backend then print response
messageForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (message.length === 0) {
        return;
    }
    
    const messageItem = document.createElement('li');
    messageItem.classList.add('message', 'sent');
    messageItem.innerHTML = `
        <div>
            <div>
                <b>Vous</b>
            </div>
            <div>
                ${message}
            </div>
        </div>`;
    messagesList.appendChild(messageItem);

    messageInput.value = '';

    fetch('', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            'message': message
        })
    })
    .then(response => response.json())
    .then(data => {
        const response = data.response;
        const messageItem = document.createElement('li');
        messageItem.innerHTML = `
        <div>
            <div>
                <b>Chatbot</b>
            </div>
            <div>
                ${response}
            </div>
        </div>
        `;
        messagesList.appendChild(messageItem);
    });
});


// Function to reset message history
const resetButton = document.querySelector('#resetButton');

resetButton.addEventListener('click', (event) => {
    event.preventDefault();
    messagesList.innerHTML = '';

    fetch('reset/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        })
    })
});