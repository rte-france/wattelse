const userIconButton = document.querySelector(".user-icon-button");
const userIconMenu = document.querySelector(".user-icon-menu");
const infoButton = document.querySelector(".info-button");
const infoMenu = document.querySelector(".info-menu");

// Info menu related functions
if (infoButton) {
    // Toggle the info menu when the info button is clicked
    infoButton.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
        infoMenu.style.display = infoMenu.style.display === 'flex' ? 'none' : 'flex';
    });

    // Don't close info menu if clicked inside
    infoMenu.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
    });

    // Hide menu when clicking outside
    document.addEventListener('click', () => {
        infoMenu.style.display = 'none';
    });
}

// User icon related functions
if (userIconButton) {
    // Toggle the menu when the user icon is clicked
    userIconButton.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
        userIconMenu.style.display = userIconMenu.style.display === 'flex' ? 'none' : 'flex';
    });

    // Don't close user menu if clicked inside
    userIconMenu.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
    });


    // Hide the menu when clicking outside
    document.addEventListener('click', () => {
        userIconMenu.style.display = 'none';
    });
}

function redirectToUrl(element) {
    // Get the URL from the data attribute
    const url = element.getAttribute('data-url');
    window.location.href = url;
}