const userButton = document.querySelector(".user-button");
const userMenu = document.querySelector(".user-menu");

// User menu related functions
if (userButton) {
    // Toggle the menu when the user icon is clicked
    userButton.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
        userMenu.style.display = userMenu.style.display === 'flex' ? 'none' : 'flex';
    });

    // Don't close user menu if clicked inside
    userMenu.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevents the click from propagating to the document
    });


    // Hide the menu when clicking outside
    document.addEventListener('click', () => {
        userMenu.style.display = 'none';
    });
}