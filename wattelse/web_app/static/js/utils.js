export function redirectToUrl(element) {
    // Get the URL from the data attribute
    const url = element.getAttribute('data-url');
    window.location.href = url;
}

export function uuid4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

let popupTimeout; // needed to avoid removing popup that is already removed
export function showPopup(message, error = false) {
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
      popup.classList.add("error");
  } else {
      popup.classList.add("success");
  }
  document.body.appendChild(popup);

  // Show the popup with a fade-in effect
  setTimeout(() => {
      popup.style.display = 'block';
      popup.style.opacity = '1';
  }, 100);

  // Set a timeout to hide and remove the popup
  popupTimeout = setTimeout(() => {
      popup.style.opacity = '0';
      setTimeout(() => {
          popup.style.display = 'none';
          if (popup.parentNode) {
              document.body.removeChild(popup); // Remove the popup from DOM after fading out
          }
      }, 500); // Wait for fade-out to finish
  }, 3000);
}

// TODO: can't use clipboard API in non-secure context (http)
export function unsecuredCopyToClipboard(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
  }