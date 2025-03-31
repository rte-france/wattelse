export function redirectToUrl(element) {
    // Get the URL from the data attribute
    const url = element.getAttribute('data-url');
    window.location.href = url;
}