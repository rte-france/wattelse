import { redirectToUrl } from "../../../static/js/utils.js";

const gptButton = document.getElementById('gpt-button');
const docButton = document.getElementById('doc-button');

gptButton.addEventListener('click', () => {
    redirectToUrl(gptButton);
});

docButton.addEventListener('click', () => {
    redirectToUrl(docButton);
});