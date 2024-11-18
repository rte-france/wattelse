// Imports
import {
    showFaqPopup,
    uuid4,
} from './faq_functions.js'

// Global variables
const csrfmiddlewaretoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
const faqTable = document.querySelector(".faq-table");
const openPopupBtn = document.getElementById('faqAddButton');

// Event listeners
openPopupBtn.addEventListener('click', () => {
    showFaqPopup(uuid4());
});

const tableRows = faqTable.querySelectorAll('tr'); // avoid taking the header row
tableRows.forEach(tr => {
    const tdList = tr.querySelectorAll("td");
    tr.addEventListener("click", () => {
        showFaqPopup(tr.id, tdList[0].textContent, tdList[1].textContent, true);
    });
});