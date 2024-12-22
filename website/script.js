// Load dataset from JSON file
let dataset = [];
let currentFilter = 'all';

const promptTypes = [
    'Professional Biography',
    'Business Email',
    'Legal Contract',
    'News Article',
    'Social Media Post'
];

// Function to highlight entities in text
function highlightEntities(text, entities) {
    let highlightedText = text;
    entities.forEach(entity => {
        const value = entity.entity_value;
        const type = entity.entity_type;
        const source = entity.entity_source;
        const span = `<span class="entity-tag entity-${type} source-${source}">${value}</span>`;
        highlightedText = highlightedText.replace(value, span);
    });
    return highlightedText;
}

// Function to create example card
function createExampleCard(example, index) {
    const promptType = promptTypes[index % 5];
    const card = document.createElement('div');
    card.className = 'card mb-3';
    card.setAttribute('data-type', promptType.toLowerCase().replace(' ', '-'));
    
    const highlightedText = highlightEntities(example.input, example.output);
    
    card.innerHTML = `
        <div class="card-header">
            <h5 class="mb-0">Example ${index + 1} - ${promptType}</h5>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <h6>Generated Text:</h6>
                <p class="card-text">${highlightedText}</p>
            </div>
            <div>
                <h6>Detected Entities:</h6>
                <div>
                    ${example.output.map(entity => `
                        <span class="entity-tag entity-${entity.entity_type} source-${entity.entity_source}">
                            ${entity.entity_type}: ${entity.entity_value}
                        </span>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    return card;
}

// Function to filter examples
function filterExamples() {
    const examplesList = document.getElementById('examplesList');
    const cards = examplesList.getElementsByClassName('card');
    
    Array.from(cards).forEach(card => {
        if (currentFilter === 'all' || card.getAttribute('data-type') === currentFilter) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

// Load and display dataset
async function loadDataset() {
    try {
        const response = await fetch('pii_dataset.json');
        dataset = await response.json();
        
        const examplesList = document.getElementById('examplesList');
        dataset.forEach((example, index) => {
            const card = createExampleCard(example, index);
            examplesList.appendChild(card);
        });
        
        // Set up filter listener
        document.getElementById('exampleType').addEventListener('change', (e) => {
            currentFilter = e.target.value;
            filterExamples();
        });
    } catch (error) {
        console.error('Error loading dataset:', error);
        document.getElementById('examplesList').innerHTML = 
            '<div class="alert alert-danger">Error loading dataset. Please try again later.</div>';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', loadDataset);
