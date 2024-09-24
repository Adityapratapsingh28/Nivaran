// GSAP Animations
gsap.registerPlugin(ScrollTrigger);

// Navbar animation
gsap.from("nav", {
    y: -100,
    opacity: 0,
    duration: 1,
    ease: "power3.out"
});

// Welcome text animation
const welcomeText = document.querySelectorAll('#welcomeText span');
gsap.to(welcomeText, {
    y: 0,
    opacity: 1,
    duration: 2 ,
    stagger: 0.1,
    ease: "bounce.out"
});

// Home section animation
gsap.from("#home p, #home a", {
    opacity: 0,
    y: 50,
    duration: 2,
    delay: 1,
    stagger: 0.2,
    ease: "power3.out"
});

const chatinput = document.querySelector(".chat-input textarea");
const sendchatbtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot-toggler");
let usermessage;

// Replace with your actual API key
const API_KEY = 'AIzaSyBYqqnBVjDCO-d4_Wwh7L-1xKWrmyOF6-c';

// Function to create a chat message (either outgoing or incoming)
const createchatli = (message, classname) => {
    const chatli = document.createElement("li");
    chatli.classList.add("chat", classname);
    let chatcontent = classname === "outgoing"
        ? `<p>${message}</p>`
        : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatli.innerHTML = chatcontent;
    chatli.querySelector("p").textContent = message;
    return chatli;
};

// Function to call the Gemini API and get the chatbot response
const generateResponse = async (incomingChatLi) => {
    const messageElement = incomingChatLi.querySelector("p");
    
    if (!API_KEY || API_KEY === 'YOUR_GEMINI_API_KEY_HERE') {
        messageElement.textContent = "Error: API key not set. Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key.";
        return;
    }

    try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: [{
                    parts: [{
                        text: usermessage
                    }]
                }]
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`API request failed: ${response.status} ${response.statusText}\n${JSON.stringify(errorData)}`);
        }

        const data = await response.json();
        if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts && data.candidates[0].content.parts[0]) {
            const generatedText = data.candidates[0].content.parts[0].text;
            messageElement.textContent = generatedText;
        } else {
            throw new Error('Unexpected API response structure');
        }
    } catch (error) {
        console.error("Error:", error);
        messageElement.textContent = `Error: ${error.message}`;
    } finally {
        chatbox.scrollTo(0, chatbox.scrollHeight);
    }
};

// Function to handle sending a message
const handlechat = () => {
    usermessage = chatinput.value.trim();
    if (!usermessage) return;
    chatinput.value = ""; // Clear the input field

    // Append outgoing chat message
    chatbox.appendChild(createchatli(usermessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    // Show a "Thinking..." message for the bot's response
    setTimeout(() => {
        const incomingChatLi = createchatli("Thinking...", "incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);

        // Call the function to generate a response from the API
        generateResponse(incomingChatLi);
    }, 600);
};

// Event listeners for sending messages and toggling the chatbot visibility
sendchatbtn.addEventListener("click", handlechat);
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));


// Disease data
const diseases = [
    {
        name: "Brain Tumor",
        description: "Learn about the symptoms and treatment of brain tumors.",
        link: "disease?disease=brain_tumor"
    },
    {
        name: "Cancer",
        description: "Understand various types of cancer, their symptoms, and treatments.",
        link: "disease?disease=cancer"
    },
    {
        name: "X Ray Fractured",
        description: "Check whether you have a Fractured bone",
        link: "disease?disease=jaundice"
    },
    {
        name: "Chronic Obstructive Pulmonary Disease",
        description: "Learn about the symptoms and treatment of COPD",
        link: "disease?disease=typhoid"
    },
    {
        name: "Diabetes",
        description: "Understand the causes and management of diabetes.",
        link: "disease?disease=diabetes"
    },
    {
        name: "Lung Cancer Detection",
        description: "Learn about the symptoms and treatment of Lung Cancer.",
        link: "disease?disease=lung-cancer"
    }

];

// Populate disease cards
const diseaseContainer = document.getElementById("diseaseCards");
diseases.forEach((disease, index) => {
    const card = document.createElement("div");
    card.className = "disease-card";
    card.innerHTML = `
        <h3 class="text-2xl font-semibold mb-4">${disease.name}</h3>
        <p class="mb-4">${disease.description}</p>
        <a href="${disease.link}" class="text-blue-600 dark:text-blue-400 hover:underline">Read more →</a>
    `;
    diseaseContainer.appendChild(card);

    // Animate each card
    gsap.to(card, {
        opacity: 1,
        x: 0,
        duration: 2.0,
        scrollTrigger: {
            trigger: card,
            start: "top 90%",
            end: "bottom 60%",
            toggleActions: "play none none reverse"
        }
    });
});

// Pharmacy section animation
gsap.from("#pharmacy h2, #map", {
    opacity: 0,
    y: 50,
    duration: 1,
    stagger: 0.2,
    scrollTrigger: {
        trigger: "#pharmacy",
        start: "top 80%",
        end: "bottom 20%",
        toggleActions: "play none none reverse"
    }
});

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const htmlElement = document.documentElement;

function setTheme(theme) {
    if (theme === 'dark') {
        htmlElement.classList.add('dark');
    } else {
        htmlElement.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
}

themeToggle.addEventListener('click', () => {
    const currentTheme = localStorage.getItem('theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
});

// Check for saved theme preference or prefer-color-scheme
const savedTheme = localStorage.getItem('theme');
const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

if (savedTheme) {
    setTheme(savedTheme);
} else if (prefersDark) {
    setTheme('dark');
}

// Pharmacy search functionality
const searchPharmacy = document.getElementById('searchPharmacy');
const pincodeInput = document.getElementById('pincode');
let map;

// Initialize the map
function initMap() {
    map = L.map('map').setView([20.5937, 78.9629], 5); // Center on India
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
}

initMap();

searchPharmacy.addEventListener('click', async () => {
    const pincode = pincodeInput.value.trim();
    const pharmacyResultsContainer = document.getElementById('pharmacyResults');
    pharmacyResultsContainer.innerHTML = ''; // Clear previous results

    if (pincode.length !== 6 || !/^\d+$/.test(pincode)) {
        alert('Please enter a valid 6-digit PIN code');
        return;
    }

    try {
        // Get the coordinates for the PIN code
        const nominatimResponse = await fetch(`https://nominatim.openstreetmap.org/search?postalcode=${pincode}&country=India&format=json`);
        const nominatimData = await nominatimResponse.json();

        if (nominatimData.length === 0) {
            alert('No location found for this PIN code');
            return;
        }

        const { lat, lon } = nominatimData[0];
        map.setView([lat, lon], 14);

        // Search for pharmacies near this location
        const overpassApiUrl = `https://overpass-api.de/api/interpreter?data=[out:json];node(around:5000,${lat},${lon})["amenity"="pharmacy"];out;`;
        const overpassResponse = await fetch(overpassApiUrl);
        const overpassData = await overpassResponse.json();

        // Clear existing markers on the map
        map.eachLayer((layer) => {
            if (layer instanceof L.Marker) {
                map.removeLayer(layer);
            }
        });

        // If no pharmacies found, display a message
        if (overpassData.elements.length === 0) {
            alert('No pharmacies found in this area');
        } else {
            // Loop through the pharmacies and create a card for each
            overpassData.elements.forEach(pharmacy => {
                const pharmacyName = pharmacy.tags.name || 'Unnamed Pharmacy';
                const pharmacyAddress = pharmacy.tags['addr:street'] || 'Address not available';

                // Add marker to the map
                L.marker([pharmacy.lat, pharmacy.lon])
                    .addTo(map)
                    .bindPopup(pharmacyName);

                // Create a card for each pharmacy
                const card = document.createElement('div');
                card.className = 'pharmacy-card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md';
                card.innerHTML = `
                    <h3 class="text-lg font-semibold mb-2">${pharmacyName}</h3>
                    <p class="text-sm mb-2">${pharmacyAddress}</p>
                    <a href="https://www.google.com/maps/search/?api=1&query=${pharmacy.lat},${pharmacy.lon}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline">View on Map</a>
                `;
                pharmacyResultsContainer.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Error fetching pharmacy data:', error);
        alert('An error occurred while searching for pharmacies');
    }
});
