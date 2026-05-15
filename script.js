/************ GLOBAL STATE ************/
let currentLat = null;
let currentLng = null;
let currentCategory = "hospital";
let predictedDisease = null
/************ LOCATION ************/
function getUserLocation(callback) {
  if (!navigator.geolocation) {
    alert("Geolocation not supported");
    return;
  }

  navigator.geolocation.watchPosition(
    (position) => {
      currentLat = position.coords.latitude;
      currentLng = position.coords.longitude;
      if (callback) callback();
    },
    (error) => {
      alert("Enable location permission & refresh");
      console.error(error);
    },
    { enableHighAccuracy: true }
  );
}

/************ MAP UPDATE ************/
function updateMap() {
  if (!currentLat || !currentLng || !predictedDisease) return;

  const iframe = document.getElementById("maps-frame");
  const mapSection = document.getElementById("map-section");

  let query = `${predictedDisease} hospitals near ${currentLat},${currentLng}`;

  if (currentCategory === "public") {
    query += " government";
  } else if (currentCategory === "private") {
    query += " private";
  } else if (currentCategory === "clinic") {
    query = `clinics near ${currentLat},${currentLng}`;
  }

  iframe.src = `https://www.google.com/maps?q=${encodeURIComponent(query)}&output=embed`;
  mapSection.style.display = "block";
}

/************ CATEGORY CHANGE (MENU BUTTONS) ************/
function changeCategory(category) {
  currentCategory = category;
  updateMap();
}

/************ MAIN SEARCH (SYMPTOMS) ************/
function handleSearch() {
  const symptoms = document.getElementById("searchText").value.trim();
  if (!symptoms) {
    alert("Enter symptoms");
    return;
  }
const predictUrl = "https://herokuapp.com";
  fetch(predictUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms })
  })
  .then(res => res.json())
  .then(data => {
    // Show disease info
    document.getElementById("resultCard").style.display = "block";
    document.getElementById("diseaseName").innerText = `🩺 ${data.disease}`;

    let info = "";
    if (data.info) info += data.info + "\n";
    if (data.confidence) info += `Confidence: ${data.confidence}%\n`;
    if (data.source) info += `Source: ${data.source}`;

    document.getElementById("diseaseInfo").innerText = info;

    // Update map with nearby hospitals
    predictedDisease = data.disease;
    if (currentLat && currentLng) {
      updateMap();
    } else {
      getUserLocation(updateMap);
    }

    // Show menu after search
    const menu = document.getElementById("menuContainer");
    menu.style.display = "flex";  // or "block" depending on your CSS
  })
  .catch(() => alert("Backend not running"));
}

/************ AI PREDICTION (WITHOUT MAP) ************/

/************ EMERGENCY BUTTON ************/
function confirmEmergency() {
  if (confirm("Call Emergency Ambulance (108)?")) {
    window.location.href = "tel:108";
  }
}

/************ INITIALIZATION ************/
window.onload = function() {
  getUserLocation(); // start location tracking

  const menu = document.getElementById("menuContainer");
  const menuBtn = document.getElementById("menuBtn");

  // Ensure menu starts hidden
  menu.style.display = "none";

  // Toggle menu on button click
  menuBtn.addEventListener("click", () => {
    if (menu.style.display === "none") {
      menu.style.display = "flex"; // show menu
    } else {
      menu.style.display = "none"; // hide menu
    }
  });
};
function toggleChat() {
  const chatWindow = document.getElementById("chatWindow");
  const toggleBtn = document.getElementById("chatToggleBtn");
  
  if (chatWindow.style.display === "none") {
    chatWindow.style.display = "flex";
    chatWindow.classList.add("slide-in");
    toggleBtn.style.transform = "scale(0)";
    setTimeout(() => toggleBtn.style.display = "none", 200);
  } else {
    chatWindow.classList.remove("slide-in");
    toggleBtn.style.display = "flex";
    setTimeout(() => toggleBtn.style.transform = "scale(1)", 10);
    chatWindow.style.display = "none";
  }
}

function handleChatKey(event) {
  if (event.key === "Enter")  {
    event.preventDefault(); // Stop the browser from refreshing the page!
    sendChatMessage();
  }
}

async function sendChatMessage(event) {
    if (event) event.preventDefault();
  const inputEl = document.getElementById("chatInput");
  const messageText = inputEl.value.trim();
  if (!messageText) return;

  const msgContainer = document.getElementById("chatMessages");

  // User Message Assembly
  const userMsg = document.createElement("div");
  userMsg.className = "message user-msg animate-bubble";
  userMsg.innerHTML = `<div class="msg-text">${escapeHTML(messageText)}</div>`;
  msgContainer.appendChild(userMsg);
  
  inputEl.value = "";
  msgContainer.scrollTop = msgContainer.scrollHeight;

  // Typing Placeholder Assembly
  const loadingId = "loading-" + Date.now();
  const loadingMsg = document.createElement("div");
  loadingMsg.className = "message bot-msg loading animate-bubble";
  loadingMsg.id = loadingId;
  loadingMsg.innerHTML = `<i class="fas fa-robot msg-icon"></i><div class="typing-indicator"><span></span><span></span><span></span></div>`;
  msgContainer.appendChild(loadingMsg);
  msgContainer.scrollTop = msgContainer.scrollHeight;

  // Base URL Setup (Updates directly to your assigned public domain URL when deployed on Render)
  const targetChatRoute = "https://herokuapp.com";
   try {
    const response = await fetch(targetChatRoute, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: messageText })
    });

    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) msgContainer.removeChild(loadingEl);
     const data = await response.json();
    if (!response.ok) throw new Error();    
    // Bot Dynamic Markdown Message Assembly
    const botMsg = document.createElement("div");
    botMsg.className = "message bot-msg animate-bubble";
    botMsg.innerHTML = `
      <i class="fas fa-robot msg-icon"></i>
      <div class="msg-text">${marked.parse(data.reply)}</div>
    `;
    msgContainer.appendChild(botMsg);

  } catch (error) {
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) msgContainer.removeChild(loadingEl);

    const errorMsg = document.createElement("div");
    errorMsg.className = "message bot-msg system-error";
    errorMsg.innerHTML = `<i class="fas fa-exclamation-triangle msg-icon"></i><div class="msg-text">Connection down. Please verify server deployment settings.</div>`;
    msgContainer.appendChild(errorMsg);
  }

  msgContainer.scrollTop = msgContainer.scrollHeight;
}

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, tag => ({ 
    '&': '&amp;', 
    '<': '&lt;', 
    '>': '&gt;', 
    "'": '&#39;', // Fixed the unclosed single quote bug
    '"': '&quot;' 
  }[tag] || tag));
}