/************ GLOBAL STATE ************/
const USE_LOCAL_BACKEND = false; // set false to use deployed backend
const LOCAL_BACKEND_HOST = "http://127.0.0.1:5000";
const REMOTE_BACKEND_HOST = "https://ai-smart-health-assistant-backend.onrender.com";
const BACKEND_HOST = USE_LOCAL_BACKEND ? LOCAL_BACKEND_HOST : REMOTE_BACKEND_HOST;

let currentLat = null;
let currentLng = null;
let currentCategory = "hospital";
let predictedDisease = null;

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
  const predictUrl = `${BACKEND_HOST}/predict`;
  fetch(predictUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ "symptoms": symptoms })
  })
  .then(res => {
    if (!res.ok) throw new Error("Server communication fault");
    return res.json();
  })
  .then(data => {
    if (data.error) throw new Error(data.error);
    document.getElementById("resultCard").style.display = "block";
    document.getElementById("diseaseName").innerText = `🩺 ${data.disease || "Unknown Condition"}`;

    let info = "";
    if (data.info) info += data.info + "\n";
    if (data.confidence !== undefined) info += `Confidence: ${data.confidence}%\n`;
    if (data.source) info += `Source: ${data.source}`;

    document.getElementById("diseaseInfo").innerText = info || "No extra data structural logs provided.";

    predictedDisease = data.disease;
    if (currentLat && currentLng) {
      updateMap();
    } else {
      getUserLocation(updateMap);
    }

    const menu = document.getElementById("menuContainer");
    if (menu) menu.style.display = "flex";
  })
  .catch((err) => {
    console.error("Prediction request failed:", err);
    const message = err?.message || "Backend not running or calculation failed.";
    alert(`Prediction failed: ${message}`);
  });
}

/************ EMERGENCY BUTTON ************/
function confirmEmergency() {
  if (confirm("Call Emergency Ambulance (108)?")) {
    window.location.href = "tel:108";
  }
}

/************ INITIALIZATION ************/
window.onload = function() {
  getUserLocation();

  const menu = document.getElementById("menuContainer");
  const menuBtn = document.getElementById("menuBtn");

  if (menu) menu.style.display = "none";

  if (menuBtn && menu) {
    menuBtn.addEventListener("click", () => {
      if (menu.style.display === "none") {
        menu.style.display = "flex";
      } else {
        menu.style.display = "none";
      }
    });
  }
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
  if (event.key === "Enter") {
    event.preventDefault();
    sendChatMessage();
  }
}

async function sendChatMessage(event) {
  if (event) event.preventDefault();
  const inputEl = document.getElementById("chatInput");
  const messageText = inputEl.value.trim();
  if (!messageText) return;

  const msgContainer = document.getElementById("chatMessages");

  const userMsg = document.createElement("div");
  userMsg.className = "message user-msg animate-bubble";
  userMsg.innerHTML = `<div class="msg-text">${escapeHTML(messageText)}</div>`;
  msgContainer.appendChild(userMsg);
  
  inputEl.value = "";
  msgContainer.scrollTop = msgContainer.scrollHeight;

  const loadingId = "loading-" + Date.now();
  const loadingMsg = document.createElement("div");
  loadingMsg.className = "message bot-msg loading animate-bubble";
  loadingMsg.id = loadingId;
  loadingMsg.innerHTML = `<i class="fas fa-robot msg-icon"></i><div class="typing-indicator"><span></span><span></span><span></span></div>`;
  msgContainer.appendChild(loadingMsg);
  msgContainer.scrollTop = msgContainer.scrollHeight;

  const targetChatRoute = `${BACKEND_HOST}/chat`;
  try {
    const response = await fetch(targetChatRoute, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ "message": messageText })
    });

    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) msgContainer.removeChild(loadingEl);
    
    const data = await response.json();
    if (!response.ok) throw new Error();    

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
    "'": '&#39;', 
    '"': '&quot;' 
  }[tag] || tag));
}
