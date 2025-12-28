/************ GLOBAL STATE ************/
let currentLat = null;
let currentLng = null;
let predictedDisease = null;
let currentCategory = "hospital";

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

  fetch("http://127.0.0.1:5000/predict", {
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
async function predictAI() {
    const symptoms = document.getElementById("aiSymptoms").value.trim();
    if (!symptoms) {
        alert("Please enter your symptoms!");
        return;
    }

    // Show loading
    const aiResultDiv = document.getElementById("aiResult");
    aiResultDiv.style.display = "block";
    aiResultDiv.innerHTML = "Analyzing...";

    // Choose backend URL: local or ngrok
    //const backendUrl = "http://127.0.0.1:5000/predict"; 
    // If using ngrok, replace with: 
    const backendUrl = "https://neva-nonoxidizable-guilefully.ngrok-free.dev/predict";

    try {
        const response = await fetch(backendUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symptoms })
        });

        if (!response.ok) {
            aiResultDiv.innerHTML = "Error: Server returned " + response.status;
            return;
        }

        const data = await response.json();

        // Fill the AI result
        document.getElementById("aiDisease").innerText = data.predicted_disease || "N/A";
        document.getElementById("aiDoctor").innerText = data.suggested_doctor || "N/A";

        const remediesList = document.getElementById("aiRemedies");
        remediesList.innerHTML = "";
        if (data.home_remedies && data.home_remedies.length > 0) {
            data.home_remedies.forEach(remedy => {
                const li = document.createElement("li");
                li.textContent = remedy;
                remediesList.appendChild(li);
            });
        } else {
            remediesList.innerHTML = "<li>N/A</li>";
        }

        const foodsList = document.getElementById("aiFoods");
        foodsList.innerHTML = "";
        if (data.recommended_foods && data.recommended_foods.length > 0) {
            data.recommended_foods.forEach(food => {
                const li = document.createElement("li");
                li.textContent = food;
                foodsList.appendChild(li);
            });
        } else {
            foodsList.innerHTML = "<li>N/A</li>";
        }

    } catch (err) {
        console.error(err);
        aiResultDiv.innerHTML = "Error: Cannot connect to AI backend.";
    }
}


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

