const params = new URLSearchParams(window.location.search);
const type = params.get("type"); // public / private / clinic

document.getElementById("heading").innerText =
  `Nearby ${type.charAt(0).toUpperCase() + type.slice(1)} Hospitals`;

let userLat, userLon;

navigator.geolocation.getCurrentPosition(position => {
  userLat = position.coords.latitude;
  userLon = position.coords.longitude;
  console.log(userLat, userLon);
  // Next step: fetch hospitals here
});

function searchHospitals() {
  const query = document.getElementById("searchBox").value;
  console.log("Search:", query, "Type:", type);
  // Call API with (lat, lon, type, query)
}
