const map = L.map("map", {
  zoomControl: false,
  attributionControl: false,
  preferCanvas: false,
  zoomAnimation: true,
  fadeAnimation: true,
  markerZoomAnimation: true,
  zoomSnap: 0.5,
  zoomDelta: 0.5,
  wheelDebounceTime: 25,
  wheelPxPerZoomLevel: 90,
  dragging: true,
  touchZoom: true,
  doubleClickZoom: true,
  scrollWheelZoom: true,
  boxZoom: false,
  keyboard: true,
  inertia: true,
  inertiaDeceleration: 2800,
  inertiaMaxSpeed: 1800,
  easeLinearity: 0.22,
}).setView([-6.2, 106.816666], 12);
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png", {
  subdomains: "abcd",
  maxZoom: 20,
  tileSize: 256,
  detectRetina: false,
  crossOrigin: true,
  updateWhenIdle: true,
  updateWhenZooming: false,
  keepBuffer: 2,
}).addTo(map);

map.createPane("routePane");
map.getPane("routePane").style.zIndex = 650;
map.getPane("routePane").style.pointerEvents = "none";

const mapContainer = document.querySelector("#map");
const refreshMap = () => map.invalidateSize({ pan: true, animate: false });
new ResizeObserver(refreshMap).observe(mapContainer);
window.addEventListener("load", () => {
  refreshMap();
  requestAnimationFrame(refreshMap);
  setTimeout(refreshMap, 150);
  setTimeout(refreshMap, 600);
});
window.addEventListener("orientationchange", () => setTimeout(refreshMap, 150));

const state = {
  pickup: null,
  destination: null,
  pickupMarker: null,
  destinationMarker: null,
  routeLayers: null,
  mapSelection: "pickup",
};

const elements = {
  pickupInput: document.querySelector("#pickupInput"),
  destinationInput: document.querySelector("#destinationInput"),
  pickupResults: document.querySelector("#pickupResults"),
  destinationResults: document.querySelector("#destinationResults"),
  estimateButton: document.querySelector("#estimateButton"),
  status: document.querySelector("#status"),
  tierList: document.querySelector("#tierList"),
  mapHint: document.querySelector("#mapHint"),
  tripSummary: document.querySelector("#tripSummary"),
  routeSummary: document.querySelector("#routeSummary"),
  weatherChips: document.querySelector("#weatherChips"),
};

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.style.color = isError ? "#c9362b" : "";
  elements.status.classList.remove("hidden");
}

function updateReadyState() {
  const ready = Boolean(state.pickup && state.destination);
  elements.estimateButton.disabled = !ready;
  if (ready) {
    elements.mapHint.textContent = "Rute siap dihitung";
    setStatus("Tekan tombol untuk mengambil rute dan cuaca terkini.");
  }
}

function markerIcon(color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<span class="map-marker" style="display:block;background:${color}"></span>`,
    iconSize: [28, 34],
    iconAnchor: [14, 31],
  });
}

function setLocation(target, location) {
  state[target] = location;
  elements[`${target}Input`].value = location.label;
  const markerKey = `${target}Marker`;
  if (state[markerKey]) state[markerKey].remove();
  state[markerKey] = L.marker([location.lat, location.lon], {
    icon: markerIcon(target === "pickup" ? "#2787e8" : "#ef4a3c"),
  }).addTo(map);

  if (target === "pickup") {
    state.mapSelection = "destination";
    elements.mapHint.textContent = "Sekarang pilih tujuan";
  }
  if (state.pickup && state.destination) {
    map.fitBounds([
      [state.pickup.lat, state.pickup.lon],
      [state.destination.lat, state.destination.lon],
    ], { padding: [55, 55], animate: true });
  } else {
    map.flyTo([location.lat, location.lon], 15, { duration: .65 });
  }
  updateReadyState();
}

async function searchLocation(target) {
  const input = elements[`${target}Input`];
  const resultsBox = elements[`${target}Results`];
  const query = input.value.trim();
  if (query.length < 3) {
    setStatus("Masukkan minimal tiga karakter.", true);
    return;
  }
  resultsBox.innerHTML = '<div class="result-item">Mencari lokasi…</div>';
  resultsBox.classList.add("open");
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error((await response.json()).detail || "Pencarian gagal");
    const results = await response.json();
    resultsBox.innerHTML = "";
    if (!results.length) {
      resultsBox.innerHTML = '<div class="result-item">Lokasi tidak ditemukan</div>';
      return;
    }
    results.forEach((result) => {
      const button = document.createElement("button");
      button.className = "result-item";
      button.textContent = result.label;
      button.addEventListener("click", () => {
        setLocation(target, result);
        resultsBox.classList.remove("open");
      });
      resultsBox.appendChild(button);
    });
  } catch (error) {
    resultsBox.classList.remove("open");
    setStatus(error.message, true);
  }
}

document.querySelectorAll(".search-action").forEach((button) => {
  button.addEventListener("click", () => searchLocation(button.dataset.target));
});
["pickup", "destination"].forEach((target) => {
  elements[`${target}Input`].addEventListener("keydown", (event) => {
    if (event.key === "Enter") searchLocation(target);
  });
});

map.on("click", (event) => {
  const target = state.mapSelection;
  setLocation(target, {
    lat: event.latlng.lat,
    lon: event.latlng.lng,
    label: `${event.latlng.lat.toFixed(5)}, ${event.latlng.lng.toFixed(5)}`,
  });
});

document.querySelector("#locateButton").addEventListener("click", () => {
  if (!navigator.geolocation) {
    setStatus("Geolocation tidak tersedia di browser ini.", true);
    return;
  }
  setStatus("Mengambil lokasi perangkat…");
  navigator.geolocation.getCurrentPosition(
    ({ coords }) => setLocation("pickup", {
      lat: coords.latitude,
      lon: coords.longitude,
      label: "Lokasi saya",
    }),
    () => setStatus("Izin lokasi ditolak atau lokasi tidak tersedia.", true),
    { enableHighAccuracy: true, timeout: 10000 },
  );
});

function rupiah(value) {
  return new Intl.NumberFormat("id-ID", {
    style: "currency", currency: "IDR", maximumFractionDigits: 0,
  }).format(value);
}

function renderRoute(geometry) {
  if (state.routeLayers) state.routeLayers.remove();
  const latLngs = geometry.coordinates.map(([longitude, latitude]) => [latitude, longitude]);
  const outline = L.polyline(latLngs, {
    pane: "routePane", color: "#ffffff", weight: 12, opacity: 1,
    lineCap: "round", lineJoin: "round", interactive: false,
  });
  const route = L.polyline(latLngs, {
    pane: "routePane", color: "#00a86b", weight: 7, opacity: 1,
    lineCap: "round", lineJoin: "round", interactive: false,
  });
  state.routeLayers = L.layerGroup([outline, route]).addTo(map);
  outline.bringToFront();
  route.bringToFront();
  refreshMap();
  map.fitBounds(route.getBounds(), { paddingTopLeft: [45, 55], paddingBottomRight: [45, 75] });
  elements.mapHint.classList.add("hidden");
}

function renderEstimates(data) {
  elements.tierList.innerHTML = "";
  const carIcon = `
    <svg viewBox="0 0 48 28" aria-hidden="true">
      <path d="M7 19h34l-2.4-8.2a4 4 0 0 0-3.8-2.8H16.1a4 4 0 0 0-3.5 2L7 19Z" />
      <path d="M3.5 18.5h41v4.2H3.5z" />
      <circle cx="12" cy="23" r="3.5" /><circle cx="36" cy="23" r="3.5" />
      <path class="car-window" d="m15.5 10.5-3.4 6h10.4v-6h-7Zm10 0v6h11.2l-1.8-6h-9.4Z" />
    </svg>`;
  data.estimates.forEach((estimate, index) => {
    const card = document.createElement("article");
    card.className = `tier-card ${estimate.service_tier_id === 2 ? "recommended" : ""}`;
    card.innerHTML = `
      <div class="tier-icon tier-${estimate.service_tier_id}">${carIcon}</div>
      <div>
        <div class="tier-name">${estimate.service_tier}</div>
        <div class="tier-meta">${estimate.service_tier_id === 2 ? "Rekomendasi" : "Estimasi model"}</div>
      </div>
      <div class="tier-price">
        ${rupiah(estimate.estimated_price)}
        <span class="tier-range">${rupiah(estimate.lower_price)}–${rupiah(estimate.upper_price)}</span>
      </div>`;
    elements.tierList.appendChild(card);
  });
  elements.routeSummary.textContent = `${data.duration_minutes} min · ${data.distance_km} km`;
  const weather = data.weather;
  elements.weatherChips.innerHTML = `
    <span class="chip"><b>Suhu</b> ${Math.round(weather.temperature_2m)}°C</span>
    <span class="chip"><b>RH</b> ${weather.relative_humidity_2m}%</span>
    <span class="chip"><b>Hujan</b> ${weather.rain ?? weather.precipitation} mm</span>`;
  elements.tripSummary.classList.remove("hidden");
  elements.status.classList.add("hidden");
  renderRoute(data.route_geometry);
}

elements.estimateButton.addEventListener("click", async () => {
  elements.estimateButton.disabled = true;
  elements.estimateButton.classList.add("loading");
  elements.estimateButton.querySelector("span").textContent = "Menghitung rute…";
  setStatus("Mengambil jarak jalan dan cuaca real-time…");
  try {
    const response = await fetch("/api/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pickup: state.pickup, destination: state.destination }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || "Estimasi gagal");
    renderEstimates(await response.json());
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    elements.estimateButton.disabled = false;
    elements.estimateButton.classList.remove("loading");
    elements.estimateButton.querySelector("span").textContent = "Hitung ulang estimasi";
  }
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js"));
}
