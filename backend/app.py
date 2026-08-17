from __future__ import annotations

import asyncio
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
ARTIFACT_PATH = ROOT / "artifacts" / "grabcar_deployment_model.joblib"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# Minimal basemap: roads remain readable, while labels/POIs and visual clutter are removed.
CARTO_TILE_URL = "https://basemaps.cartocdn.com/light_nolabels"
USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT",
    "GrabiezPortfolio/1.0 (local educational prototype; contact project owner)",
)


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    label: str = "Selected location"


class EstimateRequest(BaseModel):
    pickup: Coordinate
    destination: Coordinate


class GeocodeResult(BaseModel):
    label: str
    lat: float
    lon: float


class TierEstimate(BaseModel):
    service_tier_id: int
    service_tier: str
    estimated_price: int
    lower_price: int
    upper_price: int


class EstimateResponse(BaseModel):
    distance_km: float
    duration_minutes: float
    route_geometry: dict
    weather: dict
    timestamp: str
    estimates: list[TierEstimate]
    model: dict


app = FastAPI(title="Grabiez Pricing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not ARTIFACT_PATH.exists():
    raise RuntimeError(
        "Model artifact is missing. Run: python -m backend.build_model"
    )
ARTIFACT = joblib.load(ARTIFACT_PATH)

_geocode_cache: dict[str, list[dict]] = {}
_tile_cache: dict[tuple[int, int, int], bytes] = {}
_nominatim_lock = asyncio.Lock()
_last_nominatim_request = 0.0


async def request_json(url: str, *, params: dict | None = None) -> dict | list:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream service failed: {exc}") from exc


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "cv_rmse": ARTIFACT["cv_rmse"],
        "features": ARTIFACT["features"],
    }


@app.get("/api/tiles/{z}/{x}/{y}.png", include_in_schema=False)
async def map_tile(z: int, x: int, y: int) -> Response:
    if not (0 <= z <= 20 and x >= 0 and y >= 0):
        raise HTTPException(status_code=400, detail="Invalid tile coordinate.")
    key = (z, x, y)
    content = _tile_cache.get(key)
    if content is None:
        url = f"{CARTO_TILE_URL}/{z}/{x}/{y}.png"
        try:
            async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": USER_AGENT}) as client:
                upstream = await client.get(url)
                upstream.raise_for_status()
                content = upstream.content
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="Map tile unavailable.") from exc
        if len(_tile_cache) >= 768:
            _tile_cache.pop(next(iter(_tile_cache)))
        _tile_cache[key] = content
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/geocode", response_model=list[GeocodeResult])
async def geocode(
    q: Annotated[str, Query(min_length=3, max_length=180)],
) -> list[GeocodeResult]:
    global _last_nominatim_request
    key = q.strip().casefold()
    if key in _geocode_cache:
        return [GeocodeResult(**item) for item in _geocode_cache[key]]

    async with _nominatim_lock:
        wait_seconds = max(0.0, 1.05 - (time.monotonic() - _last_nominatim_request))
        if wait_seconds:
            await asyncio.sleep(wait_seconds)
        payload = await request_json(
            NOMINATIM_URL,
            params={
                "q": q,
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
                "countrycodes": "id",
            },
        )
        _last_nominatim_request = time.monotonic()

    results = [
        {
            "label": item["display_name"],
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
        }
        for item in payload
    ]
    _geocode_cache[key] = results
    return [GeocodeResult(**item) for item in results]


async def get_route(pickup: Coordinate, destination: Coordinate) -> dict:
    coordinates = f"{pickup.lon},{pickup.lat};{destination.lon},{destination.lat}"
    payload = await request_json(
        f"{OSRM_URL}/{coordinates}",
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
    )
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise HTTPException(status_code=422, detail="No drivable route was found.")
    return payload["routes"][0]


async def get_weather(lat: float, lon: float) -> dict:
    payload = await request_json(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "rain,cloud_cover,wind_speed_10m"
            ),
            "timezone": "auto",
        },
    )
    current = payload.get("current")
    if not current:
        raise HTTPException(status_code=502, detail="Weather data is unavailable.")
    return current


def round_idr(value: float) -> int:
    return max(1_000, int(round(value / 1_000) * 1_000))


def build_time_features(timestamp: pd.Timestamp) -> dict:
    hour = timestamp.hour + timestamp.minute / 60
    day_of_week = timestamp.dayofweek
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * day_of_week / 7),
        "dow_cos": math.cos(2 * math.pi * day_of_week / 7),
        "is_weekend": int(day_of_week in (5, 6)),
    }


def predict_tiers(distance_km: float, weather: dict, timestamp: pd.Timestamp):
    # Train data follows the original US rideshare dataset conventions.
    distance_miles = distance_km * 0.621371
    common = {
        "distance_mean": distance_miles,
        "humidity": float(weather["relative_humidity_2m"]) / 100,
        "rain": float(weather.get("rain", weather.get("precipitation", 0))) / 25.4,
        "temp": float(weather["temperature_2m"]) * 9 / 5 + 32,
        "wind": float(weather["wind_speed_10m"]) * 0.621371,
        "clouds": float(weather["cloud_cover"]) / 100,
        **build_time_features(timestamp),
    }

    estimates = []
    for tier_key, tier_types in ARTIFACT["types_by_tier"].items():
        tier_id = int(tier_key)
        raw_types = [int(value) for value in tier_types]
        rows = pd.DataFrame(
            [{"type": raw_type, "service_tier_id": tier_id, **common} for raw_type in raw_types]
        )
        predictions = ARTIFACT["model"].predict(rows[ARTIFACT["features"]])
        scale = float(ARTIFACT["price_scale_idr"])
        center = float(np.mean(predictions)) * scale
        lower = float(np.quantile(predictions, 0.20)) * scale
        upper = float(np.quantile(predictions, 0.80)) * scale
        estimates.append(
            TierEstimate(
                service_tier_id=tier_id,
                service_tier=ARTIFACT["tier_names"][tier_key],
                estimated_price=round_idr(center),
                lower_price=round_idr(min(lower, center)),
                upper_price=round_idr(max(upper, center)),
            )
        )
    return sorted(estimates, key=lambda value: value.service_tier_id)


@app.post("/api/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest) -> EstimateResponse:
    route, weather = await asyncio.gather(
        get_route(request.pickup, request.destination),
        get_weather(request.pickup.lat, request.pickup.lon),
    )
    now = pd.Timestamp.now(tz="Asia/Jakarta")
    distance_km = float(route["distance"]) / 1_000
    duration_minutes = float(route["duration"]) / 60
    estimates = predict_tiers(distance_km, weather, now)
    return EstimateResponse(
        distance_km=round(distance_km, 2),
        duration_minutes=round(duration_minutes, 1),
        route_geometry=route["geometry"],
        weather=weather,
        timestamp=now.isoformat(),
        estimates=estimates,
        model={
            "name": "Interaction Ridge",
            "latent_type_strategy": "uniform mean across tier types",
            "cv_rmse": ARTIFACT["cv_rmse"],
        },
    )


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest() -> FileResponse:
    return FileResponse(FRONTEND / "manifest.webmanifest")


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    return FileResponse(FRONTEND / "sw.js", media_type="application/javascript")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")
