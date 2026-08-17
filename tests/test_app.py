import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app, predict_tiers


client = TestClient(app)


def test_health_and_mobile_shell():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    shell = client.get("/")
    assert shell.status_code == 200
    assert "Grabiez Fare" in shell.text


def test_tier_predictions_are_complete_and_ordered():
    weather = {
        "relative_humidity_2m": 78,
        "rain": 0.2,
        "precipitation": 0.2,
        "temperature_2m": 29,
        "wind_speed_10m": 12,
        "cloud_cover": 65,
    }
    estimates = predict_tiers(
        distance_km=4.2,
        weather=weather,
        timestamp=pd.Timestamp("2026-08-17T22:30:00+07:00"),
    )
    assert [value.service_tier_id for value in estimates] == [1, 2, 3]
    assert all(value.lower_price <= value.estimated_price <= value.upper_price for value in estimates)
    assert estimates[0].estimated_price < estimates[-1].estimated_price
