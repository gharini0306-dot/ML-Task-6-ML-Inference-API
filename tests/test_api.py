
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    features = [
        17.99, 10.38, 122.8, 1001, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200
    assert response.json()["class_name"] in ["malignant", "benign"]
    assert "probabilities" in response.json()


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]}
    )

    assert response.status_code == 422