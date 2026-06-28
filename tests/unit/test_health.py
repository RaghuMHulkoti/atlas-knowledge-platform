from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_all_up():
    """Health endpoint returns 200 with status UP when all services are healthy."""
    mock_vector_store = MagicMock()
    mock_vector_store.heartbeat.return_value = True

    mock_llm = MagicMock()
    mock_llm.health_check = AsyncMock(return_value=True)

    with (
        patch("app.api.v1.health.get_vector_store", return_value=mock_vector_store),
        patch("app.api.v1.health.get_llm", return_value=mock_llm),
    ):
        response = client.get("/api/v1/health/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UP"
    assert body["services"]["chromadb"] == "UP"
    assert body["services"]["llm"] == "UP"


def test_health_check_service_down():
    """Health endpoint returns 200 with status DOWN when a service is unhealthy."""
    mock_vector_store = MagicMock()
    mock_vector_store.heartbeat.return_value = False

    mock_llm = MagicMock()
    mock_llm.health_check = AsyncMock(return_value=True)

    with (
        patch("app.api.v1.health.get_vector_store", return_value=mock_vector_store),
        patch("app.api.v1.health.get_llm", return_value=mock_llm),
    ):
        response = client.get("/api/v1/health/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DOWN"
    assert body["services"]["chromadb"] == "DOWN"
