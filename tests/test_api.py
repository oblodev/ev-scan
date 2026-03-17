"""
Tests fuer die FastAPI-Endpoints.

FastAPI stellt einen TestClient bereit, der HTTP-Requests simuliert.
Damit koennen wir die API testen OHNE einen echten Server zu starten.

Der TestClient macht alles im Speicher – kein Port, kein Netzwerk.
Das ist schneller und zuverlaessiger als echte HTTP-Calls.

Wichtig: Diese Tests testen die API-Schicht (Routing, Validierung,
Error Handling), NICHT die RAG-Logik. Die RAG Chain braucht Ollama
und ChromaDB, die in Unit Tests nicht verfuegbar sind.
"""

from fastapi.testclient import TestClient

from app.main import app

# TestClient simuliert HTTP-Requests gegen unsere FastAPI-App
client = TestClient(app)


class TestRootEndpoint:
    """Tests fuer GET /"""

    def test_root_returns_200(self) -> None:
        """Root-Endpoint gibt 200 OK zurueck."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome(self) -> None:
        """Root-Endpoint gibt eine Willkommensnachricht zurueck."""
        response = client.get("/")
        data = response.json()
        assert "nachricht" in data
        assert "Willkommen" in data["nachricht"]


class TestHealthEndpoint:
    """Tests fuer GET /api/v1/health"""

    def test_health_endpoint(self) -> None:
        """Health-Endpoint gibt immer 200 zurueck (auch ohne Ollama)."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_status(self) -> None:
        """Health-Endpoint hat ein 'status' Feld."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_ollama_field(self) -> None:
        """Health-Endpoint hat ein 'ollama' Feld (true oder false)."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "ollama" in data
        assert isinstance(data["ollama"], bool)


class TestModelsEndpoint:
    """Tests fuer GET /api/v1/models"""

    def test_models_endpoint_returns_list(self) -> None:
        """Models-Endpoint gibt eine Liste zurueck."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestAnalyzeEndpoint:
    """Tests fuer POST /api/v1/analyze"""

    def test_analyze_returns_valid_response(self) -> None:
        """Analyze gibt eine gueltige Response zurueck (auch ohne Ollama).

        Ohne Ollama kommt die "keine Daten"-Response oder ein 503,
        beides ist OK – wichtig ist dass kein 500 Server Error kommt.
        """
        response = client.post(
            "/api/v1/analyze",
            json={
                "modell": "Tesla Model 3",
                "baujahr": 2021,
                "km_stand": 50000,
            },
        )
        # Entweder 200 (Fallback-Response) oder 503 (Ollama down)
        assert response.status_code in (200, 503)

    def test_analyze_response_has_required_fields(self) -> None:
        """Wenn 200, hat die Response alle Pflichtfelder."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "modell": "Tesla Model 3",
                "baujahr": 2021,
                "km_stand": 50000,
            },
        )
        if response.status_code == 200:
            data = response.json()
            assert "modell" in data
            assert "baujahr" in data
            assert "km_stand" in data
            assert "risiko_bewertung" in data
            assert "zusammenfassung" in data
            assert "rueckrufe" in data
            assert "schwachstellen" in data
            assert "checkliste" in data

    def test_analyze_validates_baujahr_too_old(self) -> None:
        """Baujahr vor 2010 wird mit 422 Validation Error abgelehnt."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "modell": "Tesla Model 3",
                "baujahr": 2005,
                "km_stand": 50000,
            },
        )
        assert response.status_code == 422

    def test_analyze_validates_negative_km(self) -> None:
        """Negativer Kilometerstand wird mit 422 abgelehnt."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "modell": "Tesla Model 3",
                "baujahr": 2021,
                "km_stand": -1,
            },
        )
        assert response.status_code == 422

    def test_analyze_validates_missing_modell(self) -> None:
        """Fehlendes Modell wird mit 422 abgelehnt."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "baujahr": 2021,
                "km_stand": 50000,
            },
        )
        assert response.status_code == 422

    def test_analyze_validates_empty_modell(self) -> None:
        """Leeres Modell wird mit 422 abgelehnt."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "modell": "",
                "baujahr": 2021,
                "km_stand": 50000,
            },
        )
        assert response.status_code == 422
