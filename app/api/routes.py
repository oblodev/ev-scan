"""
API-Endpoints fuer den EV-Gebrauchtwagen-Berater.

Hier sind alle Endpoints definiert, gruppiert in einem Router.

Was ist ein Endpoint?
---------------------
Ein Endpoint ist eine URL + HTTP-Methode, auf die die API reagiert.
z.B. GET /api/v1/health oder POST /api/v1/analyze

Was ist ein Router?
-------------------
Ein Router ist wie ein "Mini-App" der Endpoints buendelt.
Vorteil: Wir koennen Endpoints thematisch gruppieren und in eigene
Dateien auslagern, statt alles in main.py zu packen.

Was macht @router.post()?
-------------------------
Das ist ein Decorator. Er registriert die Funktion darunter als
Handler fuer eine bestimmte HTTP-Methode und URL.
- @router.get("/health") -> reagiert auf GET /health
- @router.post("/analyze") -> reagiert auf POST /analyze

GET = Daten abfragen (keine Aenderung auf dem Server)
POST = Daten senden / eine Aktion ausloesen
"""

import logging

import httpx
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ModelInfo,
    Quelle,
    Rueckruf,
    Schwachstelle,
)

logger = logging.getLogger(__name__)

# Router-Instanz erstellen
# tags werden in der Swagger UI als Gruppierung angezeigt
router = APIRouter(tags=["EV-Scan"])


@router.get("/health")
async def health_check() -> dict:
    """Prueft ob die API und Ollama erreichbar sind.

    Warum async? Weil wir auf eine externe Antwort warten (Ollama).
    Mit async blockiert das Warten nicht andere Anfragen.

    Ablauf:
    1. Schicke eine Anfrage an Ollama
    2. Wenn Antwort kommt -> ollama: true
    3. Wenn Fehler (Timeout, Connection refused) -> ollama: false
    """
    ollama_ok = False

    try:
        # httpx.AsyncClient ist wie ein Browser der HTTP-Anfragen macht
        # "async with" stellt sicher, dass die Verbindung sauber geschlossen wird
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Ollama hat einen eigenen Health-Endpoint
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = response.status_code == 200
    except httpx.ConnectError:
        # Ollama laeuft nicht oder ist nicht erreichbar
        logger.warning("Ollama ist nicht erreichbar unter %s", settings.ollama_base_url)
    except httpx.TimeoutException:
        # Ollama antwortet zu langsam
        logger.warning("Timeout beim Verbinden mit Ollama")

    return {"status": "ok", "ollama": ollama_ok}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_vehicle(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analysiert ein gebrauchtes Elektrofahrzeug.

    Nimmt Modell, Baujahr und km_stand entgegen und gibt eine
    vollstaendige Analyse zurueck.

    AKTUELL: Gibt Dummy-Daten zurueck. Wird spaeter mit dem
    RAG-System verbunden (Schritt 8).

    Was passiert bei @router.post("/analyze")?
    1. FastAPI empfaengt den POST-Request mit JSON-Body
    2. Pydantic validiert den Body automatisch gegen AnalyzeRequest
    3. Bei ungueltigem Input -> automatisch 422 Validation Error
    4. Bei gueltigem Input -> diese Funktion wird aufgerufen
    5. Die Rueckgabe wird gegen AnalyzeResponse validiert und als JSON gesendet
    """
    logger.info(
        "Analyse-Anfrage: modell=%s, baujahr=%d, km_stand=%d",
        request.modell,
        request.baujahr,
        request.km_stand,
    )

    # === Dummy-Response (wird spaeter durch echte RAG-Analyse ersetzt) ===
    return AnalyzeResponse(
        modell=request.modell,
        baujahr=request.baujahr,
        km_stand=request.km_stand,
        risiko_bewertung="gelb",
        zusammenfassung=(
            f"Dummy-Analyse fuer {request.modell} ({request.baujahr}, "
            f"{request.km_stand:,} km). Wird spaeter durch RAG ersetzt."
        ),
        rueckrufe=[
            Rueckruf(
                beschreibung="[DUMMY] Beispiel-Rueckruf: Software-Update Batteriemanagementsystem",
                schwere="mittel",
            ),
        ],
        schwachstellen=[
            Schwachstelle(
                problem="[DUMMY] Beispiel: Ladegeschwindigkeit nimmt bei hohem km-Stand ab",
                schwere="niedrig",
                haeufigkeit="gelegentlich",
            ),
        ],
        checkliste=[
            "12V-Batterie pruefen",
            "Ladeanschluss auf Beschaedigungen pruefen",
            "Batterie-Gesundheit (SoH) auslesen lassen",
            "Probefahrt: Rekuperation testen",
            "Alle Rueckrufe beim Haendler abfragen",
        ],
        quellen=[
            Quelle(source="adac", doc_type="testbericht"),
            Quelle(source="kba", doc_type="rueckruf"),
        ],
    )


@router.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    """Gibt die verfuegbaren Fahrzeugmodelle in der Wissensbasis zurueck.

    AKTUELL: Hardcoded Liste mit 3 Modellen.
    Wird spaeter aus ChromaDB geladen.

    response_model=list[ModelInfo] sagt FastAPI:
    "Die Antwort ist eine Liste von ModelInfo-Objekten."
    Das erscheint in der Swagger-Doku und wird automatisch validiert.
    """
    # Dummy-Daten: Die 3 beliebtesten E-Autos im DACH-Raum
    return [
        ModelInfo(
            modell="Tesla Model 3",
            hersteller="Tesla",
            docs_count=12,
        ),
        ModelInfo(
            modell="VW ID.3",
            hersteller="Volkswagen",
            docs_count=8,
        ),
        ModelInfo(
            modell="Hyundai Ioniq 5",
            hersteller="Hyundai",
            docs_count=6,
        ),
    ]
