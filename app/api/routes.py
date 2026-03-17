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

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.rag_chain import RAGChain
from app.core.vector_store import VectorStore
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

# Router-Instanz erstellen
# tags werden in der Swagger UI als Gruppierung angezeigt
router = APIRouter(tags=["EV-Scan"])

# RAG Chain und VectorStore werden beim ersten Request initialisiert
# (Lazy Init), damit die App auch startet wenn Ollama nicht laeuft.
# Wuerde man sie hier direkt erstellen, muesste Ollama beim App-Start
# erreichbar sein, was in der Entwicklung laestig ist.
_rag_chain: RAGChain | None = None
_vector_store: VectorStore | None = None


def _get_rag_chain() -> RAGChain:
    """Gibt die RAG Chain zurueck (erstellt sie beim ersten Aufruf)."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def _get_vector_store() -> VectorStore:
    """Gibt den VectorStore zurueck (erstellt ihn beim ersten Aufruf)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


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

    Nimmt Modell, Baujahr und km_stand entgegen und fuehrt eine
    RAG-basierte Analyse durch:
    1. Relevante Chunks aus ChromaDB holen
    2. Kontext an Mistral uebergeben
    3. Strukturierte Analyse zurueckgeben

    Bei Fehlern (Ollama down, ChromaDB leer) gibt es eine sinnvolle
    Fehlermeldung statt eines Crashes.
    """
    logger.info(
        "Analyse-Anfrage: modell=%s, baujahr=%d, km_stand=%d",
        request.modell,
        request.baujahr,
        request.km_stand,
    )

    try:
        rag = _get_rag_chain()
        return rag.analyze(
            modell=request.modell,
            baujahr=request.baujahr,
            km_stand=request.km_stand,
        )
    except ConnectionError as e:
        # Ollama oder ChromaDB nicht erreichbar
        logger.error("Verbindungsfehler: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "Analyse-Service nicht verfuegbar. "
                "Bitte sicherstellen, dass Ollama laeuft (ollama serve) "
                "und der Ingest durchgefuehrt wurde (python -m app.core.ingest)."
            ),
        ) from e
    except Exception as e:
        logger.error("Unerwarteter Fehler bei Analyse: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler bei der Analyse. Bitte spaeter erneut versuchen.",
        ) from e


@router.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    """Gibt die verfuegbaren Fahrzeugmodelle in der Wissensbasis zurueck.

    Liest die Modelle aus ChromaDB. Wenn ChromaDB leer oder nicht
    erreichbar ist, wird eine leere Liste zurueckgegeben.
    """
    try:
        store = _get_vector_store()
        modelle = store.get_all_models()

        # Fuer jedes Modell zaehlen wir die Chunks in der DB
        result: list[ModelInfo] = []
        for modell_name in modelle:
            # Hersteller aus dem ersten Wort ableiten (vereinfacht)
            # Bei "VW ID.3" -> "VW", bei "Tesla Model 3" -> "Tesla"
            hersteller = modell_name.split()[0] if modell_name else "Unbekannt"

            # Chunks fuer dieses Modell zaehlen
            all_docs = store.collection.get(
                where={"modell": modell_name},
                include=[],
            )
            docs_count = len(all_docs["ids"])

            result.append(ModelInfo(
                modell=modell_name,
                hersteller=hersteller,
                docs_count=docs_count,
            ))

        return result

    except Exception as e:
        logger.error("Fehler beim Laden der Modelle: %s", e)
        # Leere Liste statt Fehler, damit das Frontend nicht crasht
        return []
