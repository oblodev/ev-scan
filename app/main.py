"""
FastAPI Hauptanwendung fuer den EV-Gebrauchtwagen-Berater.

Das hier ist der Einstiegspunkt der API. Hier wird:
1. Die FastAPI-Instanz erstellt (= unsere App)
2. CORS Middleware konfiguriert
3. Der Router mit allen Endpoints eingebunden
4. Ein Root-Endpoint definiert

Gestartet wird die App mit:
    uvicorn app.main:app --reload

--reload bedeutet: Server startet automatisch neu wenn sich Code aendert.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.knowledge import router as knowledge_router
from app.api.routes import router

# Logging konfigurieren, damit wir sehen was passiert
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === FastAPI-Instanz erstellen ===
# title und description erscheinen in der Swagger UI unter /docs
app = FastAPI(
    title="EV Gebrauchtwagen-Berater",
    description=(
        "KI-gestuetzter Kaufberater fuer gebrauchte Elektroautos im DACH-Raum. "
        "Gibt Rueckrufe, Schwachstellen, Risikobewertung und Besichtigungs-Checkliste."
    ),
    version="0.1.0",
)


# === CORS Middleware ===
#
# CORS = Cross-Origin Resource Sharing
#
# Was ist das Problem?
# Browser blockieren standardmaessig Anfragen von einer Webseite an eine
# andere Domain. Das ist ein Sicherheitsfeature namens "Same-Origin Policy".
#
# Beispiel: Unser Streamlit-Frontend laeuft auf http://localhost:8501,
# unsere API auf http://localhost:8000. Das sind verschiedene Origins
# (unterschiedlicher Port = unterschiedliche Origin).
# Ohne CORS wuerde der Browser die Anfragen vom Frontend an die API blockieren.
#
# Was macht die Middleware?
# Sie fuegt HTTP-Header hinzu (z.B. Access-Control-Allow-Origin),
# die dem Browser sagen: "Es ist OK, Anfragen von diesen Origins anzunehmen."
#
# allow_origins=["*"] bedeutet: Alle Origins erlaubt.
# Das ist OK fuer Entwicklung, fuer Produktion sollte man die
# erlaubten Origins einschraenken (z.B. nur das eigene Frontend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Alle Origins erlaubt (nur fuer Entwicklung!)
    allow_credentials=True,
    allow_methods=["*"],       # Alle HTTP-Methoden erlaubt (GET, POST, etc.)
    allow_headers=["*"],       # Alle Header erlaubt
)


# === Router einbinden ===
#
# Was ist ein Router?
# Ein Router gruppiert zusammengehoerige Endpoints. Statt alle Endpoints
# direkt in main.py zu definieren, lagern wir sie in routes.py aus.
# Das haelt den Code uebersichtlich: main.py kuemmert sich um App-Setup,
# routes.py um die eigentlichen Endpoints.
#
# prefix="/api/v1" bedeutet: Alle Endpoints im Router bekommen
# automatisch /api/v1 vorangestellt.
# Also wird @router.get("/health") zu GET /api/v1/health
#
# Das "v1" ist Versionierung: Falls wir die API spaeter aendern,
# koennen wir einen neuen Router mit /api/v2 hinzufuegen,
# ohne die alte Version kaputtzumachen.
app.include_router(router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    """Willkommensnachricht auf der Startseite.

    Das ist der einfachste Endpoint: Einfach eine GET-Anfrage auf /
    die eine JSON-Nachricht zurueckgibt.

    @app.get("/") ist ein Decorator. Er sagt FastAPI:
    "Wenn eine GET-Anfrage auf / kommt, fuehre diese Funktion aus."
    """
    return {
        "nachricht": "Willkommen beim EV Gebrauchtwagen-Berater!",
        "docs": "/docs",
    }
