# EV-Scan – KI-Kaufberater fuer gebrauchte Elektroautos

EV-Scan ist ein RAG-basierter Kaufberater fuer gebrauchte Elektroautos im DACH-Raum. Der User gibt Modell, Baujahr und Kilometerstand ein und erhaelt einen Report mit bekannten Rueckrufen, Schwachstellen, Risikobewertung und einer Besichtigungs-Checkliste. Alle Daten werden lokal verarbeitet – kein Cloud-Service, DSGVO-konform.

## Demo

> Screenshot hier einfuegen: `frontend/screenshots/demo.png`

```
Eingabe:  Tesla Model 3 | Baujahr 2020 | 75.000 km
Ausgabe:  Risiko GELB – 3 Rueckrufe, 5 Schwachstellen, 5-Punkte-Checkliste
```

## Architektur

```
                         +---------------------+
                         |     Streamlit UI     |  :8501
                         |  (frontend/app.py)   |
                         +---------+-----------+
                                   |
                              HTTP POST
                              /api/v1/analyze
                                   |
                         +---------v-----------+
                         |    FastAPI Backend   |  :8000
                         |   (app/main.py)      |
                         +---------+-----------+
                                   |
                    +--------------+--------------+
                    |                             |
          +---------v---------+         +---------v---------+
          |     ChromaDB      |         |   Ollama (Host)   |
          |  Vektordatenbank  |         |  Mistral 7B (LLM) |
          |    :8080          |         |  nomic-embed-text  |
          +-------------------+         |    :11434          |
                                        +-------------------+
```

**Datenfluss bei einer Analyse:**

```
1. User gibt Modell/Baujahr/km ein
2. Metadata Filter extrahiert Fahrzeug-Info aus der Anfrage
3. VectorStore: 3 gezielte Queries an ChromaDB (Rueckrufe, Schwachstellen, Datenblatt)
4. RAG Chain: Kontext + System-Prompt an Mistral senden
5. JSON-Response parsen und als strukturierte Analyse zurueckgeben
```

## Tech-Stack

| Komponente | Technologie | Warum |
|------------|------------|-------|
| Backend API | FastAPI + Uvicorn | Async, automatische OpenAPI-Doku, Pydantic-Integration |
| LLM | Ollama + Mistral 7B | Laeuft lokal, DSGVO-konform, kein API-Key noetig |
| Embeddings | nomic-embed-text | Klein (274 MB), schnell auf CPU, 768 Dimensionen |
| Vektordatenbank | ChromaDB | Open Source, einfache API, persistent, Docker-ready |
| Frontend | Streamlit | Schnelles Prototyping, Python-only, kein JS noetig |
| HTTP Client | httpx | Async + Sync Support, direkte Ollama API Calls |
| Validierung | Pydantic | Type Safety, automatische Serialisierung |
| Container | Docker + Compose | 3-Service-Setup, reproduzierbar, ein Befehl zum Starten |
| Tests | pytest | 36 Unit Tests, FastAPI TestClient |

## Quick Start

### Voraussetzungen

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installiert
- [Ollama](https://ollama.ai/) installiert und gestartet

### 3 Befehle zum Starten

```bash
# 1. Ollama-Modelle laden (einmalig)
ollama pull mistral && ollama pull nomic-embed-text

# 2. Container bauen und starten
docker compose up --build -d

# 3. Wissensbasis in ChromaDB laden (einmalig)
docker compose exec api python -m app.core.ingest
```

Danach oeffnen:
- **Frontend:** http://localhost:8501
- **API Docs (Swagger):** http://localhost:8888/docs
- **ChromaDB:** http://localhost:8080

### Ohne Docker (Entwicklung)

```bash
pip install -r requirements.txt
ollama serve

# Terminal 1: Backend
python3 -m uvicorn app.main:app --reload

# Terminal 2: Ingest (einmalig)
python3 -m app.core.ingest

# Terminal 3: Frontend
python3 -m streamlit run frontend/app.py
```

## API-Dokumentation

| Methode | Endpoint | Beschreibung |
|---------|----------|-------------|
| `GET` | `/` | Willkommensnachricht |
| `GET` | `/api/v1/health` | Health Check (prueft Ollama-Verbindung) |
| `GET` | `/api/v1/models` | Liste aller Fahrzeugmodelle in der Wissensbasis |
| `POST` | `/api/v1/analyze` | Fahrzeuganalyse (Modell, Baujahr, km) |

### Beispiel: Analyse-Request

```bash
curl -X POST http://localhost:8888/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"modell": "Tesla Model 3", "baujahr": 2020, "km_stand": 75000}'
```

### Beispiel: Response

```json
{
  "modell": "Tesla Model 3",
  "baujahr": 2020,
  "km_stand": 75000,
  "risiko_bewertung": "gelb",
  "zusammenfassung": "Der Tesla Model 3 Baujahr 2020 hat 3 bekannte Rueckrufe...",
  "rueckrufe": [
    {"beschreibung": "Hinterachse: Schraubverbindung pruefen", "schwere": "hoch"}
  ],
  "schwachstellen": [
    {"problem": "12V-Batterie Ausfall nach 2-3 Jahren", "schwere": "hoch", "haeufigkeit": "gelegentlich"}
  ],
  "checkliste": [
    "12V-Batterie pruefen",
    "Bremsscheiben auf Rost kontrollieren",
    "Batterie-Gesundheit (SoH) auslesen lassen"
  ],
  "quellen": [
    {"source": "kba", "doc_type": "rueckruf"},
    {"source": "adac", "doc_type": "testbericht"}
  ]
}
```

## Datenquellen

| Quelle | Typ | Inhalt | Format |
|--------|-----|--------|--------|
| KBA (Kraftfahrt-Bundesamt) | Rueckrufe | Offizielle Rueckrufaktionen mit Referenznummern | TXT |
| ADAC | Testberichte | Pannenstatistik, Dauertest-Ergebnisse | TXT |
| OEAMTC | Testberichte | Wintertests, Reichweiten-Messungen | TXT |
| Hersteller-Datenblaetter | Technische Daten | Batterie, Reichweite, Ladeleistung | JSON |

Aktuell in der Wissensbasis: **3 Modelle** (Tesla Model 3, VW ID.3, Hyundai Ioniq 5) mit insgesamt **9 Dokumenten** und **44 Chunks**.

## Design-Entscheidungen

### Warum Ollama statt OpenAI?

- **DSGVO-Konformitaet:** Keine Fahrzeugdaten verlassen den Rechner. Bei einem Versicherungs-Kontext (z.B. Helvetia) ist das entscheidend – Kundendaten duerfen nicht an externe APIs gesendet werden.
- **Keine Abhaengigkeit:** Kein API-Key, keine Rate Limits, keine laufenden Kosten.
- **Reproduzierbarkeit:** Gleiche Ergebnisse auf jedem Rechner, keine Modell-Updates durch den Anbieter.
- **Trade-off:** Langsamere Inferenz auf CPU (~20-30s pro Analyse), kleineres Modell (7B vs. GPT-4). Fuer einen Kaufberater ist das akzeptabel.

### Warum kein LangChain?

- **Weniger Abstraktion, mehr Kontrolle:** Die Ollama API hat 2 Endpoints (embeddings, generate). Dafuer brauchen wir kein Framework.
- **Besseres Verstaendnis:** Jeder Schritt der RAG-Pipeline ist explizit im Code sichtbar, nicht hinter Abstraktionen versteckt.
- **Weniger Abhaengigkeiten:** LangChain zieht dutzende Pakete nach sich. Unsere `requirements.txt` hat 11 Eintraege.

### Warum Metadata-Filtering?

ChromaDB unterstuetzt gefilterte Vektorsuche. Statt alle 44 Chunks zu durchsuchen, filtern wir **zuerst** nach Modell und Dokumenttyp:

```python
# Nur Rueckrufe fuer Tesla Model 3 durchsuchen
store.query(
    query_text="Rueckruf",
    where_filter={"$and": [
        {"modell": "Tesla Model 3"},
        {"doc_type": "rueckruf"}
    ]}
)
```

Das verbessert die Praezision (keine VW-Rueckrufe in einer Tesla-Analyse) und die Performance (kleinerer Suchraum).

### Warum verschiedene Chunking-Strategien?

| Dokumenttyp | Strategie | Begruendung |
|-------------|-----------|-------------|
| Datenblaetter (JSON) | Nicht chunken | Kompakte, zusammengehoerige Daten. Batterie-Info ohne Modellname ist nutzlos. |
| Rueckrufe (TXT) | Am Trennzeichen splitten | Jeder Rueckruf ist eigenstaendig. Soll als Ganzes gefunden werden. |
| Schwachstellen (TXT) | Feste Groesse + Overlap | Laengerer Fliesstext. Overlap verhindert Informationsverlust an Chunk-Grenzen. |

## Projektstruktur

```
ev-scan/
├── app/
│   ├── main.py                # FastAPI App + CORS + Router
│   ├── config.py              # Pydantic Settings (Ollama, ChromaDB, RAG)
│   ├── api/
│   │   └── routes.py          # API Endpoints (/health, /analyze, /models)
│   ├── core/
│   │   ├── document_loader.py # Laedt JSON/TXT aus data/processed/
│   │   ├── chunking.py        # 3 Strategien je nach Dokumenttyp
│   │   ├── embeddings.py      # Ollama API (nomic-embed-text)
│   │   ├── vector_store.py    # ChromaDB (local + server mode)
│   │   ├── metadata_filter.py # Keyword + LLM Fahrzeug-Erkennung
│   │   ├── rag_chain.py       # RAG Orchestrierung (Herzstueck)
│   │   └── ingest.py          # Daten-Pipeline: Load → Chunk → Embed → Store
│   └── models/
│       └── schemas.py         # Pydantic Request/Response Modelle
├── frontend/
│   └── app.py                 # Streamlit Web-UI
├── data/processed/            # Wissensbasis (9 Dateien, 3 Modelle)
│   ├── datenblaetter/         # JSON: Technische Daten
│   ├── rueckrufe/             # TXT: KBA-Rueckrufaktionen
│   └── schwachstellen/        # TXT: Schwachstellenanalysen
├── tests/                     # 36 Unit Tests (pytest)
│   ├── test_api.py            # FastAPI Endpoint Tests
│   ├── test_chunking.py       # Chunking-Strategie Tests
│   └── test_metadata_filter.py # Fahrzeug-Erkennung Tests
├── Dockerfile                 # python:3.11-slim, Multi-Stage ready
├── docker-compose.yml         # 3 Services: API, Frontend, ChromaDB
└── requirements.txt           # 11 Abhaengigkeiten (kein LangChain)
```

## Tests

```bash
pytest tests/ -v
```

```
36 passed in 0.60s

tests/test_api.py            12 Tests   API Endpoints, Validierung, Error Handling
tests/test_chunking.py       10 Tests   Chunking-Strategien, Metadaten-Erhalt
tests/test_metadata_filter.py 14 Tests  Modell-Erkennung, Baujahr, Edge Cases
```

Alle Tests laufen ohne Ollama und ChromaDB (reine Unit Tests).

## Limitierungen & Ausblick

### Aktuelle Limitierungen

- **3 Modelle:** Nur Tesla Model 3, VW ID.3, Hyundai Ioniq 5 in der Wissensbasis
- **Keine GPU:** Inferenz auf CPU dauert ~20-30s pro Analyse
- **Kein Caching:** Gleiche Anfrage wird jedes Mal neu berechnet
- **Statische Daten:** Wissensbasis wird manuell gepflegt, kein automatischer Scraper

### Moegliche Erweiterungen

- **Mehr Modelle:** Wissensbasis um weitere E-Autos erweitern (Renault Zoe, BMW iX3, etc.)
- **Preisschaetzung:** Marktpreise aus mobile.de/AutoScout24 einbinden
- **Caching:** Redis fuer haeufige Anfragen (gleiche Modell/Baujahr Kombination)
- **RAG-Evaluierung:** Systematische Bewertung der Antwortqualitaet mit Ground-Truth-Datensatz
- **GPU-Support:** NVIDIA GPU fuer schnellere Inferenz (<5s statt 30s)
- **Automatischer Ingest:** Scraper fuer KBA-Rueckrufe und ADAC-Pannenstatistik

## Hardware

Entwickelt und getestet auf:
- **Lenovo M920q** – Intel i5-8500T, 16 GB RAM, kein GPU, Linux
- Ollama mit Mistral 7B: ~4 GB RAM, ~20-30s pro Analyse
- Ollama mit nomic-embed-text: ~300 MB RAM, ~0.5s pro Embedding

## Lizenz

MIT
