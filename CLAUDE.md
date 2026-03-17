# CLAUDE.md
# Projekt: EV-Scan – EV Gebrauchtwagen-Berater

KI-gestützter Kaufberater für gebrauchte Elektroautos im DACH-Raum.
User gibt Modell, Baujahr und Kilometerstand ein, bekommt einen Report mit
Rückrufen, Schwachstellen, Risikobewertung und Besichtigungs-Checkliste.

## Tech-Stack

- Python 3.12
- FastAPI + Uvicorn (Backend API)
- Ollama mit Mistral 7B (LLM) und nomic-embed-text (Embeddings)
- ChromaDB (Vektordatenbank, läuft lokal oder als Container)
- Streamlit (Frontend)
- Docker + docker-compose
- pytest
- httpx (HTTP Client für Ollama API Calls)

## Bewusste Entscheidungen

- **KEIN LangChain.** Wir rufen Ollama und ChromaDB direkt auf. Weniger Abstraktion, mehr Kontrolle und Verständnis.
- **KEIN OpenAI.** Alles läuft lokal über Ollama. DSGVO-konform.
- **Ollama läuft auf dem HOST**, nicht im Container. Die Container greifen über `host.docker.internal:11434` darauf zu.
- **Hardware:** Lenovo M920q, 16 GB RAM, Linux, KEINE GPU. Code muss ressourcenschonend sein.

## Projektstruktur

```
ev-scan/
├── app/
│   ├── main.py              # FastAPI App
│   ├── config.py            # Pydantic Settings
│   ├── api/routes.py        # API Endpoints
│   ├── core/
│   │   ├── document_loader.py
│   │   ├── chunking.py
│   │   ├── embeddings.py    # Direkte Ollama API Calls
│   │   ├── vector_store.py  # ChromaDB Client
│   │   ├── metadata_filter.py
│   │   ├── rag_chain.py     # Orchestrierung
│   │   └── ingest.py        # Daten in ChromaDB laden
│   └── models/schemas.py    # Pydantic Request/Response Modelle
├── frontend/app.py          # Streamlit UI
├── data/processed/          # Wissensbasis (JSON, TXT)
├── tests/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Ollama API Referenz

Embeddings:
```
POST http://localhost:11434/api/embeddings
{"model": "nomic-embed-text", "prompt": "text hier"}
-> {"embedding": [0.1, 0.2, ...]}
```

Text-Generierung:
```
POST http://localhost:11434/api/generate
{"model": "mistral", "prompt": "text hier", "stream": false, "options": {"temperature": 0.1}}
-> {"response": "antwort hier"}
```

## Metadaten-Schema für ChromaDB

Jedes Dokument bekommt diese Metadaten:

- `source`: "adac" | "oeamtc" | "kba" | "carwiki" | "datenblatt"
- `doc_type`: "testbericht" | "rueckruf" | "schwachstelle" | "datenblatt"
- `modell`: z.B. "Tesla Model 3"
- `hersteller`: z.B. "Tesla"

## Code-Stil

- Type Hints überall
- Docstrings auf Deutsch für Funktionen
- Kommentare auf Deutsch (das ist ein Lernprojekt)
- Error Handling: Immer abfangen wenn Ollama oder ChromaDB nicht erreichbar
- Logging mit Python `logging` Modul
- Keine `print()` Statements in Produktionscode

## Wichtige Regeln

- Jede Funktion soll eine Sache tun und diese gut
- Keine globalen Variablen, alles über Config oder Dependency Injection
- Responses immer über Pydantic Schemas, nie rohe Dicts
- Wenn du unsicher bist: Einfacher Code > cleverer Code
- Tests für jede Kernfunktion

## Aktueller Stand

Hier dokumentieren wir den Fortschritt:

- [x] Schritt 1: Projektstruktur + Config
- [x] Schritt 2: Pydantic Schemas
- [x] Schritt 3: Beispiel-Daten
- [x] Schritt 4: FastAPI Grundgerüst
- [x] Schritt 5: Document Loader + Chunking
- [x] Schritt 6: Embeddings + ChromaDB
- [x] Schritt 7: Metadata Filter
- [x] Schritt 8: RAG Chain
- [x] Schritt 9: Streamlit Frontend
- [ ] Schritt 10: Docker
- [ ] Schritt 11: Tests
- [ ] Schritt 12: README
