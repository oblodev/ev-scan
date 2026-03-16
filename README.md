# VoltCheck – EV Gebrauchtwagen-Berater

KI-gestützter Kaufberater für gebrauchte Elektroautos im DACH-Raum.

## Was macht VoltCheck?

Du gibst **Modell, Baujahr und Kilometerstand** ein und bekommst einen Report mit:
- Bekannte Rückrufe und Schwachstellen
- Risikobewertung
- Besichtigungs-Checkliste

## Tech-Stack

- **Backend:** FastAPI + Python 3.12
- **LLM:** Ollama (Mistral 7B) – läuft komplett lokal, kein OpenAI
- **Vektordatenbank:** ChromaDB
- **Frontend:** Streamlit
- **Infrastruktur:** Docker + docker-compose

## Status

Projekt befindet sich in aktiver Entwicklung.
