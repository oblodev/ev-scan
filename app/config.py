"""
Konfiguration fuer den EV-Gebrauchtwagen-Berater.

Pydantic BaseSettings laedt Einstellungen automatisch aus:
1. Umgebungsvariablen (hoechste Prioritaet)
2. .env Datei (falls vorhanden)
3. Default-Werte (niedrigste Prioritaet)

Warum Pydantic statt einfacher Variablen?
- Automatische Typ-Validierung (z.B. Port muss int sein)
- Automatisches Laden aus .env Dateien
- Zentrale, typsichere Konfiguration an einem Ort
- Fehler werden sofort beim Start erkannt, nicht erst zur Laufzeit
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === LLM-Konfiguration (Ollama) ===

    # Basis-URL des Ollama-Servers, der die LLM-Modelle bereitstellt
    ollama_base_url: str = "http://localhost:11434"

    # Name des Sprachmodells fuer Chat-Antworten (z.B. mistral, llama3, phi3)
    llm_model: str = "mistral"

    # Name des Embedding-Modells, das Text in Vektoren umwandelt
    # Diese Vektoren ermoeglichen die semantische Suche
    embedding_model: str = "nomic-embed-text"

    # === Vektor-Datenbank (ChromaDB) ===

    # Pfad zum lokalen ChromaDB-Speicher (persistent auf Festplatte)
    # Im Entwicklungsmodus speichern wir lokal statt ueber einen Server
    chroma_persist_dir: str = "./chroma_data"

    # Host und Port des ChromaDB-Servers (fuer spaetere Docker-Nutzung)
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Name der Collection (= "Tabelle") in ChromaDB
    collection_name: str = "ev_knowledge_base"

    # === RAG-Parameter (Retrieval Augmented Generation) ===

    # Chunk-Groesse: Wie viele Zeichen pro Textabschnitt beim Aufteilen
    # Kleinere Chunks = praezisere Suche, aber weniger Kontext
    # Groessere Chunks = mehr Kontext, aber ungenauere Suche
    chunk_size: int = 500

    # Ueberlappung zwischen Chunks in Zeichen
    # Verhindert, dass wichtige Informationen an Chunk-Grenzen verloren gehen
    chunk_overlap: int = 50

    # Anzahl der relevantesten Textabschnitte, die dem LLM als Kontext
    # mitgegeben werden (Top-K Retrieval)
    top_k: int = 5

    class Config:
        # Pydantic laedt automatisch Werte aus dieser Datei
        # Werte in .env ueberschreiben die Defaults oben
        env_file = ".env"


# Singleton-Instanz: wird einmal erstellt und ueberall wiederverwendet
# Beim Import von settings werden sofort alle Werte geladen und validiert
settings = Settings()
