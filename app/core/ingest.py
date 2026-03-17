"""
Ingest-Skript: Laedt alle Dokumente, chunkt sie und speichert sie in ChromaDB.

Das ist die "Daten-Pipeline", die einmal laeuft bevor die API Fragen
beantworten kann. Sie bringt das Wissen in die Vektordatenbank.

Ablauf:
1. Dokumente laden (document_loader) -> 9 Dateien
2. Chunks erstellen (chunking)       -> ~44 Chunks
3. Embeddings berechnen (Ollama)     -> 44 Vektoren a 768 Dimensionen
4. In ChromaDB speichern             -> persistent auf Festplatte

Ausfuehren mit:
    python -m app.core.ingest

Voraussetzungen:
    - Ollama muss laufen: ollama serve
    - Embedding-Modell muss geladen sein: ollama pull nomic-embed-text
"""

import logging
import time

from app.core.chunking import chunk_documents
from app.core.document_loader import load_all_documents
from app.core.vector_store import VectorStore

# Logging konfigurieren: INFO-Level zeigt den Fortschritt
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingest(data_dir: str = "data/processed") -> None:
    """Fuehrt den kompletten Ingest-Prozess durch.

    Args:
        data_dir: Pfad zum Datenverzeichnis
    """
    start_time = time.time()

    # === Schritt 1: Dokumente laden ===
    logger.info("=" * 50)
    logger.info("SCHRITT 1: Dokumente laden")
    logger.info("=" * 50)

    documents = load_all_documents(data_dir)
    if not documents:
        logger.error("Keine Dokumente gefunden in %s - Abbruch!", data_dir)
        return

    # === Schritt 2: Chunking ===
    logger.info("=" * 50)
    logger.info("SCHRITT 2: Chunks erstellen")
    logger.info("=" * 50)

    chunks = chunk_documents(documents)
    if not chunks:
        logger.error("Keine Chunks erstellt - Abbruch!")
        return

    # === Schritt 3 + 4: Embeddings berechnen und in ChromaDB speichern ===
    logger.info("=" * 50)
    logger.info("SCHRITT 3: Embeddings berechnen und in ChromaDB speichern")
    logger.info("=" * 50)

    try:
        store = VectorStore()
        # Alte Daten loeschen, damit kein Muell uebrig bleibt
        store.clear()
        # Neue Chunks speichern (berechnet Embeddings automatisch)
        store.add_documents(chunks)
    except ConnectionError as e:
        logger.error("Ollama nicht erreichbar: %s", e)
        logger.error("Starte Ollama mit: ollama serve")
        logger.error("Lade das Modell mit: ollama pull nomic-embed-text")
        return

    # === Statistiken ===
    elapsed = time.time() - start_time
    modelle = store.get_all_models()

    logger.info("=" * 50)
    logger.info("INGEST ABGESCHLOSSEN")
    logger.info("=" * 50)
    logger.info("Geladen:    %d Dokumente", len(documents))
    logger.info("Erstellt:   %d Chunks", len(chunks))
    logger.info("Gespeichert: %d Vektoren in ChromaDB", store.count())
    logger.info("Modelle:    %d (%s)", len(modelle), ", ".join(modelle))
    logger.info("Dauer:      %.1f Sekunden", elapsed)


if __name__ == "__main__":
    run_ingest()
