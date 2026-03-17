"""
Vector Store: Speichert und sucht Chunks in ChromaDB.

Was ist eine Vektordatenbank?
-----------------------------
Eine normale Datenbank sucht nach exakten Treffern:
  SELECT * FROM autos WHERE modell = 'Tesla Model 3'

Eine Vektordatenbank sucht nach AEHNLICHKEIT:
  "Finde die 5 Texte, die am aehnlichsten zu meiner Frage sind"

Wie funktioniert die Aehnlichkeitssuche?
1. Jeder Text wird als Vektor (Liste von Zahlen) gespeichert
2. Die Frage wird ebenfalls in einen Vektor umgewandelt
3. ChromaDB berechnet den Abstand zwischen Frage-Vektor und allen
   gespeicherten Vektoren (Cosinus-Aehnlichkeit)
4. Die naechsten Nachbarn (= aehnlichsten Vektoren) werden zurueckgegeben

Was ist der where_filter?
-------------------------
Zusaetzlich zur Vektorsuche koennen wir nach Metadaten filtern.
Beispiel: where={"modell": "Tesla Model 3"} sucht NUR in Chunks
die zum Tesla Model 3 gehoeren. Das kombiniert:
- Semantische Suche (Vektoraehnlichkeit) fuer "WAS ist relevant?"
- Metadaten-Filter fuer "In WELCHEM Bereich suchen?"

Warum ChromaDB?
- Open Source, laeuft lokal (DSGVO-konform)
- Einfache Python API, wenig Boilerplate
- Persistent auf Festplatte (ueberlebt Neustarts)
- Kann spaeter auch als Server laufen (Docker)
"""

from __future__ import annotations

import logging

import chromadb

from app.config import settings
from app.core.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper um ChromaDB fuer unsere EV-Wissensbasis.

    Kapselt die ChromaDB-Operationen: Dokumente speichern, suchen,
    und Metadaten abfragen. Nutzt OllamaEmbeddings fuer die
    Vektorisierung.
    """

    def __init__(self) -> None:
        """Stellt die Verbindung zu ChromaDB her und laedt/erstellt die Collection.

        Zwei Modi:
        - "local": PersistentClient, speichert auf Festplatte (Entwicklung)
        - "server": HttpClient, verbindet sich zu einem ChromaDB-Server (Docker)

        Eine "Collection" in ChromaDB ist wie eine Tabelle in SQL:
        Sie speichert Dokumente, Embeddings und Metadaten zusammen.
        get_or_create_collection() erstellt sie beim ersten Mal und
        laedt sie bei jedem weiteren Start.
        """
        if settings.chroma_mode == "server":
            # Server-Modus: ChromaDB laeuft als eigener Container/Prozess
            # HttpClient verbindet sich ueber HTTP (wie ein Browser)
            logger.info(
                "Verbinde mit ChromaDB-Server: %s:%d",
                settings.chroma_host,
                settings.chroma_port,
            )
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        else:
            # Lokaler Modus: Daten direkt auf der Festplatte
            logger.info(
                "Verbinde mit ChromaDB (persistent: %s)",
                settings.chroma_persist_dir,
            )
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
            )

        # Collection holen oder erstellen
        # cosine = Cosinus-Aehnlichkeit als Distanzmetrik
        # (Standard und beste Wahl fuer Textembeddings)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding-Klasse fuer Vektorisierung
        self.embeddings = OllamaEmbeddings()

        logger.info(
            "Collection '%s' geladen: %d Dokumente vorhanden",
            settings.collection_name,
            self.collection.count(),
        )

    def add_documents(self, chunks: list[dict]) -> None:
        """Speichert Chunks mit Embeddings in ChromaDB.

        Ablauf fuer jeden Chunk:
        1. Text -> Embedding (ueber Ollama)
        2. Embedding + Text + Metadaten -> ChromaDB

        Die ID wird aus Metadaten generiert, damit derselbe Chunk
        nicht doppelt gespeichert wird (Idempotenz).

        Args:
            chunks: Liste von Dicts mit "content" und "metadata"
        """
        if not chunks:
            logger.warning("Keine Chunks zum Speichern")
            return

        logger.info("Speichere %d Chunks in ChromaDB ...", len(chunks))

        # Texte und Metadaten extrahieren
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # IDs generieren: modell_doctype_chunkindex
        # z.B. "Tesla Model 3_rueckruf_0", "Tesla Model 3_rueckruf_1"
        ids = [
            f"{meta['modell']}_{meta['doc_type']}_{meta.get('chunk_index', i)}"
            for i, meta in enumerate(metadatas)
        ]

        # Alle Embeddings auf einmal berechnen
        logger.info("Berechne Embeddings fuer %d Chunks ...", len(texts))
        embeddings = self.embeddings.embed_texts(texts)

        # In ChromaDB speichern
        # upsert statt add: Falls die ID schon existiert, wird ueberschrieben
        # (wichtig beim erneuten Ausfuehren von ingest)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "%d Chunks in ChromaDB gespeichert (Collection: %d gesamt)",
            len(chunks),
            self.collection.count(),
        )

    def query(
        self,
        query_text: str,
        n_results: int | None = None,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """Sucht die relevantesten Chunks zu einer Frage.

        Ablauf:
        1. Frage-Text -> Embedding (ueber Ollama)
        2. ChromaDB sucht die n aehnlichsten Vektoren
        3. Optional: Nur Chunks mit bestimmten Metadaten (where_filter)

        Args:
            query_text: Die Frage des Users (z.B. "Hat der Tesla Model 3 Rueckrufe?")
            n_results: Anzahl Ergebnisse (Default: top_k aus Config)
            where_filter: Metadaten-Filter, z.B. {"modell": "Tesla Model 3"}
                          Schraenkt die Suche auf passende Chunks ein

        Returns:
            Liste von Dicts mit "content", "metadata" und "distance"
            (distance = Abstand, kleiner = aehnlicher)
        """
        if n_results is None:
            n_results = settings.top_k

        logger.info(
            "Suche: '%s' (top %d, filter=%s)",
            query_text[:80],
            n_results,
            where_filter,
        )

        # Frage in Vektor umwandeln
        query_embedding = self.embeddings.embed_text(query_text)

        # ChromaDB-Abfrage
        query_params: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }

        # Optionaler Metadaten-Filter
        # where={"modell": "Tesla Model 3"} -> nur Tesla-Chunks durchsuchen
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        # Ergebnisse in ein lesbares Format bringen
        # ChromaDB gibt verschachtelte Listen zurueck (weil man mehrere
        # Queries gleichzeitig stellen kann). Wir haben nur eine Query,
        # daher nehmen wir immer Index [0].
        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        logger.info("%d Ergebnisse gefunden", len(output))
        return output

    def get_all_models(self) -> list[str]:
        """Gibt alle einzigartigen Modellnamen in der Datenbank zurueck.

        Liest alle Metadaten aus der Collection und extrahiert die
        eindeutigen Modellnamen. Nuetzlich fuer den /models Endpoint.
        """
        # Alle Dokumente abrufen (nur Metadaten, nicht den Text)
        all_docs = self.collection.get(include=["metadatas"])

        # Eindeutige Modellnamen sammeln
        modelle: set[str] = set()
        for meta in all_docs["metadatas"]:
            if "modell" in meta:
                modelle.add(meta["modell"])

        return sorted(modelle)

    def count(self) -> int:
        """Gibt die Anzahl der Dokumente in der Collection zurueck."""
        return self.collection.count()

    def clear(self) -> None:
        """Loescht alle Dokumente aus der Collection.

        Nuetzlich beim Re-Ingest: Erst alles loeschen, dann neu befuellen.
        """
        # Collection loeschen und neu erstellen
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' geleert", settings.collection_name)
