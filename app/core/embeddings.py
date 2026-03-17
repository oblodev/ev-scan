"""
Embeddings: Wandelt Text in Zahlenvektoren um (ueber Ollama).

Was ist ein Embedding?
---------------------
Ein Embedding ist eine Liste von Zahlen (z.B. 768 Stueck), die den
"Sinn" eines Textes repraesentiert. Man kann sich das als Koordinate
in einem hochdimensionalen Raum vorstellen.

Warum sind Embeddings nuetzlich?
- Texte mit aehnlicher Bedeutung haben aehnliche Vektoren
- "12V-Batterie defekt" und "Bordbatterie kaputt" haben AEHNLICHE Vektoren,
  obwohl kein einziges Wort gleich ist
- "12V-Batterie defekt" und "Lackqualitaet schlecht" haben UNTERSCHIEDLICHE Vektoren
- Diese Aehnlichkeit messen wir mit Cosinus-Aehnlichkeit (Winkel zwischen Vektoren)

Wie funktioniert das technisch?
1. Wir schicken einen Text an Ollama (Modell: nomic-embed-text)
2. Ollama berechnet den Vektor (768 Dimensionen bei nomic-embed-text)
3. Wir speichern diesen Vektor in ChromaDB
4. Bei einer Suche berechnen wir den Vektor der Frage
5. ChromaDB findet die Vektoren (= Texte) die am aehnlichsten sind

Warum Ollama statt OpenAI?
- Laeuft lokal -> DSGVO-konform, keine Daten verlassen den Rechner
- Kostenlos, keine API-Keys noetig
- nomic-embed-text ist klein (274 MB) und schnell, auch ohne GPU
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Erzeugt Embeddings ueber die Ollama REST API.

    Nutzt das nomic-embed-text Modell, das speziell fuer Embeddings
    trainiert wurde. Es erzeugt 768-dimensionale Vektoren.

    Warum eine Klasse statt einfacher Funktionen?
    Weil wir die Konfiguration (URL, Modell) nur einmal setzen wollen
    und den httpx-Client wiederverwenden koennen.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialisiert die Embedding-Klasse.

        Args:
            base_url: Ollama API URL (Default aus Config)
            model: Name des Embedding-Modells (Default aus Config)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embedding_model

        # Der Endpoint fuer Embeddings bei Ollama
        self.embed_url = f"{self.base_url}/api/embeddings"

        logger.info(
            "OllamaEmbeddings initialisiert: model=%s, url=%s",
            self.model,
            self.embed_url,
        )

    def embed_text(self, text: str) -> list[float]:
        """Erzeugt einen Embedding-Vektor fuer einen einzelnen Text.

        Schickt den Text an Ollama und bekommt eine Liste von Zahlen zurueck.

        Args:
            text: Der Text der in einen Vektor umgewandelt werden soll

        Returns:
            Liste von Floats (z.B. 768 Zahlen bei nomic-embed-text)

        Raises:
            ConnectionError: Wenn Ollama nicht erreichbar ist
            RuntimeError: Wenn Ollama einen Fehler zurueckgibt
        """
        try:
            # Synchroner HTTP-Call an Ollama
            # Wir nutzen sync statt async, weil Embedding-Erzeugung
            # oft beim Ingest passiert (nicht im API-Request-Kontext)
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.embed_url,
                    json={
                        "model": self.model,
                        "prompt": text,
                    },
                )

            # Fehler von Ollama abfangen (z.B. Modell nicht geladen)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama Embedding-Fehler (HTTP {response.status_code}): "
                    f"{response.text}"
                )

            data = response.json()
            embedding = data["embedding"]

            logger.debug(
                "Embedding erstellt: %d Zeichen Text -> %d Dimensionen",
                len(text),
                len(embedding),
            )

            return embedding

        except httpx.ConnectError as e:
            # Ollama laeuft nicht oder ist nicht erreichbar
            logger.error(
                "Ollama nicht erreichbar unter %s. "
                "Ist Ollama gestartet? (ollama serve)",
                self.base_url,
            )
            raise ConnectionError(
                f"Ollama nicht erreichbar unter {self.base_url}. "
                f"Starte Ollama mit: ollama serve"
            ) from e

        except httpx.TimeoutException as e:
            logger.error("Timeout beim Embedding-Aufruf (>30s)")
            raise ConnectionError(
                "Ollama antwortet nicht (Timeout). "
                "Eventuell ist das Modell noch nicht geladen: "
                f"ollama pull {self.model}"
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Erzeugt Embeddings fuer mehrere Texte.

        Ruft embed_text() fuer jeden Text einzeln auf.
        Ollama hat keinen Batch-Endpoint, deshalb sequentiell.

        Auf unserem Lenovo M920q (kein GPU) dauert ein Embedding ca. 0.5-1s,
        also ca. 45 Sekunden fuer unsere 44 Chunks. Das ist akzeptabel
        fuer den einmaligen Ingest.

        Args:
            texts: Liste von Texten

        Returns:
            Liste von Embedding-Vektoren (gleiche Reihenfolge wie Input)
        """
        embeddings: list[list[float]] = []

        for i, text in enumerate(texts):
            logger.info("Embedding %d/%d ...", i + 1, len(texts))
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        logger.info("Alle %d Embeddings erstellt", len(embeddings))
        return embeddings
