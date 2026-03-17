# Knowledge-Management Endpoints.
#
# Diese Endpoints machen das System von einem statischen Tool zu einem
# lebendigen System: Ein Fachexperte (z.B. KFZ-Gutachter) kann neues
# Wissen einspeisen, ohne Code zu schreiben oder die Kommandozeile
# zu benutzen. Er kopiert einfach einen Text rein oder laedt eine
# Datei hoch, und das Wissen steht sofort fuer Analysen bereit.
#
# Das ist entscheidend fuer den Praxiseinsatz: Wenn ein neuer Rueckruf
# veroeffentlicht wird, soll ein Sachbearbeiter den Text reinkopieren
# koennen, ohne auf einen Entwickler warten zu muessen.

from __future__ import annotations

import csv
import io
import logging
import hashlib

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pypdf import PdfReader

from app.core.chunking import chunk_documents
from app.core.vector_store import VectorStore
from app.models.schemas import IngestResponse, IngestTextRequest, KnowledgeStats

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Wissens-Management"])

# Lazy Init wie bei den anderen Endpoints
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    """Gibt den VectorStore zurueck (erstellt ihn beim ersten Aufruf)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _ingest_text(
    text: str,
    kategorie: str,
    modell: str,
    quelle: str,
) -> int:
    """Gemeinsame Logik: Text chunken, embedden und in ChromaDB speichern.

    Wird von /ingest/text und /ingest/file verwendet.

    Returns:
        Anzahl erstellter Chunks
    """
    # Dokument-Dict im gleichen Format wie document_loader erstellt
    doc = {
        "content": text,
        "metadata": {
            "source": quelle,
            "doc_type": kategorie,
            "modell": modell,
            "hersteller": modell.split()[0] if modell else "Unbekannt",
        },
    }

    # Chunken (gleiche Strategien wie beim initialen Ingest)
    chunks = chunk_documents([doc])

    if not chunks:
        return 0

    # IDs muessen eindeutig sein – wir nutzen einen Hash des Inhalts,
    # damit derselbe Text nicht doppelt gespeichert wird
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(
            chunk["content"].encode()
        ).hexdigest()[:8]
        chunk["metadata"]["chunk_index"] = i
        # ID-Format: modell_kategorie_hash_index
        chunk["_id"] = f"{modell}_{kategorie}_{content_hash}_{i}"

    # In ChromaDB speichern
    store = _get_vector_store()

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["_id"] for c in chunks]

    embeddings = store.embeddings.embed_texts(texts)

    store.collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "%d Chunks fuer %s (%s) in ChromaDB gespeichert",
        len(chunks),
        modell,
        kategorie,
    )
    return len(chunks)


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Fuegt einen Text in die Wissensbasis ein.

    Der Text wird gechunkt, mit Metadaten versehen, embeddet
    und in ChromaDB gespeichert. Doppelte Texte werden durch
    Content-Hashing in der ID erkannt und ueberschrieben (upsert).
    """
    logger.info(
        "Text-Ingest: modell=%s, kategorie=%s, %d Zeichen",
        request.modell,
        request.kategorie,
        len(request.text),
    )

    try:
        chunks_added = _ingest_text(
            text=request.text,
            kategorie=request.kategorie,
            modell=request.modell,
            quelle=request.quelle,
        )
        return IngestResponse(status="ok", chunks_added=chunks_added)
    except ConnectionError as e:
        logger.error("Ollama nicht erreichbar: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Ollama nicht erreichbar. Bitte sicherstellen, dass Ollama laeuft.",
        ) from e


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    kategorie: str = Form(...),
    modell: str = Form(...),
    quelle: str = Form(default="datei-upload"),
) -> IngestResponse:
    """Laedt eine Datei hoch und fuegt den Inhalt in die Wissensbasis ein.

    Unterstuetzte Formate:
    - TXT: Direkt verarbeiten
    - PDF: Text mit pypdf extrahieren
    - CSV: Jede Zeile als eigenes Dokument (erste Zeile = Header)

    Warum File-Upload?
    Ein Fachexperte hat vielleicht einen PDF-Testbericht vom ADAC
    oder eine CSV mit Rueckrufdaten. Er soll diese Datei einfach
    hochladen koennen, ohne sie erst in Text umwandeln zu muessen.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Kein Dateiname")

    filename = file.filename.lower()
    raw_bytes = await file.read()

    logger.info(
        "Datei-Ingest: %s (%d Bytes), modell=%s, kategorie=%s",
        file.filename,
        len(raw_bytes),
        modell,
        kategorie,
    )

    # Text aus der Datei extrahieren je nach Format
    if filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8")

    elif filename.endswith(".pdf"):
        text = _extract_pdf_text(raw_bytes)
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Die PDF enthaelt keinen extrahierbaren Text "
                    "(evtl. gescanntes Dokument oder Bild-PDF)."
                ),
            )

    elif filename.endswith(".csv"):
        text = _extract_csv_text(raw_bytes)

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dateiformat '{filename.split('.')[-1]}' nicht unterstuetzt. "
                f"Erlaubt: .txt, .pdf, .csv"
            ),
        )

    try:
        chunks_added = _ingest_text(
            text=text,
            kategorie=kategorie,
            modell=modell,
            quelle=quelle or f"datei:{file.filename}",
        )
        return IngestResponse(
            status="ok",
            chunks_added=chunks_added,
            filename=file.filename or "",
        )
    except ConnectionError as e:
        logger.error("Ollama nicht erreichbar: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Ollama nicht erreichbar.",
        ) from e


@router.get("/knowledge/stats", response_model=KnowledgeStats)
async def knowledge_stats() -> KnowledgeStats:
    """Gibt Statistiken ueber die Wissensbasis zurueck.

    Zaehlt Chunks pro Modell und pro Kategorie.
    Nuetzlich fuer die Wissensbasis-Uebersicht im Frontend.
    """
    try:
        store = _get_vector_store()
        all_docs = store.collection.get(include=["metadatas"])

        models: dict[str, int] = {}
        categories: dict[str, int] = {}

        for meta in all_docs["metadatas"]:
            # Pro Modell zaehlen
            m = meta.get("modell", "Unbekannt")
            models[m] = models.get(m, 0) + 1

            # Pro Kategorie zaehlen
            c = meta.get("doc_type", "unbekannt")
            categories[c] = categories.get(c, 0) + 1

        return KnowledgeStats(
            total_chunks=len(all_docs["ids"]),
            models=models,
            categories=categories,
        )
    except Exception as e:
        logger.error("Fehler beim Laden der Stats: %s", e)
        return KnowledgeStats(total_chunks=0)


@router.delete("/knowledge/{modell}")
async def delete_model_knowledge(modell: str) -> dict:
    """Loescht alle Chunks fuer ein bestimmtes Modell.

    Nuetzlich wenn Daten veraltet sind und neu eingespielt werden sollen.
    Der Modellname muss URL-encoded sein (z.B. "Tesla%20Model%203").
    """
    logger.info("Loesche Wissen fuer: %s", modell)

    try:
        store = _get_vector_store()

        # Alle IDs fuer dieses Modell holen
        docs = store.collection.get(
            where={"modell": modell},
            include=[],
        )
        count = len(docs["ids"])

        if count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Keine Daten fuer '{modell}' gefunden.",
            )

        # Chunks loeschen
        store.collection.delete(ids=docs["ids"])

        logger.info("%d Chunks fuer '%s' geloescht", count, modell)
        return {"status": "ok", "deleted": count, "modell": modell}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Fehler beim Loeschen: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Extrahiert Text aus einer PDF-Datei mit pypdf.

    Geht seitenweise durch die PDF und sammelt den Text.
    Bei gescannten PDFs (Bilder statt Text) kommt leerer String zurueck.
    """
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n\n".join(pages)


def _extract_csv_text(raw_bytes: bytes) -> str:
    """Wandelt CSV-Daten in lesbaren Text um.

    Jede Zeile wird zu einem Absatz: "Spalte1: Wert1, Spalte2: Wert2, ..."
    Die erste Zeile wird als Header (Spaltennamen) interpretiert.
    """
    text_io = io.StringIO(raw_bytes.decode("utf-8"))
    reader = csv.DictReader(text_io)
    lines: list[str] = []

    for row in reader:
        parts = [f"{key}: {value}" for key, value in row.items() if value]
        lines.append(", ".join(parts))

    return "\n".join(lines)
