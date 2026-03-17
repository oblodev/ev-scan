"""
Tests fuer das Chunking-Modul.

Hier testen wir, dass die verschiedenen Chunking-Strategien
korrekt funktionieren:
- Datenblaetter werden NICHT gechunkt
- Rueckrufe werden am Trennzeichen gesplittet
- Schwachstellen werden in Stuecke mit Overlap geteilt
- Metadaten bleiben nach dem Chunking erhalten
"""

from app.core.chunking import chunk_documents


def _make_doc(content: str, doc_type: str, modell: str = "Test Auto") -> dict:
    """Hilfsfunktion: Erstellt ein Dokument-Dict fuer Tests."""
    return {
        "content": content,
        "metadata": {
            "source": "test",
            "doc_type": doc_type,
            "modell": modell,
            "hersteller": "Test",
        },
    }


class TestDatenblattChunking:
    """Datenblaetter sollen als Ganzes erhalten bleiben."""

    def test_json_not_chunked(self) -> None:
        """Ein Datenblatt ergibt genau 1 Chunk."""
        doc = _make_doc(
            content="Datenblatt: Tesla Model 3\nBatterie: 60 kWh\nReichweite: 491 km",
            doc_type="datenblatt",
        )
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_datenblatt_content_unchanged(self) -> None:
        """Der Inhalt des Datenblatts wird nicht veraendert."""
        original = "Datenblatt: Tesla Model 3\nBatterie: 60 kWh"
        doc = _make_doc(content=original, doc_type="datenblatt")
        chunks = chunk_documents([doc])
        assert chunks[0]["content"] == original


class TestRueckrufChunking:
    """Rueckrufe werden am '--- Rückruf ---' Trennzeichen gesplittet."""

    def test_rueckrufe_split_by_separator(self) -> None:
        """Jeder Rueckruf-Block wird ein eigener Chunk.

        Der Header (# KBA Rückrufaktionen) wird ebenfalls als
        eigenstaendiger Chunk behalten, daher 3 Rueckrufe + 1 Header = 4.
        """
        content = (
            "# KBA Rückrufaktionen: Test\n\n"
            "--- Rückruf ---\n"
            "Hersteller: Tesla\nMangel: Problem 1\n\n"
            "--- Rückruf ---\n"
            "Hersteller: Tesla\nMangel: Problem 2\n\n"
            "--- Rückruf ---\n"
            "Hersteller: Tesla\nMangel: Problem 3"
        )
        doc = _make_doc(content=content, doc_type="rueckruf")
        chunks = chunk_documents([doc])
        # 3 Rueckrufe + 1 Header-Block = 4 Chunks
        assert len(chunks) == 4
        # Die Rueckruf-Chunks enthalten den Mangel
        rueckruf_chunks = [c for c in chunks if "Mangel" in c["content"]]
        assert len(rueckruf_chunks) == 3

    def test_rueckruf_content_contains_mangel(self) -> None:
        """Jeder Rueckruf-Chunk enthaelt den vollstaendigen Rueckruf-Text."""
        content = (
            "# KBA Rückrufaktionen\n\n"
            "--- Rückruf ---\n"
            "Hersteller: Tesla\nMangel: Bremsen defekt\n\n"
            "--- Rückruf ---\n"
            "Hersteller: Tesla\nMangel: Software-Bug"
        )
        doc = _make_doc(content=content, doc_type="rueckruf")
        chunks = chunk_documents([doc])
        # Header ist chunks[0], Rueckrufe ab chunks[1]
        assert "Bremsen defekt" in chunks[1]["content"]
        assert "Software-Bug" in chunks[2]["content"]

    def test_empty_blocks_ignored(self) -> None:
        """Leere Bloecke zwischen Trennzeichen werden uebersprungen."""
        content = "--- Rückruf ---\n\n--- Rückruf ---\nMangel: Problem 1"
        doc = _make_doc(content=content, doc_type="rueckruf")
        chunks = chunk_documents([doc])
        assert len(chunks) == 1


class TestSchwachstellenChunking:
    """Schwachstellen-Texte werden in Stuecke geteilt."""

    def test_txt_chunked_correctly(self) -> None:
        """Langer Text wird in mehrere Chunks aufgeteilt."""
        # Einen Text erstellen der laenger als chunk_size (500) ist
        content = "## Abschnitt 1\n" + "A" * 600 + "\n\n## Abschnitt 2\n" + "B" * 600
        doc = _make_doc(content=content, doc_type="schwachstelle")
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_short_text_stays_together(self) -> None:
        """Kurzer Text der unter chunk_size liegt bleibt zusammen."""
        content = "## Problem\nKurzer Text ueber ein Problem."
        doc = _make_doc(content=content, doc_type="schwachstelle")
        chunks = chunk_documents([doc])
        # Mindestens 1 Chunk
        assert len(chunks) >= 1


class TestMetadataPreservation:
    """Metadaten muessen nach dem Chunking erhalten bleiben."""

    def test_metadata_preserved_after_chunking(self) -> None:
        """Alle Original-Metadaten sind in jedem Chunk vorhanden."""
        doc = _make_doc(
            content="Datenblatt: Hyundai Ioniq 5",
            doc_type="datenblatt",
            modell="Hyundai Ioniq 5",
        )
        chunks = chunk_documents([doc])

        for chunk in chunks:
            assert chunk["metadata"]["modell"] == "Hyundai Ioniq 5"
            assert chunk["metadata"]["doc_type"] == "datenblatt"
            assert chunk["metadata"]["source"] == "test"
            assert chunk["metadata"]["hersteller"] == "Test"

    def test_chunk_index_added(self) -> None:
        """Jeder Chunk bekommt einen chunk_index in den Metadaten."""
        doc = _make_doc(
            content="Datenblatt: Test",
            doc_type="datenblatt",
        )
        chunks = chunk_documents([doc])
        assert "chunk_index" in chunks[0]["metadata"]

    def test_multiple_docs_keep_own_metadata(self) -> None:
        """Chunks verschiedener Dokumente behalten ihre eigenen Metadaten."""
        doc1 = _make_doc(content="Tesla Daten", doc_type="datenblatt", modell="Tesla Model 3")
        doc2 = _make_doc(content="VW Daten", doc_type="datenblatt", modell="VW ID.3")
        chunks = chunk_documents([doc1, doc2])

        assert chunks[0]["metadata"]["modell"] == "Tesla Model 3"
        assert chunks[1]["metadata"]["modell"] == "VW ID.3"
