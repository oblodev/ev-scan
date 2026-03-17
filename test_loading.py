"""
Testskript fuer Document Loader + Chunking.

Ausfuehren mit: python3 test_loading.py
"""

import logging

from app.core.document_loader import load_all_documents
from app.core.chunking import chunk_documents

# Logging einschalten, damit wir sehen was passiert
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """Laedt alle Dokumente und chunked sie."""
    print("=" * 60)
    print("SCHRITT 1: Dokumente laden")
    print("=" * 60)

    documents = load_all_documents("data/processed")
    print(f"\n-> {len(documents)} Dokumente geladen\n")

    for doc in documents:
        meta = doc["metadata"]
        print(
            f"  {meta['modell']:20s} | {meta['doc_type']:15s} | "
            f"{meta['source']:10s} | {len(doc['content']):5d} Zeichen"
        )

    print()
    print("=" * 60)
    print("SCHRITT 2: Chunking")
    print("=" * 60)

    chunks = chunk_documents(documents)
    print(f"\n-> {len(chunks)} Chunks aus {len(documents)} Dokumenten\n")

    # Chunks pro Typ zaehlen
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        dt = chunk["metadata"]["doc_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1

    print("Chunks pro Dokumenttyp:")
    for dt, count in sorted(type_counts.items()):
        print(f"  {dt:15s}: {count} Chunks")

    print()
    print("=" * 60)
    print("SCHRITT 3: Beispiel-Chunks anzeigen")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:3]):
        meta = chunk["metadata"]
        print(f"\n--- Chunk {i} ({meta['doc_type']}, {meta['modell']}) ---")
        # Nur die ersten 200 Zeichen anzeigen
        preview = chunk["content"][:200]
        if len(chunk["content"]) > 200:
            preview += "..."
        print(preview)

    print()
    print("=" * 60)
    print("SCHRITT 4: Metadaten-Check")
    print("=" * 60)

    # Pruefen ob alle Chunks die nötigen Metadaten haben
    required_keys = {"source", "doc_type", "modell", "hersteller", "chunk_index"}
    ok_count = 0
    for chunk in chunks:
        missing = required_keys - set(chunk["metadata"].keys())
        if missing:
            print(f"  FEHLER: Chunk fehlt: {missing}")
        else:
            ok_count += 1

    print(f"\n-> {ok_count}/{len(chunks)} Chunks haben vollstaendige Metadaten")


if __name__ == "__main__":
    main()
