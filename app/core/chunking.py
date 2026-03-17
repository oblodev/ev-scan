"""
Chunking: Teilt Dokumente in kleinere Stuecke fuer die Vektorsuche.

Warum Chunking?
---------------
Ein LLM hat ein begrenztes Kontextfenster (z.B. 4096 Tokens bei Mistral 7B).
Wir koennen nicht das gesamte Wissen auf einmal reingeben. Stattdessen:

1. Wir teilen alle Dokumente in kleine Stuecke ("Chunks")
2. Jeder Chunk wird als Vektor in ChromaDB gespeichert
3. Bei einer Frage suchen wir die relevantesten Chunks
4. Nur diese Chunks geben wir dem LLM als Kontext

Was passiert wenn Chunks zu GROSS sind?
- Die Vektorsuche wird ungenau, weil ein grosser Chunk viele Themen mischen kann
- Wir verschwenden Platz im LLM-Kontextfenster mit irrelevanten Infos
- Weniger Chunks passen in den Kontext

Was passiert wenn Chunks zu KLEIN sind?
- Wichtiger Kontext geht verloren (z.B. ein Satz ohne seinen Absatz)
- Die Embeddings sind weniger aussagekraeftig
- Wir brauchen mehr Chunks = mehr Speicher und langsamere Suche

Die Goldene Mitte liegt bei ca. 300-800 Zeichen, je nach Dokumenttyp.

Warum UNTERSCHIEDLICHE Strategien pro Dokumenttyp?
-------------------------------------------------
Verschiedene Dokumente haben verschiedene Strukturen:

- Datenblaetter (JSON): Kompakte, strukturierte Daten. Gehoeren zusammen,
  weil z.B. Batterie-Info ohne Modellname nutzlos ist. -> NICHT chunken.

- Rueckrufe (TXT): Klar getrennte Bloecke ("--- Rueckruf ---").
  Jeder Rueckruf ist eigenstaendig und soll einzeln gefunden werden.
  -> Am natuerlichen Trennzeichen splitten.

- Schwachstellen (TXT): Laengere Fliesstext-Analysen mit Abschnitten.
  -> In gleich grosse Stuecke mit Ueberlappung teilen.

Warum OVERLAP (Ueberlappung)?
-----------------------------
Wenn wir Text in Stuecke schneiden, kann es passieren, dass ein wichtiger
Satz genau an der Schnittstelle liegt und in zwei Haelften geteilt wird.
Mit Overlap wiederholen wir die letzten N Zeichen des vorherigen Chunks
am Anfang des naechsten. So geht kein Kontext an den Raendern verloren.

Beispiel mit overlap=20:
  Chunk 1: "...Batterie hält im Schnitt 8 Jahre."
  Chunk 2: "hält im Schnitt 8 Jahre. Die Garantie..."
  -> Der Satz ist in beiden Chunks vollstaendig enthalten.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Teilt Dokumente in Chunks auf, je nach Dokumenttyp.

    Jeder Chunk erhaelt die Metadaten des urspruenglichen Dokuments,
    damit wir spaeter wissen woher der Chunk stammt.

    Args:
        documents: Liste von Dicts mit "content" und "metadata"
                   (Ausgabe von load_all_documents)

    Returns:
        Liste von Chunks, jeder mit "content" und "metadata"
    """
    all_chunks: list[dict] = []

    for doc in documents:
        doc_type = doc["metadata"].get("doc_type", "")

        if doc_type == "datenblatt":
            chunks = _chunk_datenblatt(doc)
        elif doc_type == "rueckruf":
            chunks = _chunk_rueckrufe(doc)
        elif doc_type == "schwachstelle":
            chunks = _chunk_schwachstellen(doc)
        else:
            # Fallback: Wie Schwachstellen behandeln (generischer Splitter)
            logger.warning(
                "Unbekannter doc_type '%s' fuer %s, nutze Standard-Chunking",
                doc_type,
                doc["metadata"].get("modell", "?"),
            )
            chunks = _chunk_schwachstellen(doc)

        all_chunks.extend(chunks)
        logger.info(
            "%s (%s): %d Chunks erstellt",
            doc["metadata"].get("modell", "?"),
            doc_type,
            len(chunks),
        )

    logger.info("Insgesamt %d Chunks aus %d Dokumenten", len(all_chunks), len(documents))
    return all_chunks


def _chunk_datenblatt(doc: dict) -> list[dict]:
    """Datenblaetter werden NICHT gechukt.

    Warum? Ein Datenblatt ist eine kompakte, strukturierte Zusammenfassung.
    Alle Infos gehoeren zusammen: Batterie-Daten ohne Modellname sind nutzlos,
    Staerken ohne die bekannten Probleme geben ein schiefes Bild.

    Das gesamte Datenblatt wird als ein einziger Chunk gespeichert.
    """
    return [
        {
            "content": doc["content"],
            "metadata": {**doc["metadata"], "chunk_index": 0},
        }
    ]


def _chunk_rueckrufe(doc: dict) -> list[dict]:
    """Rueckrufe werden am Trennzeichen '--- Rueckruf ---' gesplittet.

    Warum? Jeder Rueckruf ist ein eigenstaendiger, abgeschlossener Eintrag
    mit allen noetigen Infos (Hersteller, Bauzeitraum, Mangel, Abhilfe).
    Wenn jemand nach einem bestimmten Rueckruf sucht, soll der komplette
    Eintrag gefunden werden, nicht ein halber.

    Wir brauchen hier keinen Overlap, weil die Bloecke inhaltlich
    voneinander unabhaengig sind.
    """
    content = doc["content"]
    chunks: list[dict] = []

    # Am Trennzeichen splitten und leere Bloecke filtern
    blocks = content.split("--- Rückruf ---")

    for i, block in enumerate(blocks):
        block = block.strip()

        # Leere Bloecke ueberspringen (z.B. der Bereich vor dem ersten Trennzeichen)
        if not block:
            continue

        # Header-Zeile (z.B. "# KBA Rückrufaktionen: Tesla Model 3") ueberspringen
        # Die steht vor dem ersten Trennzeichen und ist kein Rueckruf
        if block.startswith("#") and "Rückruf" not in block.split("\n", 1)[-1]:
            continue

        chunks.append({
            "content": block,
            "metadata": {**doc["metadata"], "chunk_index": i},
        })

    return chunks


def _chunk_schwachstellen(doc: dict) -> list[dict]:
    """Schwachstellen-Texte werden mit fester Groesse + Overlap gechunkt.

    Das ist der klassische "Recursive Character Text Splitter"-Ansatz,
    den wir hier OHNE LangChain selbst implementieren.

    Strategie:
    1. Versuche an Absaetzen (Doppel-Newline) zu trennen
    2. Wenn ein Absatz zu lang ist, trenne an Saetzen (Punkt + Leerzeichen)
    3. Wenn ein Satz zu lang ist, trenne an der chunk_size Grenze

    So bleiben Absaetze und Saetze moeglichst zusammen.
    """
    content = doc["content"]
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    chunks: list[dict] = []

    # Schritt 1: Am liebsten an Abschnitts-Ueberschriften (## ...) oder
    # Doppel-Newlines trennen, weil das natuerliche Themen-Grenzen sind
    sections = _split_by_sections(content)

    # Schritt 2: Sektionen die zu lang sind, weiter aufteilen
    raw_chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            raw_chunks.append(section)
        else:
            # Zu lang -> an Saetzen trennen und zusammenfassen
            raw_chunks.extend(_split_with_overlap(section, chunk_size, chunk_overlap))

    # Schritt 3: Chunks mit Overlap zusammenbauen
    final_chunks = _apply_overlap(raw_chunks, chunk_overlap)

    # Schritt 4: Metadaten hinzufuegen
    for i, chunk_text in enumerate(final_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        chunks.append({
            "content": chunk_text,
            "metadata": {**doc["metadata"], "chunk_index": i},
        })

    return chunks


def _split_by_sections(text: str) -> list[str]:
    """Trennt Text an Markdown-Ueberschriften (## ...).

    Jede Ueberschrift startet einen neuen Abschnitt.
    Die Ueberschrift bleibt Teil des Abschnitts (als Kontext).
    """
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        # Neue Sektion bei Markdown-Ueberschrift (## oder ###)
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)

    # Letzte Sektion nicht vergessen
    if current:
        sections.append("\n".join(current))

    return sections


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Teilt einen langen Text in Stuecke mit Ueberlappung.

    Versucht an Satzgrenzen zu trennen (". "), damit Saetze nicht
    mitten im Wort abgeschnitten werden.
    """
    # An Saetzen trennen
    sentences = text.replace(". ", ".\n").split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Passt der Satz noch in den aktuellen Chunk?
        if current_length + len(sentence) + 1 > chunk_size and current_chunk:
            # Chunk ist voll -> speichern und neuen anfangen
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += len(sentence) + 1

    # Restliche Saetze als letzten Chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Fuegt Ueberlappung zwischen aufeinanderfolgenden Chunks hinzu.

    Nimmt die letzten `overlap` Zeichen des vorherigen Chunks
    und stellt sie dem naechsten Chunk voran.
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    result: list[str] = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        # Die letzten `overlap` Zeichen des vorherigen Chunks
        overlap_text = prev_chunk[-overlap:]

        # Am naechsten Wort-Anfang starten (nicht mitten im Wort)
        space_pos = overlap_text.find(" ")
        if space_pos != -1:
            overlap_text = overlap_text[space_pos + 1:]

        result.append(overlap_text + " " + chunks[i])

    return result
