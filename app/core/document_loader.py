"""
Document Loader: Laedt Dokumente aus dem data/processed/ Ordner.

Was macht ein Document Loader?
------------------------------
Er liest Rohdateien (JSON, TXT) von der Festplatte und bringt sie in ein
einheitliches Format: {"content": str, "metadata": dict}

Warum ein einheitliches Format?
Weil die nachfolgenden Schritte (Chunking, Embedding, ChromaDB) sich nicht
darum kuemmern sollen, ob die Daten aus JSON oder TXT kamen.
Jeder Schritt in der Pipeline hat eine klare Aufgabe.

Ordnerstruktur der Daten:
    data/processed/
    ├── datenblaetter/    -> JSON-Dateien mit technischen Daten
    ├── rueckrufe/        -> TXT-Dateien mit KBA-Rueckrufaktionen
    └── schwachstellen/   -> TXT-Dateien mit Schwachstellenanalysen
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping: Ordnername -> doc_type fuer ChromaDB-Metadaten
# So wissen wir spaeter bei der Suche, was fuer ein Dokument es ist
ORDNER_ZU_DOCTYPE: dict[str, str] = {
    "datenblaetter": "datenblatt",
    "rueckrufe": "rueckruf",
    "schwachstellen": "schwachstelle",
}

# Mapping: Modellname (aus Dateiname) -> Hersteller
# Wird benutzt um den Hersteller automatisch aus dem Dateinamen abzuleiten
MODELL_ZU_HERSTELLER: dict[str, str] = {
    "tesla_model_3": "Tesla",
    "vw_id3": "Volkswagen",
    "hyundai_ioniq_5": "Hyundai",
}

# Mapping: Dateiname-Prefix -> schoener Modellname
MODELL_NAMEN: dict[str, str] = {
    "tesla_model_3": "Tesla Model 3",
    "vw_id3": "VW ID.3",
    "hyundai_ioniq_5": "Hyundai Ioniq 5",
}


def _extract_modell_key(filename: str) -> str:
    """Extrahiert den Modell-Key aus dem Dateinamen.

    Beispiel: 'tesla_model_3_rueckrufe.txt' -> 'tesla_model_3'
              'vw_id3.json' -> 'vw_id3'

    Strategie: Wir entfernen bekannte Suffixe (_rueckrufe, _schwachstellen)
    und die Dateiendung, dann bleibt der Modell-Key uebrig.
    """
    # Dateiendung und bekannte Suffixe entfernen
    name = filename.replace(".json", "").replace(".txt", "")
    for suffix in ("_rueckrufe", "_schwachstellen"):
        name = name.replace(suffix, "")
    return name


def _build_metadata(filepath: Path) -> dict[str, str]:
    """Baut die Metadaten aus Dateipfad und Dateiname zusammen.

    Diese Metadaten werden spaeter in ChromaDB gespeichert und ermoeglichen
    gefilterte Suchen (z.B. "zeige nur Rueckrufe fuer Tesla Model 3").
    """
    # Ordnername = uebergeordneter Ordner (z.B. "rueckrufe")
    ordner = filepath.parent.name
    doc_type = ORDNER_ZU_DOCTYPE.get(ordner, "unbekannt")

    # Modell aus Dateiname extrahieren
    modell_key = _extract_modell_key(filepath.name)
    modell = MODELL_NAMEN.get(modell_key, modell_key)
    hersteller = MODELL_ZU_HERSTELLER.get(modell_key, "Unbekannt")

    # source: Bei Rueckrufen ist die Quelle das KBA,
    # bei Schwachstellen verschiedene Quellen (wir nutzen den Ordnernamen)
    source_mapping: dict[str, str] = {
        "datenblaetter": "datenblatt",
        "rueckrufe": "kba",
        "schwachstellen": "carwiki",
    }
    source = source_mapping.get(ordner, ordner)

    return {
        "source": source,
        "doc_type": doc_type,
        "modell": modell,
        "hersteller": hersteller,
    }


def load_text_file(filepath: Path) -> dict:
    """Laedt eine Textdatei und gibt Inhalt + Metadaten zurueck.

    Args:
        filepath: Pfad zur TXT-Datei

    Returns:
        Dict mit "content" (der Text) und "metadata" (Infos ueber das Dokument)

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert
        UnicodeDecodeError: Wenn die Datei kein gueltiges UTF-8 ist
    """
    logger.info("Lade Textdatei: %s", filepath)

    content = filepath.read_text(encoding="utf-8")
    metadata = _build_metadata(filepath)

    logger.debug(
        "Textdatei geladen: %d Zeichen, Metadaten: %s",
        len(content),
        metadata,
    )

    return {"content": content, "metadata": metadata}


def load_json_file(filepath: Path) -> dict:
    """Laedt eine JSON-Datei und gibt Inhalt + Metadaten zurueck.

    Bei JSON-Dateien (Datenblaetter) lesen wir die Metadaten direkt
    aus dem JSON-Inhalt, weil dort "modell" und "hersteller" als
    Felder vorhanden sind.

    Der Inhalt wird in einen lesbaren Text umgewandelt, damit er
    spaeter als Embedding verarbeitet werden kann.
    (ChromaDB speichert Text, kein JSON.)

    Args:
        filepath: Pfad zur JSON-Datei

    Returns:
        Dict mit "content" (Text-Darstellung) und "metadata"
    """
    logger.info("Lade JSON-Datei: %s", filepath)

    raw = filepath.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Metadaten aus Dateipfad UND JSON-Inhalt
    metadata = _build_metadata(filepath)

    # JSON-Felder ueberschreiben die aus dem Dateinamen abgeleiteten Werte,
    # weil sie zuverlaessiger sind
    if "modell" in data:
        metadata["modell"] = data["modell"]
    if "hersteller" in data:
        metadata["hersteller"] = data["hersteller"]

    # JSON in lesbaren Text umwandeln fuer Embeddings
    content = _json_to_text(data)

    logger.debug(
        "JSON-Datei geladen: %d Zeichen Text, Metadaten: %s",
        len(content),
        metadata,
    )

    return {"content": content, "metadata": metadata}


def _json_to_text(data: dict) -> str:
    """Wandelt ein Datenblatt-JSON in lesbaren Fliesstext um.

    Warum nicht einfach json.dumps()? Weil das LLM und die Embeddings
    mit natuerlichem Text besser arbeiten als mit JSON-Syntax.
    '{"modell": "Tesla Model 3"}' -> 'Modell: Tesla Model 3'
    """
    lines: list[str] = []

    # Grunddaten
    lines.append(f"Datenblatt: {data.get('modell', 'Unbekannt')}")
    lines.append(f"Hersteller: {data.get('hersteller', 'Unbekannt')}")
    lines.append(f"Bauzeit: {data.get('bauzeit', 'Unbekannt')}")
    lines.append(f"Fahrzeugklasse: {data.get('fahrzeugklasse', 'Unbekannt')}")

    # Technische Daten
    tech = data.get("technische_daten", {})
    if tech:
        lines.append("")
        lines.append("Technische Daten:")
        # Batterie-Optionen
        for batterie in tech.get("batterie_optionen", []):
            lines.append(
                f"  - {batterie['bezeichnung']}: "
                f"{batterie['kapazitaet_kwh']} kWh, "
                f"{batterie['reichweite_wltp_km']} km WLTP, "
                f"Zellchemie: {batterie['zellchemie']}"
            )
        # Restliche technische Daten
        for key in ("ladeanschluss", "max_dc_ladeleistung_kw",
                     "max_ac_ladeleistung_kw", "antrieb",
                     "beschleunigung_0_100", "hoechstgeschwindigkeit_kmh"):
            if key in tech:
                label = key.replace("_", " ").title()
                lines.append(f"  {label}: {tech[key]}")

    # Bekannte Probleme
    for problem in data.get("bekannte_probleme", []):
        lines.append("")
        lines.append(
            f"Bekanntes Problem: {problem['problem']} "
            f"(Schwere: {problem['schwere']}, "
            f"Haeufigkeit: {problem['haeufigkeit']})"
        )
        lines.append(f"  {problem['beschreibung']}")

    # Staerken
    staerken = data.get("staerken", [])
    if staerken:
        lines.append("")
        lines.append("Staerken:")
        for s in staerken:
            lines.append(f"  - {s}")

    # Empfehlung
    empfehlung = data.get("empfehlung_gebrauchtkauf", "")
    if empfehlung:
        lines.append("")
        lines.append(f"Empfehlung Gebrauchtkauf: {empfehlung}")

    return "\n".join(lines)


def load_all_documents(data_dir: str = "data/processed") -> list[dict]:
    """Laedt alle Dokumente aus dem Datenverzeichnis.

    Geht durch alle Unterordner (datenblaetter, rueckrufe, schwachstellen)
    und laedt jede Datei mit dem passenden Loader.

    Args:
        data_dir: Pfad zum Datenverzeichnis (relativ zum Projektroot)

    Returns:
        Liste von Dicts, jedes mit "content" und "metadata"
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error("Datenverzeichnis nicht gefunden: %s", data_path)
        return []

    documents: list[dict] = []

    # Alle Dateien in Unterordnern durchgehen
    for filepath in sorted(data_path.rglob("*")):
        # Nur Dateien, keine Ordner
        if not filepath.is_file():
            continue

        try:
            if filepath.suffix == ".json":
                doc = load_json_file(filepath)
            elif filepath.suffix == ".txt":
                doc = load_text_file(filepath)
            else:
                logger.warning("Unbekanntes Dateiformat: %s (uebersprungen)", filepath)
                continue

            documents.append(doc)
            logger.info(
                "Dokument geladen: %s (%s, %s)",
                filepath.name,
                doc["metadata"]["doc_type"],
                doc["metadata"]["modell"],
            )

        except json.JSONDecodeError as e:
            # JSON-Datei ist kaputt (z.B. ungueltige Syntax)
            logger.error("Fehler beim Parsen von %s: %s", filepath, e)
        except UnicodeDecodeError as e:
            # Datei ist kein gueltiges UTF-8 (z.B. Binaerdatei)
            logger.error("Encoding-Fehler in %s: %s", filepath, e)
        except Exception as e:
            # Unerwarteter Fehler: loggen und mit naechster Datei weitermachen
            logger.error("Unerwarteter Fehler bei %s: %s", filepath, e)

    logger.info("Insgesamt %d Dokumente geladen", len(documents))
    return documents
