"""
Metadata Filter: Extrahiert Fahrzeug-Infos aus natuerlicher Sprache.

Warum brauchen wir das?
-----------------------
Wenn ein User fragt "Rueckrufe Tesla Model 3 2020", wollen wir in ChromaDB
nicht einfach nach dem gesamten Satz suchen, sondern GEZIELT filtern:
  where={"modell": "Tesla Model 3"}

Dafuer muessen wir aus der Frage das Modell, den Hersteller und evtl.
das Baujahr extrahieren. Das ist ueberraschend schwierig, weil User
auf viele Arten fragen koennen:
  - "Probleme beim Ioniq 5"          -> Hyundai Ioniq 5
  - "Tesla Model 3 Schwachstellen"   -> Tesla Model 3
  - "Was taugt der ID.3?"            -> VW ID.3
  - "Lohnt sich ein gebrauchter?"    -> kein Modell

Zwei Ansaetze, warum beide?
----------------------------
1. KEYWORD-MATCHER (schnell, aber dumm):
   + Laeuft sofort, kein LLM-Aufruf noetig (~0ms)
   + Funktioniert zuverlaessig fuer bekannte Modelle
   - Erkennt nur exakte oder vordefinierte Schreibweisen
   - "Teslar Model 3" oder "M3" wuerde nicht erkannt

2. LLM-EXTRAKTION (schlau, aber langsam):
   + Versteht auch Tippfehler, Abkuerzungen, Umgangssprache
   + Kann aus Kontext ableiten ("der Koreaner mit 800V" -> Ioniq 5)
   - Dauert 2-5 Sekunden (Mistral auf CPU ohne GPU)
   - Kann halluzinieren oder falsches JSON zurueckgeben

Strategie: LLM ist der primaere Ansatz (bessere Erkennung).
Falls Ollama nicht laeuft oder Muell zurueckgibt, fangen wir das ab
und fallen auf den Keyword-Matcher zurueck.
Auf dem Lenovo M920q (kein GPU) dauert der LLM-Call ca. 2-5s.
Das ist akzeptabel, weil die Extraktion nur 1x pro Anfrage passiert.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# === Wissensbasis: Bekannte Modelle und ihre Varianten ===
#
# Fuer jeden bekannten Modellnamen listen wir alternative Schreibweisen auf.
# Der Keyword-Matcher durchsucht die Query nach diesen Varianten.
# Reihenfolge ist wichtig: Laengere Matches zuerst pruefen,
# damit "Tesla Model 3" vor "Tesla" gefunden wird.
BEKANNTE_MODELLE: list[dict[str, str | list[str]]] = [
    {
        "modell": "Tesla Model 3",
        "hersteller": "Tesla",
        # Alle Varianten in Kleinbuchstaben fuer case-insensitive Suche
        "varianten": [
            "tesla model 3", "model 3", "tesla m3", "teslamodel3",
        ],
    },
    {
        "modell": "VW ID.3",
        "hersteller": "Volkswagen",
        "varianten": [
            "vw id.3", "vw id3", "id.3", "id3", "volkswagen id.3",
            "volkswagen id3",
        ],
    },
    {
        "modell": "Hyundai Ioniq 5",
        "hersteller": "Hyundai",
        "varianten": [
            "hyundai ioniq 5", "ioniq 5", "ioniq5", "hyundai ioniq5",
        ],
    },
]


def extract_vehicle_info(query: str) -> dict[str, str | None]:
    """Extrahiert Fahrzeug-Infos aus einer User-Frage.

    Versucht zuerst die LLM-Extraktion, dann Keyword-Matching als Fallback.

    Args:
        query: Die Frage des Users, z.B. "Rueckrufe Tesla Model 3 2020"

    Returns:
        Dict mit Schluesseln: "modell", "hersteller", "baujahr"
        Werte sind None wenn nichts erkannt wurde.
        Beispiel: {"modell": "Tesla Model 3", "hersteller": "Tesla", "baujahr": "2020"}
    """
    # Zuerst den LLM-Ansatz versuchen (besser bei Tippfehlern, Umgangssprache)
    result = _extract_via_llm(query)

    if result and result.get("modell"):
        logger.info("LLM-Extraktion erfolgreich: %s", result)
        return result

    # Fallback: Keyword-Matching (schnell, aber nur fuer bekannte Schreibweisen)
    logger.info("LLM-Extraktion fehlgeschlagen, nutze Keyword-Matcher")
    result = _extract_via_keywords(query)
    logger.info("Keyword-Extraktion: %s", result)
    return result


def _extract_via_llm(query: str) -> dict[str, str | None] | None:
    """Extrahiert Fahrzeug-Infos ueber das LLM (Mistral via Ollama).

    Schickt einen speziell formulierten Prompt an Mistral, der das Modell
    dazu bringt, strukturiertes JSON zurueckzugeben.

    Returns:
        Dict mit extrahierten Infos, oder None bei Fehler
    """
    # Der Prompt ist sehr spezifisch formuliert, damit das LLM
    # nur JSON zurueckgibt und nicht anfaengt zu plaudern
    prompt = (
        "Extrahiere aus der folgenden Frage das Elektroauto-Modell, "
        "den Hersteller und falls genannt das Baujahr.\n"
        "Antworte NUR mit einem JSON-Objekt, KEIN anderer Text.\n"
        "Wenn kein Modell erkannt wird, setze null.\n"
        "Format: {\"modell\": \"...\", \"hersteller\": \"...\", \"baujahr\": \"...\"}\n"
        "\n"
        f"Frage: {query}\n"
        "JSON:"
    )

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    # Niedrige Temperatur = weniger Kreativitaet, mehr Praezision
                    # Fuer Datenextraktion wollen wir keine "kreativen" Antworten
                    "options": {"temperature": 0.1},
                },
            )

        if response.status_code != 200:
            logger.warning("Ollama-Fehler (HTTP %d)", response.status_code)
            return None

        raw_text = response.json().get("response", "")
        logger.debug("LLM-Rohantwort: %s", raw_text)

        # JSON aus der Antwort extrahieren
        return _parse_llm_response(raw_text)

    except httpx.ConnectError:
        logger.warning("Ollama nicht erreichbar, ueberspringe LLM-Extraktion")
        return None
    except httpx.TimeoutException:
        logger.warning("LLM-Extraktion Timeout (>15s)")
        return None


def _parse_llm_response(raw_text: str) -> dict[str, str | None] | None:
    """Parst die LLM-Antwort und extrahiert das JSON.

    LLMs sind unberechenbar. Manchmal kommt sauberes JSON,
    manchmal kommt Text drumherum. Wir versuchen verschiedene
    Strategien um das JSON zu finden.

    Returns:
        Geparstes Dict oder None wenn nichts zu retten ist
    """
    # Strategie 1: Direktes JSON-Parsing (Idealfall)
    try:
        data = json.loads(raw_text.strip())
        return _normalize_result(data)
    except json.JSONDecodeError:
        pass

    # Strategie 2: JSON aus dem Text extrahieren
    # Manchmal antwortet das LLM: "Hier ist das JSON: {...}"
    json_match = re.search(r'\{[^{}]*\}', raw_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _normalize_result(data)
        except json.JSONDecodeError:
            pass

    logger.warning("Konnte kein JSON aus LLM-Antwort parsen: %s", raw_text[:100])
    return None


def _normalize_result(data: dict) -> dict[str, str | None]:
    """Normalisiert die extrahierten Daten.

    Stellt sicher, dass das Ergebnis immer die gleichen Keys hat
    und dass leere Strings zu None werden.
    """
    def clean(value: str | None) -> str | None:
        """Leere Strings und 'null'-Strings zu None umwandeln."""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ("", "null", "none", "unbekannt", "n/a"):
                return None
        return value

    return {
        "modell": clean(data.get("modell")),
        "hersteller": clean(data.get("hersteller")),
        "baujahr": clean(str(data["baujahr"]) if data.get("baujahr") else None),
    }


def _extract_via_keywords(query: str) -> dict[str, str | None]:
    """Extrahiert Fahrzeug-Infos per einfachem String-Matching.

    Durchsucht die Query nach bekannten Modellnamen und ihren Varianten.
    Schnell (~0ms), aber erkennt nur vordefinierte Schreibweisen.

    Trick: Wir sortieren die Varianten nach Laenge (laengste zuerst),
    damit "Tesla Model 3" vor "Tesla" gefunden wird.
    """
    query_lower = query.lower()

    # Alle Varianten sammeln und nach Laenge sortieren
    # Warum laengste zuerst? Damit "VW ID.3" vor "ID3" matcht
    for modell_info in BEKANNTE_MODELLE:
        varianten = modell_info["varianten"]
        # Laengste Variante zuerst pruefen (spezifischer Match)
        for variante in sorted(varianten, key=len, reverse=True):
            if variante in query_lower:
                # Modell gefunden! Jetzt noch nach Baujahr suchen
                baujahr = _extract_baujahr(query)
                return {
                    "modell": modell_info["modell"],
                    "hersteller": modell_info["hersteller"],
                    "baujahr": baujahr,
                }

    # Kein Modell erkannt
    return {
        "modell": None,
        "hersteller": None,
        "baujahr": _extract_baujahr(query),
    }


def _extract_baujahr(query: str) -> str | None:
    """Extrahiert eine Jahreszahl (2010-2026) aus der Query.

    Sucht nach 4-stelligen Zahlen im Bereich 2010-2026.
    Die meisten E-Autos fuer den Massenmarkt gibt es erst ab ca. 2010.
    """
    # Alle 4-stelligen Zahlen finden
    years = re.findall(r'\b(20[12]\d)\b', query)

    if years:
        # Die erste gefundene Jahreszahl nehmen
        return years[0]

    return None


def build_where_filter(vehicle_info: dict[str, str | None]) -> dict | None:
    """Baut einen ChromaDB-where-Filter aus den extrahierten Fahrzeug-Infos.

    Nur Felder mit Wert werden in den Filter aufgenommen.
    Wenn nichts erkannt wurde, geben wir None zurueck (= keine Filterung).

    Beispiel:
        {"modell": "Tesla Model 3"} -> {"modell": "Tesla Model 3"}
        {"modell": None}            -> None (kein Filter)
    """
    where: dict[str, str] = {}

    if vehicle_info.get("modell"):
        where["modell"] = vehicle_info["modell"]

    if not where:
        return None

    return where
