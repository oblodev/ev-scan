# Text-Splitter: Teilt einen Text automatisch nach Fahrzeugmodellen auf.
#
# Problem: Ein Artikel wie "Tesla Probleme" enthaelt Abschnitte fuer
# Model 3, Model S, Model X und Model Y durcheinander. Wenn wir den
# ganzen Text unter "Tesla Model 3" speichern, bekommt das Model 3
# auch Model S-Probleme zugeordnet.
#
# Loesung: Den Text an Ueberschriften splitten und fuer jeden Abschnitt
# erkennen, welche Modelle erwaehnt werden. Dann jeden Abschnitt nur
# den richtigen Modellen zuordnen.

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Tesla-Modelle und ihre Varianten (case-insensitive)
TESLA_MODELLE: dict[str, list[str]] = {
    "Tesla Model 3": ["model 3", "model3"],
    "Tesla Model Y": ["model y", "modely"],
    "Tesla Model S": ["model s", "models"],
    "Tesla Model X": ["model x", "modelx"],
}

# Alle bekannten Modelle (erweiterbar)
ALLE_MODELLE: dict[str, list[str]] = {
    **TESLA_MODELLE,
    "VW ID.3": ["id.3", "id3", "vw id"],
    "VW ID.4": ["id.4", "id4"],
    "Hyundai Ioniq 5": ["ioniq 5", "ioniq5"],
}


def split_text_by_models(
    text: str,
    fallback_modell: str = "",
) -> list[dict[str, str]]:
    """Teilt einen Text in Abschnitte und ordnet jedem die erkannten Modelle zu.

    Strategie:
    1. Text an Ueberschriften splitten (Zeilen die wie Abschnittstart aussehen)
    2. Fuer jeden Abschnitt: Welche Modelle werden erwaehnt?
    3. Abschnitte die keinem Modell zugeordnet werden koennen,
       bekommen alle erkannten Modelle (Einleitung, allgemeine Tipps)

    Args:
        text: Der Gesamttext
        fallback_modell: Modell das verwendet wird wenn nichts erkannt wird

    Returns:
        Liste von Dicts: [{"modell": "Tesla Model 3", "text": "..."}]
        Ein Abschnitt kann mehrmals vorkommen (fuer verschiedene Modelle)
    """
    sections = _split_into_sections(text)

    if not sections:
        return [{"modell": fallback_modell, "text": text}] if text.strip() else []

    # Fuer jeden Abschnitt die Modelle erkennen
    section_models: list[tuple[str, list[str]]] = []
    all_found_models: set[str] = set()

    for section_text in sections:
        models = _detect_models_in_text(section_text)
        section_models.append((section_text, models))
        all_found_models.update(models)

    # Wenn insgesamt nur 1 Modell gefunden wurde oder gar keins,
    # gibt es nichts aufzuteilen
    if len(all_found_models) <= 1:
        modell = list(all_found_models)[0] if all_found_models else fallback_modell
        return [{"modell": modell, "text": text}]

    # Abschnitte den Modellen zuordnen
    result: list[dict[str, str]] = []
    # Abschnitte ohne erkannte Modelle sammeln (Einleitung, allgemeine Tipps)
    # Diese werden spaeter allen Modellen zugeordnet
    general_sections: list[str] = []

    for section_text, models in section_models:
        if not models:
            general_sections.append(section_text)
        else:
            for modell in models:
                result.append({"modell": modell, "text": section_text})

    # Allgemeine Abschnitte (z.B. "Rostende Bremsscheiben: alle Tesla")
    # jedem gefundenen Modell zuordnen
    if general_sections:
        general_text = "\n\n".join(general_sections)
        for modell in sorted(all_found_models):
            result.append({"modell": modell, "text": general_text})

    logger.info(
        "Text aufgeteilt: %d Abschnitte -> %d Zuordnungen fuer %d Modelle",
        len(sections),
        len(result),
        len(all_found_models),
    )

    return result


def _split_into_sections(text: str) -> list[str]:
    """Teilt den Text an Ueberschrift-artigen Zeilen.

    Erkennt Abschnittsgrenzen an:
    - Zeilen die mit einem bekannten Problem-Muster beginnen
    - Zeilen die kurz sind und vor einem "Betroffen:"-Block stehen
    """
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        # Neue Sektion erkennen: Kurze Zeile gefolgt von "Betroffen:" Muster
        # Oder: Zeile die wie eine Ueberschrift aussieht
        if _is_section_header(line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections


def _is_section_header(line: str) -> bool:
    """Erkennt ob eine Zeile eine Abschnittsueberschrift ist.

    Typische Muster:
    - "Defekte Panasonic NCA 2170L Akkus beim Model 3"
    - "Lack blättert ab (häufig an Seitenschwellern)"
    - "## Karosserie und Verarbeitung"
    """
    line = line.strip()

    # Zu kurz oder zu lang fuer eine Ueberschrift
    if len(line) < 10 or len(line) > 150:
        return False

    # Markdown-Ueberschrift
    if line.startswith("## ") or line.startswith("### "):
        return True

    # Enthaelt kein Satzzeichen am Ende (Ueberschriften enden selten mit . oder ,)
    if line.endswith(".") or line.endswith(","):
        return False

    # Enthaelt ein Modell-Keyword -> wahrscheinlich Ueberschrift
    line_lower = line.lower()
    model_keywords = ["model 3", "model s", "model x", "model y",
                       "id.3", "id3", "ioniq"]
    for keyword in model_keywords:
        if keyword in line_lower:
            return True

    # Enthaelt typische Problem-Woerter -> wahrscheinlich Ueberschrift
    problem_keywords = ["defekt", "verschleiß", "ausfall", "problem",
                         "rost", "wasser", "lack", "lenkrad", "batterie",
                         "akku", "bremse", "türgriff", "ladeleistung"]
    for keyword in problem_keywords:
        if keyword in line_lower:
            return True

    return False


def _detect_models_in_text(text: str) -> list[str]:
    """Erkennt welche Fahrzeugmodelle in einem Textabschnitt erwaehnt werden.

    Sucht nach "Betroffen:"-Zeilen und nach Modellnamen im Text.
    Priorisiert die "Betroffen:"-Zeile weil sie explizit sagt
    welche Modelle betroffen sind.
    """
    text_lower = text.lower()
    found: set[str] = set()

    # Zuerst in "Betroffen:"-Zeile suchen (am zuverlaessigsten)
    betroffen_match = re.search(
        r"betroffen:\s*\n?(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    search_text = betroffen_match.group(1).lower() if betroffen_match else text_lower

    # "alle Tesla" oder "alle Modelle" -> alle Tesla-Modelle
    if re.search(r"alle\s+tesla|alle\s+modelle|alle\s+fahrzeuge", search_text):
        found.update(TESLA_MODELLE.keys())
        return sorted(found)

    # Spezifische Modelle suchen
    for modell_name, varianten in ALLE_MODELLE.items():
        for variante in varianten:
            if variante in search_text:
                found.add(modell_name)
                break

    return sorted(found)
