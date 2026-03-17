# URL-Scraper: Extrahiert lesbaren Text aus Webseiten.
#
# Warum nicht automatisch speichern?
# ----------------------------------
# Webseiten enthalten oft Muell: Werbung, Cookie-Banner, Navigation,
# Footer-Links, Social-Media-Buttons. Selbst nach dem Filtern kann
# irrelevanter Text uebrig bleiben. Deshalb zeigen wir den extrahierten
# Text dem User zur Pruefung – er kann ihn bearbeiten, kuerzen oder
# verwerfen. Qualitaetskontrolle durch den Menschen.
#
# Warum User-Agent?
# -----------------
# Viele Webseiten blockieren Anfragen ohne User-Agent Header, weil
# sie vermuten dass es ein Bot ist. Mit einem beschreibenden User-Agent
# sind wir transparent und werden seltener blockiert.
#
# Wie funktioniert BeautifulSoup?
# ------------------------------
# BeautifulSoup parst HTML in einen Baum aus Elementen.
# Man kann dann gezielt Elemente suchen, filtern und den Text extrahieren.
# Beispiel: soup.find("article") findet das <article>-Element,
# .get_text() gibt den reinen Text ohne HTML-Tags zurueck.
#
# Limitation: JavaScript-Rendering
# ---------------------------------
# Manche Seiten laden Inhalte per JavaScript nach (Single-Page-Apps).
# httpx laedt nur das initiale HTML, fuehrt kein JavaScript aus.
# Fuer solche Seiten braeuchte man Playwright oder Selenium (schwerer,
# braucht einen Browser). Fuer die meisten Nachrichtenseiten und
# Fachportale (ADAC, KBA, CarWiki) reicht reines HTML-Parsing.

from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# User-Agent: Beschreibt wer wir sind. Transparent und hoeflich.
USER_AGENT = "EV-Scan/1.0 (EV Kaufberater; +https://github.com/oblodev/ev-scan)"

# Maximale Textlaenge: Verhindert dass riesige Seiten den Speicher sprengen
MAX_TEXT_LENGTH = 50_000

# HTML-Tags die wir komplett entfernen (enthalten nie nuetzlichen Content)
REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "iframe", "noscript", "svg", "form", "button",
}

# CSS-Klassen/IDs die auf Nicht-Content-Bereiche hindeuten
REMOVE_PATTERNS = re.compile(
    r"cookie|banner|advert|sidebar|menu|comment|social|share|popup|modal|newsletter|gdpr"
    r"|related-post|post-navigation|widget|breadcrumb|author-bio|donation|support-box"
    r"|cta-box|call-to-action",
    re.IGNORECASE,
)


def extract_text_from_url(url: str) -> dict[str, str | int]:
    """Laedt eine Webseite und extrahiert den Haupttext.

    Ablauf:
    1. Seite mit httpx laden
    2. HTML mit BeautifulSoup parsen
    3. Unerwuenschte Elemente entfernen (Werbung, Navigation, etc.)
    4. Haupttext finden (article, main, oder groesster Content-Block)
    5. Text bereinigen (Whitespace normalisieren)

    Args:
        url: Die URL der Webseite

    Returns:
        Dict mit "title", "extracted_text", "char_count"

    Raises:
        ValueError: Bei ungueltige URL, nicht erreichbar, kein Text gefunden
    """
    # URL validieren
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL muss mit http:// oder https:// beginnen.")

    # Seite laden
    html = _fetch_url(url)

    # HTML parsen
    soup = BeautifulSoup(html, "lxml")

    # Titel extrahieren
    title = _extract_title(soup)

    # Unerwuenschte Elemente entfernen
    _remove_unwanted_elements(soup)

    # Haupttext finden
    text = _find_main_content(soup)

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Kein verwertbarer Text auf der Seite gefunden. "
            "Moeglicherweise wird der Inhalt per JavaScript nachgeladen."
        )

    # Text bereinigen
    text = _clean_text(text)

    # Laenge begrenzen
    truncated = False
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        truncated = True
        logger.warning("Text auf %d Zeichen gekuerzt", MAX_TEXT_LENGTH)

    logger.info(
        "URL gescrapt: %s – Titel: '%s', %d Zeichen%s",
        url, title[:50], len(text),
        " (gekuerzt)" if truncated else "",
    )

    return {
        "title": title,
        "extracted_text": text,
        "char_count": len(text),
    }


def _fetch_url(url: str) -> str:
    """Laedt den HTML-Inhalt einer URL."""
    try:
        with httpx.Client(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)

        if response.status_code != 200:
            raise ValueError(
                f"Seite nicht erreichbar (HTTP {response.status_code})"
            )

        return response.text

    except httpx.ConnectError:
        raise ValueError(f"Seite nicht erreichbar: {url}")
    except httpx.TimeoutException:
        raise ValueError(f"Timeout nach 15 Sekunden: {url}")
    except httpx.InvalidURL:
        raise ValueError(f"Ungueltige URL: {url}")


def _extract_title(soup: BeautifulSoup) -> str:
    """Extrahiert den Seitentitel aus <title> oder <h1>."""
    # Zuerst <title> versuchen
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # Fallback: Erstes <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "Ohne Titel"


def _remove_unwanted_elements(soup: BeautifulSoup) -> None:
    """Entfernt unerwuenschte HTML-Elemente aus dem DOM.

    Geht den HTML-Baum durch und entfernt:
    1. Tags die nie Content enthalten (script, style, nav, etc.)
    2. Elemente deren class oder id auf Werbung/Navigation hindeuten

    WICHTIG: Immer erst sammeln, dann loeschen.
    decompose() veraendert den DOM-Baum, sodass Referenzen in einer
    Live-Iteration ungueltig werden und AttributeErrors ausloesen.
    """
    # Schritt 1: Bekannte Nicht-Content-Tags sammeln und entfernen
    tags_to_remove: list[Tag] = []
    for tag_name in REMOVE_TAGS:
        tags_to_remove.extend(soup.find_all(tag_name))

    for element in tags_to_remove:
        try:
            element.decompose()
        except AttributeError:
            pass

    # Schritt 2: Elemente mit verdaechtigen class/id sammeln
    # Wir schliessen strukturelle Top-Level-Tags aus (body, html, main, article),
    # damit nicht versehentlich der gesamte Content geloescht wird
    SKIP_TAGS = {"body", "html", "main", "article"}
    to_remove: list[Tag] = []

    for element in list(soup.find_all(True)):
        if element is None or not isinstance(element, Tag):
            continue
        if element.name in SKIP_TAGS:
            continue

        try:
            classes = element.get("class", [])
            if isinstance(classes, list):
                classes = " ".join(classes)
            elif classes is None:
                classes = ""

            element_id = element.get("id") or ""
            combined = f"{classes} {element_id}"

            if REMOVE_PATTERNS.search(combined):
                to_remove.append(element)
        except AttributeError:
            continue

    for element in to_remove:
        try:
            element.decompose()
        except AttributeError:
            pass


def _find_main_content(soup: BeautifulSoup) -> str:
    """Findet den Haupttext der Seite.

    Strategie (in Prioritaetsreihenfolge):
    1. <article> Element (semantisch korrekte Seiten)
    2. <main> Element
    3. Element mit role="main"
    4. Groesster Content-Block (div/section mit den meisten <p>-Tags)
    """
    # Strategie 1: <article>
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
        if len(text) > 100:
            return text

    # Strategie 2: <main>
    main = soup.find("main")
    if main:
        text = main.get_text(separator="\n", strip=True)
        if len(text) > 100:
            return text

    # Strategie 3: role="main"
    role_main = soup.find(attrs={"role": "main"})
    if role_main:
        text = role_main.get_text(separator="\n", strip=True)
        if len(text) > 100:
            return text

    # Strategie 4: Groesstes div/section mit den meisten <p>-Tags
    best_block = None
    best_p_count = 0

    for container in soup.find_all(["div", "section"]):
        p_tags = container.find_all("p")
        if len(p_tags) > best_p_count:
            best_p_count = len(p_tags)
            best_block = container

    if best_block and best_p_count >= 2:
        return best_block.get_text(separator="\n", strip=True)

    # Letzter Fallback: Gesamter Body-Text
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    return ""


# Zeilen die auf Sharing, Werbung oder Navigation hindeuten
_JUNK_LINE_PATTERNS = re.compile(
    r"^(teilen|share|tweet|pin it|e-mail|drucken|print"
    r"|vorheriger beitrag|nächster beitrag|previous post|next post"
    r"|beitrag teilen|passend dazu|weiterlesen|read more"
    r"|jetzt unterstützen|jetzt spenden|hier klicken"
    r"|werbung|anzeige|sponsored"
    r"|als amazon-partner|partner-link"
    r"|hast du in diesem beitrag"
    r"|wenn dir .* mehrwert bietet"
    r"|herzlichen dank"
    r"|starte hier"
    r"|bildquelle:"
    r"|\* in diesem beitrag sind partner).*$",
    re.IGNORECASE,
)

# Erkennt den Anfang des Muell-Footers am Ende des Textes
_FOOTER_START_PATTERNS = re.compile(
    r"^(beitrag teilen|passend dazu|vorheriger beitrag|nächster beitrag"
    r"|wenn dir .* mehrwert|hast du in diesem beitrag"
    r"|du möchtest einen tesla|tesla kaufen\?"
    r"|du kannst meine arbeit|unterstütze die .+-community"
    r"|hier geht es zum nächsten beitrag"
    r"|\* in diesem beitrag sind partner)",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Bereinigt den extrahierten Text.

    - Mehrfache Leerzeilen zu einer zusammenfassen
    - Fuehrende/nachfolgende Whitespaces pro Zeile entfernen
    - Sehr kurze Zeilen (< 3 Zeichen) entfernen (oft Artefakte)
    - Typische Sharing/Werbe/Footer-Zeilen entfernen
    """
    lines = text.split("\n")

    # Zeilen bereinigen
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        # Zu kurze Zeilen ("|", "·", "—") sind meist Artefakte
        if len(line) < 3:
            continue
        # Typische Muell-Zeilen entfernen
        if _JUNK_LINE_PATTERNS.match(line):
            continue
        cleaned.append(line)

    # Footer-Muell am Ende abschneiden: Ab der ersten Footer-Zeile
    # ist alles irrelevant (Related Posts, Sharing, Affiliate-Hinweise)
    cut_index = len(cleaned)
    for i, line in enumerate(cleaned):
        if _FOOTER_START_PATTERNS.match(line):
            cut_index = i
            break
    cleaned = cleaned[:cut_index]

    # Navigation am Anfang entfernen: Kurze Zeilen vor dem ersten
    # langen Absatz sind meist Menue-Links
    start_index = 0
    for i, line in enumerate(cleaned):
        if len(line) > 80:
            # Erster richtiger Absatz gefunden
            start_index = i
            break
    # Titel-Zeile (direkt vor dem ersten Absatz) behalten
    if start_index > 0:
        for j in range(start_index - 1, -1, -1):
            if len(cleaned[j]) > 20:
                start_index = j
                break
    cleaned = cleaned[start_index:]

    # Mehrfache Leerzeilen zusammenfassen
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
