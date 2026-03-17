"""
Tests fuer den URL-Scraper.

Testet die HTML-Parsing-Logik mit lokalen HTML-Strings,
damit die Tests schnell laufen und keine Netzwerk-Abhaengigkeit haben.
"""

from bs4 import BeautifulSoup

from app.core.url_scraper import (
    _clean_text,
    _extract_title,
    _find_main_content,
    _remove_unwanted_elements,
)


def _make_soup(html: str) -> BeautifulSoup:
    """Erstellt ein BeautifulSoup-Objekt aus einem HTML-String."""
    return BeautifulSoup(html, "lxml")


class TestRemoveUnwantedElements:
    """Tests fuer das Entfernen unerwuenschter HTML-Elemente."""

    def test_removes_script_tags(self) -> None:
        """Script-Tags werden entfernt."""
        soup = _make_soup("<body><script>alert('x')</script><p>Text</p></body>")
        _remove_unwanted_elements(soup)
        assert soup.find("script") is None
        assert "Text" in soup.get_text()

    def test_removes_nav_tags(self) -> None:
        """Nav-Tags werden entfernt."""
        soup = _make_soup("<body><nav>Menu</nav><p>Content</p></body>")
        _remove_unwanted_elements(soup)
        assert "Menu" not in soup.get_text()
        assert "Content" in soup.get_text()

    def test_removes_elements_with_cookie_class(self) -> None:
        """Elemente mit 'cookie' in der Klasse werden entfernt."""
        soup = _make_soup(
            '<body><div class="cookie-banner">Cookies!</div><p>Text</p></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Cookies" not in soup.get_text()
        assert "Text" in soup.get_text()

    def test_removes_elements_with_sidebar_class(self) -> None:
        """Elemente mit 'sidebar' in der Klasse werden entfernt."""
        soup = _make_soup(
            '<body><div class="widget-area sidebar">Werbung</div><p>Text</p></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Werbung" not in soup.get_text()

    def test_removes_elements_with_ad_id(self) -> None:
        """Elemente mit 'advert' in der ID werden entfernt."""
        soup = _make_soup(
            '<body><div id="advert-top">Werbung</div><p>Text</p></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Werbung" not in soup.get_text()

    def test_never_removes_body(self) -> None:
        """Body darf nie entfernt werden, auch wenn Klassen matchen.

        Das war der Bug: Wenn <body> eine Klasse hat die zufaellig
        ein Pattern enthaelt (z.B. 'sidebar' in einer WordPress-Klasse),
        wurde der gesamte Seiteninhalt geloescht.
        """
        soup = _make_soup(
            '<body class="has-sidebar-layout"><article><p>Wichtiger Text</p></article></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Wichtiger Text" in soup.get_text()

    def test_never_removes_main(self) -> None:
        """Main-Element darf nie entfernt werden."""
        soup = _make_soup(
            '<body><main class="main-sidebar-wrapper"><p>Content</p></main></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Content" in soup.get_text()

    def test_never_removes_article(self) -> None:
        """Article-Element darf nie entfernt werden."""
        soup = _make_soup(
            '<body><article class="post-with-comments"><p>Artikel</p></article></body>'
        )
        _remove_unwanted_elements(soup)
        assert "Artikel" in soup.get_text()

    def test_survives_deeply_nested_removal(self) -> None:
        """Crash-Sicherheit: Verschachtelte Elemente mit decompose().

        Das war der Kern-Bug: Wenn ein Eltern-Element per decompose()
        entfernt wird, werden alle Kinder zu None. Die Iteration ueber
        soup.find_all() traf dann auf tote Referenzen und crashte mit
        AttributeError: 'NoneType' object has no attribute 'get'.
        """
        soup = _make_soup("""
            <body>
                <div class="sidebar">
                    <div class="sidebar-inner">
                        <div class="sidebar-widget">
                            <p>Widget Text</p>
                        </div>
                    </div>
                </div>
                <article><p>Haupttext</p></article>
            </body>
        """)
        # Darf nicht crashen
        _remove_unwanted_elements(soup)
        assert "Haupttext" in soup.get_text()
        assert "Widget Text" not in soup.get_text()

    def test_survives_many_mixed_elements(self) -> None:
        """Kein Crash bei komplexem HTML mit vielen entfernbaren Elementen."""
        soup = _make_soup("""
            <body class="wp-single post-template-default">
                <header><nav>Menu</nav></header>
                <main>
                    <article>
                        <div class="entry-content">
                            <p>Absatz 1</p>
                            <p>Absatz 2</p>
                        </div>
                        <div class="comment-section">Kommentare</div>
                    </article>
                </main>
                <aside class="sidebar"><p>Sidebar</p></aside>
                <footer>Footer</footer>
                <div class="cookie-modal">Cookie Banner</div>
                <div class="newsletter-popup">Newsletter</div>
                <script>var x = 1;</script>
                <style>.x { color: red; }</style>
            </body>
        """)
        _remove_unwanted_elements(soup)
        text = soup.get_text()
        # Content bleibt erhalten
        assert "Absatz 1" in text
        assert "Absatz 2" in text
        # Muell ist weg
        assert "Cookie Banner" not in text
        assert "Newsletter" not in text
        assert "Sidebar" not in text
        assert "var x" not in text


class TestFindMainContent:
    """Tests fuer die Haupttext-Erkennung."""

    def test_finds_article(self) -> None:
        """Findet Text in <article>."""
        soup = _make_soup(
            "<body><article><p>Artikel mit genug Text fuer die Erkennung. "
            "Mindestens hundert Zeichen brauchen wir hier.</p></article></body>"
        )
        text = _find_main_content(soup)
        assert "Artikel mit genug Text" in text

    def test_finds_main(self) -> None:
        """Findet Text in <main> wenn kein <article> vorhanden."""
        soup = _make_soup(
            "<body><main><p>Hauptinhalt der Seite mit genuegend Text. "
            "Mehr als hundert Zeichen sind noetig.</p></main></body>"
        )
        text = _find_main_content(soup)
        assert "Hauptinhalt" in text

    def test_finds_largest_div(self) -> None:
        """Fallback: Groesstes div mit den meisten p-Tags."""
        soup = _make_soup("""
            <body>
                <div class="small"><p>Klein</p></div>
                <div class="big"><p>Absatz 1</p><p>Absatz 2</p><p>Absatz 3</p></div>
            </body>
        """)
        text = _find_main_content(soup)
        assert "Absatz 1" in text


class TestExtractTitle:
    """Tests fuer die Titel-Extraktion."""

    def test_extracts_from_title_tag(self) -> None:
        """Extrahiert Titel aus <title>."""
        soup = _make_soup("<html><head><title>Mein Titel</title></head><body></body></html>")
        assert _extract_title(soup) == "Mein Titel"

    def test_falls_back_to_h1(self) -> None:
        """Fallback auf <h1> wenn kein <title>."""
        soup = _make_soup("<body><h1>Ueberschrift</h1></body>")
        assert _extract_title(soup) == "Ueberschrift"

    def test_returns_default_when_no_title(self) -> None:
        """Gibt 'Ohne Titel' zurueck wenn nichts gefunden."""
        soup = _make_soup("<body><p>Nur Text</p></body>")
        assert _extract_title(soup) == "Ohne Titel"


class TestCleanText:
    """Tests fuer die Text-Bereinigung."""

    def test_removes_short_lines(self) -> None:
        """Zeilen unter 3 Zeichen werden entfernt (Artefakte)."""
        text = _clean_text("Langer Text hier\n|\n·\nNoch ein Satz")
        assert "|" not in text
        assert "Langer Text" in text

    def test_collapses_multiple_newlines(self) -> None:
        """Mehrfache Leerzeilen werden zu einer zusammengefasst."""
        text = _clean_text("Absatz 1\n\n\n\n\nAbsatz 2")
        assert "\n\n\n" not in text
        assert "Absatz 1" in text
        assert "Absatz 2" in text
