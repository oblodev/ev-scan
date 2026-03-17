"""
Tests fuer den Metadata Filter (Keyword-Matcher).

Wir testen hier NUR den Keyword-Matcher (_extract_via_keywords),
nicht die LLM-Extraktion. Warum?
- LLM-Tests brauchen einen laufenden Ollama-Server
- LLM-Antworten sind nicht deterministisch (gleiche Frage, andere Antwort)
- Unit Tests muessen schnell und reproduzierbar sein

Die LLM-Extraktion wuerde man in Integrationstests testen,
die separat und seltener laufen.
"""

from app.core.metadata_filter import (
    _extract_via_keywords,
    build_where_filter,
)


class TestExtractViaKeywords:
    """Tests fuer die Keyword-basierte Fahrzeug-Erkennung."""

    def test_extracts_tesla_model_3(self) -> None:
        """Erkennt 'Tesla Model 3' in verschiedenen Formulierungen."""
        result = _extract_via_keywords("Probleme beim Tesla Model 3")
        assert result["modell"] == "Tesla Model 3"
        assert result["hersteller"] == "Tesla"

    def test_extracts_tesla_short_form(self) -> None:
        """Erkennt 'Model 3' auch ohne 'Tesla' davor."""
        result = _extract_via_keywords("Was kostet ein gebrauchter Model 3?")
        assert result["modell"] == "Tesla Model 3"

    def test_extracts_ioniq_5(self) -> None:
        """Erkennt 'Ioniq 5' in verschiedenen Schreibweisen."""
        result = _extract_via_keywords("Was sind Schwachstellen beim Ioniq 5?")
        assert result["modell"] == "Hyundai Ioniq 5"
        assert result["hersteller"] == "Hyundai"

    def test_extracts_ioniq5_without_space(self) -> None:
        """Erkennt 'Ioniq5' auch ohne Leerzeichen."""
        result = _extract_via_keywords("Ioniq5 Rueckruf")
        assert result["modell"] == "Hyundai Ioniq 5"

    def test_extracts_vw_id3(self) -> None:
        """Erkennt 'VW ID.3' und 'ID3' ohne Punkt."""
        result = _extract_via_keywords("Probleme beim VW ID3?")
        assert result["modell"] == "VW ID.3"
        assert result["hersteller"] == "Volkswagen"

    def test_extracts_id3_with_dot(self) -> None:
        """Erkennt 'ID.3' mit Punkt."""
        result = _extract_via_keywords("ID.3 kaufen?")
        assert result["modell"] == "VW ID.3"

    def test_handles_unknown_model(self) -> None:
        """Gibt None zurueck wenn kein Modell erkannt wird."""
        result = _extract_via_keywords("Welches E-Auto ist am besten?")
        assert result["modell"] is None
        assert result["hersteller"] is None

    def test_handles_unknown_specific_model(self) -> None:
        """Gibt None zurueck fuer Modelle die nicht in der DB sind."""
        result = _extract_via_keywords("Lohnt sich ein e-Golf?")
        assert result["modell"] is None

    def test_extracts_baujahr(self) -> None:
        """Extrahiert Baujahr aus der Query."""
        result = _extract_via_keywords("Rueckrufe Tesla Model 3 2020")
        assert result["modell"] == "Tesla Model 3"
        assert result["baujahr"] == "2020"

    def test_extracts_baujahr_2024(self) -> None:
        """Extrahiert auch neuere Baujahre."""
        result = _extract_via_keywords("VW ID3 2024 kaufen")
        assert result["baujahr"] == "2024"

    def test_no_baujahr_when_absent(self) -> None:
        """Gibt None zurueck wenn kein Baujahr in der Query steht."""
        result = _extract_via_keywords("Tesla Model 3 Schwachstellen")
        assert result["baujahr"] is None

    def test_case_insensitive(self) -> None:
        """Erkennung ist case-insensitive."""
        result = _extract_via_keywords("TESLA MODEL 3 probleme")
        assert result["modell"] == "Tesla Model 3"


class TestBuildWhereFilter:
    """Tests fuer den ChromaDB-Filter-Builder."""

    def test_builds_filter_with_modell(self) -> None:
        """Baut einen Filter wenn ein Modell erkannt wurde."""
        info = {"modell": "Tesla Model 3", "hersteller": "Tesla", "baujahr": None}
        result = build_where_filter(info)
        assert result == {"modell": "Tesla Model 3"}

    def test_returns_none_without_modell(self) -> None:
        """Gibt None zurueck wenn kein Modell erkannt wurde."""
        info = {"modell": None, "hersteller": None, "baujahr": None}
        result = build_where_filter(info)
        assert result is None
