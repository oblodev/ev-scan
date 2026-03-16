"""
Pydantic Datenmodelle fuer die EV-Scan API.

Warum Pydantic-Modelle statt einfache Dicts?
---------------------------------------------
1. VALIDIERUNG: Pydantic prueft automatisch ob die Daten den richtigen Typ haben.
   Wenn jemand baujahr="abc" schickt, gibt es sofort einen klaren Fehler (422),
   statt dass der Fehler erst irgendwo tief im Code auftaucht.

2. DOKUMENTATION: FastAPI generiert aus den Modellen automatisch eine
   OpenAPI/Swagger-Doku. Jeder sieht sofort welche Felder es gibt und
   welche Typen erwartet werden.

3. TYPSICHERHEIT: Die IDE kennt die Felder und deren Typen -> Autocomplete
   und Fehlererkennung beim Entwickeln.

4. SERIALISIERUNG: Pydantic wandelt die Modelle automatisch in JSON um
   und zurueck. Kein manuelles dict-Gebastel noetig.

5. EINSCHRAENKUNG: Mit Literal[] koennen wir erlaubte Werte definieren.
   z.B. schwere darf nur "niedrig", "mittel" oder "hoch" sein.
   Bei einem Dict wuerde jeder beliebige String durchgehen.
"""

from typing import Literal

from pydantic import BaseModel, Field


# === Request-Modelle ===

class AnalyzeRequest(BaseModel):
    """Anfrage fuer die Fahrzeuganalyse.

    Der User schickt Modell, Baujahr und Kilometerstand.
    Pydantic stellt sicher, dass baujahr und km_stand wirklich
    Ganzzahlen sind - schickt jemand einen String wie "zwanzig",
    kommt automatisch ein 422 Validation Error zurueck.
    """
    modell: str = Field(
        ...,
        description="Fahrzeugmodell, z.B. 'Tesla Model 3'",
        min_length=1
    )
    baujahr: int = Field(
        ...,
        description="Baujahr des Fahrzeugs",
        ge=2010,  # Erste massentaugliche EVs ab ca. 2010
        le=2026
    )
    km_stand: int = Field(
        ...,
        description="Aktueller Kilometerstand",
        ge=0,
        le=1_000_000
    )


# === Teile der Analyse-Antwort ===

class Rueckruf(BaseModel):
    """Ein einzelner Rueckruf fuer das Fahrzeug."""
    beschreibung: str = Field(
        ...,
        description="Was wurde zurueckgerufen und warum"
    )
    schwere: Literal["niedrig", "mittel", "hoch"] = Field(
        ...,
        description="Schweregrad des Rueckrufs"
    )


class Schwachstelle(BaseModel):
    """Eine bekannte Schwachstelle des Fahrzeugmodells."""
    problem: str = Field(
        ...,
        description="Beschreibung des Problems"
    )
    schwere: Literal["niedrig", "mittel", "hoch"] = Field(
        ...,
        description="Schweregrad der Schwachstelle"
    )
    haeufigkeit: Literal["selten", "gelegentlich", "haeufig"] = Field(
        ...,
        description="Wie oft dieses Problem auftritt"
    )


class Quelle(BaseModel):
    """Quellenangabe fuer die Analyse-Ergebnisse."""
    source: str = Field(
        ...,
        description="Woher die Info kommt, z.B. 'adac', 'kba', 'oeamtc'"
    )
    doc_type: str = Field(
        ...,
        description="Art des Dokuments, z.B. 'rueckruf', 'testbericht'"
    )


# === Response-Modelle ===

class AnalyzeResponse(BaseModel):
    """Vollstaendige Analyse-Antwort.

    Enthaelt alle Informationen die der User nach der Analyse bekommt:
    Risikobewertung, Rueckrufe, Schwachstellen und eine Checkliste
    fuer die Besichtigung.
    """
    modell: str = Field(
        ...,
        description="Analysiertes Fahrzeugmodell"
    )
    baujahr: int = Field(
        ...,
        description="Baujahr des analysierten Fahrzeugs"
    )
    km_stand: int = Field(
        ...,
        description="Kilometerstand des analysierten Fahrzeugs"
    )
    risiko_bewertung: Literal["gruen", "gelb", "rot"] = Field(
        ...,
        description="Gesamtbewertung: gruen=gut, gelb=Vorsicht, rot=Finger weg"
    )
    zusammenfassung: str = Field(
        ...,
        description="Kurze Zusammenfassung der Analyse in 2-3 Saetzen"
    )
    rueckrufe: list[Rueckruf] = Field(
        default_factory=list,
        description="Liste bekannter Rueckrufe"
    )
    schwachstellen: list[Schwachstelle] = Field(
        default_factory=list,
        description="Liste bekannter Schwachstellen"
    )
    checkliste: list[str] = Field(
        default_factory=list,
        description="Punkte die bei der Besichtigung geprueft werden sollten"
    )
    quellen: list[Quelle] = Field(
        default_factory=list,
        description="Welche Quellen fuer die Analyse herangezogen wurden"
    )


class ModelInfo(BaseModel):
    """Info ueber ein Fahrzeugmodell in unserer Wissensbasis."""
    modell: str = Field(
        ...,
        description="Fahrzeugmodell, z.B. 'Tesla Model 3'"
    )
    hersteller: str = Field(
        ...,
        description="Hersteller, z.B. 'Tesla'"
    )
    docs_count: int = Field(
        ...,
        description="Anzahl Dokumente in der Wissensbasis fuer dieses Modell",
        ge=0
    )
