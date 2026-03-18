"""
RAG Chain: Das Herzstueck der Anwendung.

RAG = Retrieval-Augmented Generation
------------------------------------
Das Prinzip in 4 Schritten:

1. RETRIEVAL (Abrufen): Relevante Texte aus ChromaDB holen
   -> "Was wissen wir ueber dieses Fahrzeug?"

2. AUGMENTATION (Anreichern): Diese Texte als Kontext in den Prompt packen
   -> "Hier sind die Fakten, die du nutzen sollst"

3. GENERATION (Erzeugen): LLM generiert eine Antwort basierend auf dem Kontext
   -> "Fasse das fuer den Kaeufer zusammen"

Warum RAG statt einfach das LLM fragen?
- LLMs halluzinieren: Ohne Kontext erfindet Mistral Rueckrufe die es nicht gibt
- LLMs sind veraltet: Das Trainings-Cutoff kennt keine aktuellen Rueckrufe
- Mit RAG nutzt das LLM NUR unsere verifizierten Daten als Grundlage
- Wir koennen die Quellen pruefen und anzeigen

Warum DREI separate Queries statt einer?
-----------------------------------------
Eine einzige Query wie "Alles ueber Tesla Model 3" wuerde zufaellige
Chunks zurueckgeben. Mit drei gezielten Queries stellen wir sicher:

1. Rueckruf-Query:    Findet garantiert Rueckrufe (auch wenn wenige)
2. Schwachstellen-Query: Findet die wichtigsten Probleme
3. Datenblatt-Query:  Holt technische Daten und Empfehlungen

So hat das LLM fuer jeden Bereich des Reports relevanten Kontext.

Warum temperature=0.1?
-----------------------
Temperature steuert die "Kreativitaet" des LLMs:
- 0.0 = Immer die wahrscheinlichste Antwort (deterministisch)
- 0.1 = Fast deterministisch, minimale Variation
- 1.0 = Kreativ, aber unvorhersagbar

Fuer einen Kaufberater wollen wir FAKTEN, keine Kreativitaet.
Daher sehr niedrige Temperatur.

Warum JSON als Output-Format?
-----------------------------
Wir muessen die LLM-Antwort in eine AnalyzeResponse umwandeln.
Dafuer brauchen wir strukturierte Daten. Freitext waere:
- Schwer zu parsen (wo faengt die Checkliste an?)
- Inkonsistent (mal Aufzaehlungszeichen, mal nicht)
- Fehleranfaellig bei der Umwandlung

JSON ist eindeutig strukturiert und direkt in Pydantic ladbar.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import settings
from app.core.vector_store import VectorStore
from app.models.schemas import (
    AnalyzeResponse,
    Quelle,
    Rueckruf,
    Schwachstelle,
)

logger = logging.getLogger(__name__)

# System-Prompt: Definiert die Rolle und Regeln fuer das LLM
# Das ist der wichtigste Teil des RAG-Systems. Hier steuern wir,
# WIE das LLM antwortet.
SYSTEM_PROMPT = """Du bist ein erfahrener KFZ-Gutachter und E-Auto-Experte im DACH-Raum.
Du beraetest Kaeufer von gebrauchten Elektroautos.

REGELN:
1. Antworte NUR basierend auf dem bereitgestellten Kontext. Erfinde NICHTS dazu.
2. Wenn der Kontext keine Information zu einem Punkt enthaelt, sage das ehrlich.
3. Beruecksichtige Baujahr und Kilometerstand bei der Bewertung.
4. Antworte NUR als valides JSON im exakten Format unten. KEIN anderer Text.

JSON-FORMAT:
{
  "risiko_bewertung": "gruen" oder "gelb" oder "rot",
  "zusammenfassung": "2-3 Saetze Gesamtbewertung",
  "rueckrufe": [
    {"beschreibung": "Was wurde zurueckgerufen", "schwere": "niedrig/mittel/hoch"}
  ],
  "schwachstellen": [
    {"problem": "Problembeschreibung", "schwere": "niedrig/mittel/hoch", "haeufigkeit": "selten/gelegentlich/haeufig"}
  ],
  "checkliste": ["Punkt 1", "Punkt 2", "..."],
  "quellen": [
    {"source": "z.B. adac, kba", "doc_type": "z.B. rueckruf, testbericht"}
  ]
}

BEWERTUNGSKRITERIEN fuer risiko_bewertung:
- "gruen": Wenige/keine Rueckrufe, ueberschaubare Schwachstellen, guter Gesamtzustand erwartet
- "gelb": Einige Rueckrufe oder Schwachstellen, bestimmte Punkte muessen geprueft werden
- "rot": Schwere Rueckrufe, viele Schwachstellen, hohes Risiko bei diesem Baujahr/km-Stand"""


class RAGChain:
    """Orchestriert den RAG-Prozess: Retrieval -> Prompt -> LLM -> Response.

    Das ist die zentrale Klasse die alles zusammenbringt:
    VectorStore (Suche) + Ollama (LLM) + Schemas (Response).
    """

    def __init__(self) -> None:
        """Initialisiert VectorStore-Verbindung."""
        self.store = VectorStore()
        logger.info("RAGChain initialisiert")

    def analyze(
        self,
        modell: str,
        baujahr: int,
        km_stand: int,
    ) -> AnalyzeResponse:
        """Fuehrt eine vollstaendige Fahrzeuganalyse durch.

        Das ist der Hauptablauf:
        1. Relevante Chunks aus ChromaDB holen (3 gezielte Queries)
        2. Kontext-String zusammenbauen
        3. Prompt an Mistral schicken
        4. JSON-Antwort parsen und in AnalyzeResponse umwandeln

        Args:
            modell: Fahrzeugmodell, z.B. "Tesla Model 3"
            baujahr: Baujahr des Fahrzeugs
            km_stand: Aktueller Kilometerstand

        Returns:
            AnalyzeResponse mit Risikobewertung, Rueckrufen, etc.
        """
        logger.info(
            "Starte RAG-Analyse: %s, Baujahr %d, %d km",
            modell,
            baujahr,
            km_stand,
        )

        # === Schritt 1: Relevante Chunks aus ChromaDB holen ===
        # Drei separate Queries, weil wir fuer jeden Bereich des Reports
        # gezielt die besten Chunks brauchen (siehe Modul-Docstring oben)
        modell_filter = {"modell": modell}

        # Query a) Rueckrufe: Zuerst spezifisch nach doc_type="rueckruf",
        # falls keine gefunden -> allgemeine Suche nur nach Modell
        rueckruf_chunks = self._safe_query(
            query_text=f"Rueckruf Rueckrufaktion {modell}",
            n_results=5,
            where_filter={"$and": [
                {"modell": modell},
                {"doc_type": "rueckruf"},
            ]},
        )
        logger.info("Rueckruf-Chunks gefunden: %d", len(rueckruf_chunks))

        # Query b) Schwachstellen: Zuerst spezifisch, dann breit
        schwachstellen_chunks = self._safe_query(
            query_text=f"Probleme Schwachstellen Maengel {modell}",
            n_results=5,
            where_filter={"$and": [
                {"modell": modell},
                {"doc_type": "schwachstelle"},
            ]},
        )
        logger.info("Schwachstellen-Chunks gefunden: %d", len(schwachstellen_chunks))

        # Query c) Datenblatt
        datenblatt_chunks = self._safe_query(
            query_text=f"{modell} technische Daten Batterie Empfehlung",
            n_results=2,
            where_filter={"$and": [
                {"modell": modell},
                {"doc_type": "datenblatt"},
            ]},
        )
        logger.info("Datenblatt-Chunks gefunden: %d", len(datenblatt_chunks))

        # Query d) Allgemeine Suche: Fuer alle anderen Kategorien
        # (testbericht, manuell hinzugefuegte Texte, etc.)
        # Diese Query faengt alles auf was nicht in a/b/c gefunden wurde.
        # Wir filtern nur nach Modell, nicht nach doc_type.
        # Damit werden auch Testberichte und manuell hinzugefuegte Texte gefunden.
        allgemein_chunks = self._safe_query(
            query_text=f"{modell} Probleme Schwachstellen Rueckruf Erfahrung",
            n_results=5,
            where_filter={"modell": modell},
        )
        # Duplikate entfernen: Chunks die schon in a/b/c gefunden wurden
        existing_ids = {
            c["id"] for c in rueckruf_chunks + schwachstellen_chunks + datenblatt_chunks
        }
        allgemein_chunks = [c for c in allgemein_chunks if c["id"] not in existing_ids]
        logger.info("Allgemein-Chunks gefunden (nach Dedup): %d", len(allgemein_chunks))

        # === Schritt 2: Kontext-String zusammenbauen ===
        all_chunks = (
            rueckruf_chunks + schwachstellen_chunks
            + datenblatt_chunks + allgemein_chunks
        )

        # Was passiert wenn KEINE Chunks gefunden werden?
        # Das bedeutet: Das Modell ist nicht in unserer Wissensbasis.
        # Wir geben trotzdem eine Antwort, aber mit Hinweis.
        if not all_chunks:
            logger.warning("Keine Chunks fuer %s gefunden!", modell)
            return self._build_no_data_response(modell, baujahr, km_stand)

        context = self._build_context(
            rueckruf_chunks,
            schwachstellen_chunks,
            datenblatt_chunks,
            allgemein_chunks,
        )

        # Quellen sammeln (fuer die Response)
        quellen = self._extract_quellen(all_chunks)

        # === Schritt 3: Prompt bauen und an LLM schicken ===
        prompt = self._build_prompt(modell, baujahr, km_stand, context)
        llm_response = self._call_llm(prompt)

        # === Schritt 4: JSON parsen und AnalyzeResponse bauen ===
        if llm_response:
            result = self._parse_response(llm_response, modell, baujahr, km_stand, quellen)
            if result:
                return result

        # Fallback: Wenn das LLM kein valides JSON zurueckgibt,
        # bauen wir die Response manuell aus den Chunks
        logger.warning("LLM-Antwort konnte nicht geparst werden, nutze Fallback")
        return self._build_fallback_response(
            modell, baujahr, km_stand, all_chunks, quellen,
        )

    def _safe_query(
        self,
        query_text: str,
        n_results: int,
        where_filter: dict | None,
    ) -> list[dict]:
        """Fuehrt eine ChromaDB-Query mit Fehlerbehandlung durch.

        Wenn ChromaDB oder Ollama nicht erreichbar sind, geben wir
        eine leere Liste zurueck statt die ganze Analyse abzubrechen.
        """
        try:
            return self.store.query(
                query_text=query_text,
                n_results=n_results,
                where_filter=where_filter,
            )
        except Exception as e:
            logger.error("Fehler bei ChromaDB-Query: %s", e)
            return []

    def _build_context(
        self,
        rueckruf_chunks: list[dict],
        schwachstellen_chunks: list[dict],
        datenblatt_chunks: list[dict],
        allgemein_chunks: list[dict] | None = None,
    ) -> str:
        """Baut den Kontext-String fuer den LLM-Prompt.

        Strukturiert die Chunks in Kategorien, damit das LLM
        die verschiedenen Informationstypen unterscheiden kann.
        """
        sections: list[str] = []

        if rueckruf_chunks:
            sections.append("=== RUECKRUFE (KBA) ===")
            for chunk in rueckruf_chunks:
                sections.append(chunk["content"])

        if schwachstellen_chunks:
            sections.append("\n=== SCHWACHSTELLEN ===")
            for chunk in schwachstellen_chunks:
                sections.append(chunk["content"])

        if datenblatt_chunks:
            sections.append("\n=== TECHNISCHE DATEN / EMPFEHLUNGEN ===")
            for chunk in datenblatt_chunks:
                sections.append(chunk["content"])

        if allgemein_chunks:
            sections.append("\n=== WEITERE INFORMATIONEN ===")
            for chunk in allgemein_chunks:
                sections.append(chunk["content"])

        return "\n\n".join(sections)

    def _build_prompt(
        self,
        modell: str,
        baujahr: int,
        km_stand: int,
        context: str,
    ) -> str:
        """Baut den vollstaendigen Prompt fuer Mistral.

        Der Prompt besteht aus:
        1. System-Prompt (Rolle + Regeln + JSON-Format)
        2. Kontext (die gefundenen Chunks aus ChromaDB)
        3. Fahrzeug-Info (was der User wissen will)
        """
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== KONTEXT (Wissensbasis) ===\n"
            f"{context}\n\n"
            f"=== FAHRZEUG ===\n"
            f"Modell: {modell}\n"
            f"Baujahr: {baujahr}\n"
            f"Kilometerstand: {km_stand:,} km\n\n"
            f"Erstelle jetzt die Analyse als JSON:"
        )

    def _call_llm(self, prompt: str) -> str | None:
        """Schickt den Prompt an Mistral ueber Ollama.

        Timeout ist 60 Sekunden, weil Mistral auf CPU (kein GPU)
        laeuft und laengere Antworten generieren muss.

        Returns:
            Die LLM-Antwort als String, oder None bei Fehler
        """
        logger.info("Sende Prompt an Mistral (%d Zeichen) ...", len(prompt))

        try:
            # 180s Timeout: Mistral auf dem Lenovo M920q (CPU only, 16 GB RAM)
            # braucht je nach Kontext-Laenge 30-120s fuer eine Analyse.
            # Bei langen Kontexten (viele Chunks) kann es laenger dauern.
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                        },
                    },
                )

            if response.status_code != 200:
                logger.error(
                    "Ollama-Fehler (HTTP %d): %s",
                    response.status_code,
                    response.text[:200],
                )
                return None

            result = response.json().get("response", "")
            logger.info("LLM-Antwort erhalten (%d Zeichen)", len(result))
            logger.debug("LLM-Rohantwort: %s", result[:500])
            return result

        except httpx.ConnectError:
            logger.error(
                "Ollama nicht erreichbar unter %s",
                settings.ollama_base_url,
            )
            return None
        except httpx.TimeoutException:
            logger.error("LLM-Timeout (>180s) - Antwort zu langsam")
            return None

    def _parse_response(
        self,
        raw_text: str,
        modell: str,
        baujahr: int,
        km_stand: int,
        quellen: list[Quelle],
    ) -> AnalyzeResponse | None:
        """Parst die LLM-JSON-Antwort in eine AnalyzeResponse.

        LLMs geben nicht immer sauberes JSON zurueck. Wir versuchen
        verschiedene Strategien um das JSON zu extrahieren.
        """
        # JSON aus der Antwort extrahieren
        data = self._extract_json(raw_text)
        if not data:
            return None

        try:
            return AnalyzeResponse(
                modell=modell,
                baujahr=baujahr,
                km_stand=km_stand,
                risiko_bewertung=self._sanitize_risiko(
                    data.get("risiko_bewertung", "gelb")
                ),
                zusammenfassung=data.get(
                    "zusammenfassung",
                    "Analyse konnte nicht vollstaendig erstellt werden.",
                ),
                rueckrufe=[
                    Rueckruf(
                        beschreibung=r.get("beschreibung", ""),
                        schwere=self._sanitize_schwere(r.get("schwere", "mittel")),
                    )
                    for r in data.get("rueckrufe", [])
                    if r.get("beschreibung")
                ],
                schwachstellen=[
                    Schwachstelle(
                        problem=s.get("problem", ""),
                        schwere=self._sanitize_schwere(s.get("schwere", "mittel")),
                        haeufigkeit=self._sanitize_haeufigkeit(
                            s.get("haeufigkeit", "gelegentlich")
                        ),
                    )
                    for s in data.get("schwachstellen", [])
                    if s.get("problem")
                ],
                checkliste=data.get("checkliste", []),
                quellen=quellen,
            )
        except Exception as e:
            logger.error("Fehler beim Erstellen der AnalyzeResponse: %s", e)
            return None

    def _extract_json(self, raw_text: str) -> dict | None:
        """Extrahiert JSON aus der LLM-Antwort.

        Versucht mehrere Strategien, weil LLMs manchmal Text drumherum packen.
        """
        # Strategie 1: Direktes Parsen
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        # Strategie 2: JSON-Block im Markdown-Code-Block finden
        # Manchmal antwortet das LLM mit: ```json\n{...}\n```
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategie 3: Groesstes JSON-Objekt im Text finden
        # Sucht nach dem aeussersten {...} Block
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Kein valides JSON in LLM-Antwort gefunden: %s",
            raw_text[:200],
        )
        return None

    def _extract_quellen(self, chunks: list[dict]) -> list[Quelle]:
        """Sammelt eindeutige Quellen aus den gefundenen Chunks."""
        seen: set[tuple[str, str]] = set()
        quellen: list[Quelle] = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = (meta.get("source", ""), meta.get("doc_type", ""))
            if key not in seen and key[0]:
                seen.add(key)
                quellen.append(Quelle(source=key[0], doc_type=key[1]))

        return quellen

    def _build_no_data_response(
        self,
        modell: str,
        baujahr: int,
        km_stand: int,
    ) -> AnalyzeResponse:
        """Response wenn das Modell nicht in der Wissensbasis ist."""
        return AnalyzeResponse(
            modell=modell,
            baujahr=baujahr,
            km_stand=km_stand,
            risiko_bewertung="gelb",
            zusammenfassung=(
                f"Fuer {modell} liegen leider keine Daten in unserer "
                f"Wissensbasis vor. Bitte informiere dich bei ADAC, "
                f"OEAMTC oder KBA ueber Rueckrufe und bekannte Probleme."
            ),
            rueckrufe=[],
            schwachstellen=[],
            checkliste=[
                "Batterie-Gesundheit (SoH) auslesen lassen",
                "Service-Historie pruefen",
                "Rueckrufe beim Hersteller abfragen",
                "Probefahrt: Reichweite und Ladeverhalten testen",
                "Unabhaengigen Gutachter hinzuziehen",
            ],
            quellen=[],
        )

    def _build_fallback_response(
        self,
        modell: str,
        baujahr: int,
        km_stand: int,
        chunks: list[dict],
        quellen: list[Quelle],
    ) -> AnalyzeResponse:
        """Fallback-Response wenn das LLM kein valides JSON liefert.

        Baut die Antwort manuell aus den Chunk-Metadaten zusammen.
        Nicht so huebsch wie die LLM-Antwort, aber besser als nichts.
        """
        return AnalyzeResponse(
            modell=modell,
            baujahr=baujahr,
            km_stand=km_stand,
            risiko_bewertung="gelb",
            zusammenfassung=(
                f"Analyse fuer {modell} ({baujahr}, {km_stand:,} km). "
                f"Es wurden {len(chunks)} relevante Eintraege in der "
                f"Wissensbasis gefunden. Die automatische Auswertung "
                f"konnte nicht abgeschlossen werden."
            ),
            rueckrufe=[],
            schwachstellen=[],
            checkliste=[
                "Batterie-Gesundheit (SoH) auslesen lassen",
                "Alle Rueckrufe beim Haendler abfragen",
                "Service-Historie vollstaendig pruefen",
                "Probefahrt: Rekuperation und Ladeverhalten testen",
                "Lack und Spaltmasse kontrollieren",
            ],
            quellen=quellen,
        )

    @staticmethod
    def _sanitize_risiko(value: str) -> str:
        """Stellt sicher, dass risiko_bewertung ein erlaubter Wert ist.

        LLMs geben manchmal 'Gelb' statt 'gelb' oder 'mittel' statt 'gelb'.
        """
        value = value.lower().strip()
        if value in ("gruen", "grün"):
            return "gruen"
        if value in ("rot", "red"):
            return "rot"
        return "gelb"  # Default: im Zweifel Vorsicht

    @staticmethod
    def _sanitize_schwere(value: str) -> str:
        """Normalisiert den Schweregrad auf erlaubte Werte."""
        value = value.lower().strip()
        if value in ("niedrig", "gering", "low"):
            return "niedrig"
        if value in ("hoch", "kritisch", "high", "schwer"):
            return "hoch"
        return "mittel"

    @staticmethod
    def _sanitize_haeufigkeit(value: str) -> str:
        """Normalisiert die Haeufigkeit auf erlaubte Werte."""
        value = value.lower().strip()
        if value in ("selten", "rare"):
            return "selten"
        if value in ("haeufig", "häufig", "oft", "frequent"):
            return "haeufig"
        return "gelegentlich"
