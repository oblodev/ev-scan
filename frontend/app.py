# Streamlit Frontend fuer den EV Gebrauchtwagen-Berater.
#
# Zwei Tabs:
# 1. "Analyse" – Fahrzeuganalyse (Hauptfunktion)
# 2. "Wissen verwalten" – Wissensbasis erweitern und ueberblicken
#
# Das Frontend kommuniziert mit dem FastAPI-Backend ueber HTTP-Requests.
# Es kennt die Geschaeftslogik NICHT – es zeigt nur an was die API zurueckgibt.

from __future__ import annotations

import os

import httpx
import streamlit as st

# === Konfiguration ===

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000") + "/api/v1"

RISIKO_FARBEN = {
    "gruen": ("#27ae60", "#eafaf1", "Gruen – Guter Zustand"),
    "gelb": ("#f39c12", "#fef9e7", "Gelb – Vorsicht geboten"),
    "rot": ("#e74c3c", "#fdedec", "Rot – Finger weg"),
}

SCHWERE_FARBEN = {
    "niedrig": "#3498db",
    "mittel": "#f39c12",
    "hoch": "#e74c3c",
}

HAEUFIGKEIT_FARBEN = {
    "selten": "#3498db",
    "gelegentlich": "#f39c12",
    "haeufig": "#e74c3c",
}

KATEGORIEN = ["testbericht", "rueckruf", "schwachstelle", "datenblatt"]


# === Seiten-Konfiguration ===

st.set_page_config(
    page_title="EV Gebrauchtwagen-Berater",
    page_icon="⚡",
    layout="wide",
)


# === Hilfsfunktionen ===

def lade_modelle() -> list[dict]:
    """Laedt die verfuegbaren Modelle vom Backend."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return []


def sende_analyse(modell: str, baujahr: int, km_stand: int) -> dict | None:
    """Schickt eine Analyse-Anfrage an das Backend."""
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{API_BASE_URL}/analyze",
                json={"modell": modell, "baujahr": baujahr, "km_stand": km_stand},
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend-Fehler (HTTP {response.status_code}): {response.text}")
            return None
    except httpx.ConnectError:
        st.error("Backend nicht erreichbar. Bitte starten mit: `uvicorn app.main:app --reload`")
        return None
    except httpx.TimeoutException:
        st.error("Timeout – die Analyse dauert zu lange. Bitte erneut versuchen.")
        return None


def pruefe_backend() -> bool:
    """Prueft ob das Backend erreichbar ist."""
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def lade_stats() -> dict | None:
    """Laedt Wissensbasis-Statistiken vom Backend."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/knowledge/stats")
        if response.status_code == 200:
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None


def sende_text_ingest(text: str, kategorie: str, modell: str, quelle: str) -> dict | None:
    """Sendet Text zum Ingest an das Backend."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE_URL}/ingest/text",
                json={
                    "text": text,
                    "kategorie": kategorie,
                    "modell": modell,
                    "quelle": quelle,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Fehler (HTTP {response.status_code}): {response.text}")
            return None
    except httpx.ConnectError:
        st.error("Backend nicht erreichbar.")
        return None
    except httpx.TimeoutException:
        st.error("Timeout beim Verarbeiten.")
        return None


def sende_datei_ingest(
    file_bytes: bytes,
    filename: str,
    kategorie: str,
    modell: str,
    quelle: str,
) -> dict | None:
    """Sendet eine Datei zum Ingest an das Backend."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_BASE_URL}/ingest/file",
                files={"file": (filename, file_bytes)},
                data={
                    "kategorie": kategorie,
                    "modell": modell,
                    "quelle": quelle,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Fehler (HTTP {response.status_code}): {response.text}")
            return None
    except httpx.ConnectError:
        st.error("Backend nicht erreichbar.")
        return None
    except httpx.TimeoutException:
        st.error("Timeout beim Verarbeiten der Datei.")
        return None


def loesche_modell(modell: str) -> bool:
    """Loescht alle Daten fuer ein Modell."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.delete(f"{API_BASE_URL}/knowledge/{modell}")
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# === Hauptseite ===

st.title("⚡ EV Gebrauchtwagen-Berater")
st.markdown("*Dein KI-gestuetzter Kaufberater fuer gebrauchte Elektroautos*")

if not pruefe_backend():
    st.warning("Backend nicht erreichbar. Bitte starten mit: `uvicorn app.main:app --reload`")

# === Tabs ===
tab_analyse, tab_wissen = st.tabs(["Analyse", "Wissen verwalten"])

# ==========================================
# TAB 1: Analyse (bisherige Hauptfunktion)
# ==========================================
with tab_analyse:
    st.divider()

    modelle = lade_modelle()
    modell_namen = [m["modell"] for m in modelle] if modelle else []
    if not modell_namen:
        modell_namen = ["Tesla Model 3", "VW ID.3", "Hyundai Ioniq 5"]

    col1, col2, col3 = st.columns(3)

    with col1:
        modell = st.selectbox(
            "Fahrzeugmodell",
            options=modell_namen,
            help="Waehle das E-Auto-Modell das du kaufen moechtest",
        )

    with col2:
        baujahr = st.number_input(
            "Baujahr",
            min_value=2012, max_value=2026, value=2021, step=1,
        )

    with col3:
        km_stand = st.number_input(
            "Kilometerstand",
            min_value=0, max_value=500000, value=50000, step=5000,
        )

    if st.button("Analysieren", type="primary", use_container_width=True):
        with st.spinner("Analysiere... (kann bis zu 30 Sekunden dauern auf CPU)"):
            ergebnis = sende_analyse(modell, baujahr, km_stand)

        if ergebnis:
            st.divider()

            # Risikobewertung
            risiko = ergebnis.get("risiko_bewertung", "gelb")
            farbe, hintergrund, label = RISIKO_FARBEN.get(risiko, RISIKO_FARBEN["gelb"])

            st.markdown(
                f'<div style="background-color: {hintergrund}; border-left: 5px solid {farbe}; '
                f'padding: 16px 20px; border-radius: 4px; margin-bottom: 16px;">'
                f'<span style="color: {farbe}; font-weight: bold; font-size: 1.3em;">'
                f'Risikobewertung: {label}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(f"**Zusammenfassung:** {ergebnis.get('zusammenfassung', '')}")
            st.divider()

            # Rueckrufe und Schwachstellen nebeneinander
            col_left, col_right = st.columns(2)

            with col_left:
                rueckrufe = ergebnis.get("rueckrufe", [])
                st.subheader(f"Rueckrufe ({len(rueckrufe)})")
                if rueckrufe:
                    for i, r in enumerate(rueckrufe, 1):
                        s = r.get("schwere", "mittel")
                        with st.expander(f"Rueckruf {i} – Schwere: {s.capitalize()}"):
                            st.markdown(
                                f"<span style='color: {SCHWERE_FARBEN.get(s, '#999')}; "
                                f"font-weight: bold;'>Schwere: {s.upper()}</span>",
                                unsafe_allow_html=True,
                            )
                            st.write(r.get("beschreibung", ""))
                else:
                    st.info("Keine Rueckrufe bekannt.")

            with col_right:
                schwachstellen = ergebnis.get("schwachstellen", [])
                st.subheader(f"Schwachstellen ({len(schwachstellen)})")
                if schwachstellen:
                    for i, sw in enumerate(schwachstellen, 1):
                        s = sw.get("schwere", "mittel")
                        h = sw.get("haeufigkeit", "gelegentlich")
                        with st.expander(f"Schwachstelle {i} – {s.capitalize()} / {h.capitalize()}"):
                            st.markdown(
                                f"<span style='color: {SCHWERE_FARBEN.get(s, '#999')}; "
                                f"font-weight: bold;'>Schwere: {s.upper()}</span> &nbsp;|&nbsp; "
                                f"<span style='color: {HAEUFIGKEIT_FARBEN.get(h, '#999')};'>"
                                f"Haeufigkeit: {h}</span>",
                                unsafe_allow_html=True,
                            )
                            st.write(sw.get("problem", ""))
                else:
                    st.info("Keine Schwachstellen bekannt.")

            st.divider()

            # Checkliste
            checkliste = ergebnis.get("checkliste", [])
            if checkliste:
                st.subheader("Besichtigungs-Checkliste")
                for i, punkt in enumerate(checkliste, 1):
                    st.checkbox(punkt, key=f"check_{i}", value=False)

            # Quellen
            quellen = ergebnis.get("quellen", [])
            if quellen:
                st.divider()
                st.subheader("Quellen")
                tags_html = " ".join(
                    f"<span style='background-color: #f0f0f0; padding: 4px 10px; "
                    f"border-radius: 12px; font-size: 0.85em; margin-right: 6px; "
                    f"color: #555;'>{q['source'].upper()} ({q['doc_type']})</span>"
                    for q in quellen
                )
                st.markdown(tags_html, unsafe_allow_html=True)


# ==========================================
# TAB 2: Wissen verwalten
# ==========================================
with tab_wissen:
    st.divider()

    # Modelle fuer Dropdowns laden
    modelle_wissen = lade_modelle()
    vorhandene_modelle = [m["modell"] for m in modelle_wissen] if modelle_wissen else []

    # --- Bereich 1: Text hinzufuegen ---
    st.subheader("Text hinzufuegen")
    st.caption(
        "Kopiere einen Text (z.B. aus einem ADAC-Artikel oder KBA-Rueckruf) "
        "hier rein. Er wird automatisch aufbereitet und in die Wissensbasis gespeichert."
    )

    text_input = st.text_area(
        "Text",
        height=200,
        placeholder="Mindestens 50 Zeichen. Z.B. einen Absatz aus einem Testbericht...",
        key="ingest_text",
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        text_kategorie = st.selectbox(
            "Kategorie",
            options=KATEGORIEN,
            key="text_kat",
        )
    with col_t2:
        # Vorhandene Modelle + Option fuer neues Modell
        modell_optionen = vorhandene_modelle + ["-- Neues Modell --"]
        text_modell_auswahl = st.selectbox(
            "Modell",
            options=modell_optionen if modell_optionen else ["-- Neues Modell --"],
            key="text_modell",
        )

    # Wenn "Neues Modell" gewaehlt, Textfeld anzeigen
    if text_modell_auswahl == "-- Neues Modell --":
        text_modell = st.text_input(
            "Neuer Modellname",
            placeholder="z.B. Renault Zoe",
            key="text_neues_modell",
        )
    else:
        text_modell = text_modell_auswahl

    text_quelle = st.text_input(
        "Quelle (optional)",
        placeholder="z.B. ADAC Test 2024",
        key="text_quelle",
    )

    if st.button("Hinzufuegen", key="btn_text", use_container_width=True):
        if not text_input or len(text_input) < 50:
            st.error("Der Text muss mindestens 50 Zeichen lang sein.")
        elif not text_modell:
            st.error("Bitte ein Modell auswaehlen oder eingeben.")
        else:
            with st.spinner("Verarbeite Text..."):
                result = sende_text_ingest(
                    text=text_input,
                    kategorie=text_kategorie,
                    modell=text_modell,
                    quelle=text_quelle or "manuell",
                )
            if result:
                st.success(
                    f"Erfolgreich! {result['chunks_added']} Chunks erstellt "
                    f"und in die Wissensbasis gespeichert."
                )

    st.divider()

    # --- Bereich 2: Datei hochladen ---
    st.subheader("Datei hochladen")
    st.caption("Lade eine PDF, CSV oder TXT-Datei hoch. Der Text wird automatisch extrahiert.")

    uploaded_file = st.file_uploader(
        "Datei waehlen",
        type=["pdf", "csv", "txt"],
        key="file_upload",
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        file_kategorie = st.selectbox(
            "Kategorie",
            options=KATEGORIEN,
            key="file_kat",
        )
    with col_f2:
        file_modell_optionen = vorhandene_modelle + ["-- Neues Modell --"]
        file_modell_auswahl = st.selectbox(
            "Modell",
            options=file_modell_optionen if file_modell_optionen else ["-- Neues Modell --"],
            key="file_modell",
        )

    if file_modell_auswahl == "-- Neues Modell --":
        file_modell = st.text_input(
            "Neuer Modellname",
            placeholder="z.B. BMW iX3",
            key="file_neues_modell",
        )
    else:
        file_modell = file_modell_auswahl

    file_quelle = st.text_input(
        "Quelle (optional)",
        placeholder="z.B. KBA Rueckrufliste 2024",
        key="file_quelle",
    )

    if st.button("Hochladen und verarbeiten", key="btn_file", use_container_width=True):
        if not uploaded_file:
            st.error("Bitte eine Datei auswaehlen.")
        elif not file_modell:
            st.error("Bitte ein Modell auswaehlen oder eingeben.")
        else:
            with st.spinner(f"Verarbeite {uploaded_file.name}..."):
                result = sende_datei_ingest(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    kategorie=file_kategorie,
                    modell=file_modell,
                    quelle=file_quelle or f"datei:{uploaded_file.name}",
                )
            if result:
                st.success(
                    f"Erfolgreich! {result['chunks_added']} Chunks aus "
                    f"'{result.get('filename', uploaded_file.name)}' erstellt."
                )

    st.divider()

    # --- Bereich 3: Wissensbasis-Uebersicht ---
    st.subheader("Wissensbasis-Uebersicht")

    stats = lade_stats()

    if stats and stats.get("total_chunks", 0) > 0:
        st.metric("Gesamt-Chunks", stats["total_chunks"])

        # Tabelle: Modell | pro Kategorie | Gesamt
        models = stats.get("models", {})
        if models:
            # Daten fuer die Tabelle aufbereiten
            table_data: list[dict] = []
            for modell_name, count in sorted(models.items()):
                table_data.append({
                    "Modell": modell_name,
                    "Chunks": count,
                })

            st.table(table_data)

            # Kategorien anzeigen
            categories = stats.get("categories", {})
            if categories:
                st.markdown("**Chunks pro Kategorie:**")
                cat_cols = st.columns(len(categories))
                for i, (cat, count) in enumerate(sorted(categories.items())):
                    with cat_cols[i]:
                        st.metric(cat.capitalize(), count)

            st.divider()

            # Loeschen-Bereich
            st.subheader("Modell-Daten loeschen")
            st.caption("Loescht alle Chunks fuer ein Modell. Nuetzlich bei veralteten Daten.")

            del_modell = st.selectbox(
                "Modell zum Loeschen",
                options=list(models.keys()),
                key="del_modell",
            )

            if st.button(
                f"Alle Daten fuer '{del_modell}' loeschen",
                key="btn_delete",
                type="secondary",
            ):
                if loesche_modell(del_modell):
                    st.success(f"Alle Chunks fuer '{del_modell}' geloescht.")
                    st.rerun()
                else:
                    st.error("Fehler beim Loeschen.")
    else:
        st.info(
            "Die Wissensbasis ist leer. Fuehre den Ingest durch oder "
            "fuege oben manuell Texte hinzu."
        )

# === Footer ===
st.divider()
st.caption("DSGVO-konform – Alle Daten werden lokal verarbeitet. Kein Cloud-Service, keine externen APIs.")
