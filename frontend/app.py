# Streamlit Frontend fuer den EV Gebrauchtwagen-Berater.
#
# Streamlit ist ein Python-Framework fuer Web-Apps. Man schreibt normales
# Python und Streamlit macht eine interaktive Webseite daraus.
#
# Starten mit:
#     streamlit run frontend/app.py
#
# Das Frontend kommuniziert mit dem FastAPI-Backend ueber HTTP-Requests.
# Es kennt die Geschaeftslogik NICHT – es zeigt nur an was die API zurueckgibt.
# Das ist das "Client-Server"-Prinzip: Frontend = Client, Backend = Server.

from __future__ import annotations

import httpx
import streamlit as st

# === Konfiguration ===

# URL des FastAPI-Backends
# Muss zum Port passen auf dem das Backend laeuft
API_BASE_URL = "http://localhost:8000/api/v1"

# Farben fuer die Risikobewertung
RISIKO_FARBEN = {
    "gruen": ("#27ae60", "#eafaf1", "Gruen – Guter Zustand"),
    "gelb": ("#f39c12", "#fef9e7", "Gelb – Vorsicht geboten"),
    "rot": ("#e74c3c", "#fdedec", "Rot – Finger weg"),
}

# Farben fuer Schweregrade
SCHWERE_FARBEN = {
    "niedrig": "#3498db",
    "mittel": "#f39c12",
    "hoch": "#e74c3c",
}

# Farben fuer Haeufigkeit
HAEUFIGKEIT_FARBEN = {
    "selten": "#3498db",
    "gelegentlich": "#f39c12",
    "haeufig": "#e74c3c",
}


# === Seiten-Konfiguration ===

st.set_page_config(
    page_title="EV Gebrauchtwagen-Berater",
    page_icon="⚡",
    layout="wide",
)


# === Hilfsfunktionen ===

def lade_modelle() -> list[dict]:
    """Laedt die verfuegbaren Modelle vom Backend.

    Wenn das Backend nicht erreichbar ist, geben wir eine leere Liste
    zurueck und zeigen eine Fehlermeldung.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
    except httpx.ConnectError:
        pass
    except httpx.TimeoutException:
        pass
    return []


def sende_analyse(modell: str, baujahr: int, km_stand: int) -> dict | None:
    """Schickt eine Analyse-Anfrage an das Backend.

    Timeout ist 90 Sekunden, weil das LLM auf CPU laenger brauchen kann.
    """
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{API_BASE_URL}/analyze",
                json={
                    "modell": modell,
                    "baujahr": baujahr,
                    "km_stand": km_stand,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend-Fehler (HTTP {response.status_code}): {response.text}")
            return None
    except httpx.ConnectError:
        st.error(
            "Backend nicht erreichbar. Bitte sicherstellen, dass "
            "das Backend laeuft: `uvicorn app.main:app --reload`"
        )
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


# === Hauptseite ===

# Titel und Untertitel
st.title("⚡ EV Gebrauchtwagen-Berater")
st.markdown("*Dein KI-gestuetzter Kaufberater fuer gebrauchte Elektroautos*")

# Backend-Status pruefen
if not pruefe_backend():
    st.warning(
        "Backend nicht erreichbar. Bitte starten mit: "
        "`uvicorn app.main:app --reload`"
    )

st.divider()

# === Eingabebereich ===

# Modelle vom Backend laden
modelle = lade_modelle()
modell_namen = [m["modell"] for m in modelle] if modelle else []

# Fallback: Wenn keine Modelle geladen werden koennen,
# bieten wir ein freies Textfeld an
if not modell_namen:
    modell_namen = ["Tesla Model 3", "VW ID.3", "Hyundai Ioniq 5"]

# Drei Spalten nebeneinander fuer die Eingabefelder
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
        min_value=2012,
        max_value=2026,
        value=2021,
        step=1,
        help="In welchem Jahr wurde das Fahrzeug gebaut?",
    )

with col3:
    km_stand = st.number_input(
        "Kilometerstand",
        min_value=0,
        max_value=500000,
        value=50000,
        step=5000,
        help="Wie viele Kilometer hat das Fahrzeug auf dem Tacho?",
    )

# Analyse-Button
if st.button("Analysieren", type="primary", use_container_width=True):

    # Spinner waehrend der Analyse
    with st.spinner("Analysiere... (kann bis zu 30 Sekunden dauern auf CPU)"):
        ergebnis = sende_analyse(modell, baujahr, km_stand)

    if ergebnis:
        st.divider()

        # === Risikobewertung ===
        risiko = ergebnis.get("risiko_bewertung", "gelb")
        farbe, hintergrund, label = RISIKO_FARBEN.get(
            risiko, RISIKO_FARBEN["gelb"]
        )

        st.markdown(
            f"""
            <div style="
                background-color: {hintergrund};
                border-left: 5px solid {farbe};
                padding: 16px 20px;
                border-radius: 4px;
                margin-bottom: 16px;
            ">
                <span style="
                    color: {farbe};
                    font-weight: bold;
                    font-size: 1.3em;
                ">Risikobewertung: {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # === Zusammenfassung ===
        st.markdown(f"**Zusammenfassung:** {ergebnis.get('zusammenfassung', '')}")

        st.divider()

        # === Rueckrufe und Schwachstellen nebeneinander ===
        col_left, col_right = st.columns(2)

        # --- Rueckrufe ---
        with col_left:
            rueckrufe = ergebnis.get("rueckrufe", [])
            st.subheader(f"Rueckrufe ({len(rueckrufe)})")

            if rueckrufe:
                for i, rueckruf in enumerate(rueckrufe, 1):
                    schwere = rueckruf.get("schwere", "mittel")
                    schwere_farbe = SCHWERE_FARBEN.get(schwere, "#999")

                    with st.expander(
                        f"Rueckruf {i} – Schwere: {schwere.capitalize()}"
                    ):
                        st.markdown(
                            f"<span style='color: {schwere_farbe}; "
                            f"font-weight: bold;'>Schwere: {schwere.upper()}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )
                        st.write(rueckruf.get("beschreibung", ""))
            else:
                st.info("Keine Rueckrufe bekannt.")

        # --- Schwachstellen ---
        with col_right:
            schwachstellen = ergebnis.get("schwachstellen", [])
            st.subheader(f"Schwachstellen ({len(schwachstellen)})")

            if schwachstellen:
                for i, sw in enumerate(schwachstellen, 1):
                    schwere = sw.get("schwere", "mittel")
                    haeufigkeit = sw.get("haeufigkeit", "gelegentlich")
                    schwere_farbe = SCHWERE_FARBEN.get(schwere, "#999")
                    haeufigkeit_farbe = HAEUFIGKEIT_FARBEN.get(
                        haeufigkeit, "#999"
                    )

                    with st.expander(
                        f"Schwachstelle {i} – {schwere.capitalize()} / "
                        f"{haeufigkeit.capitalize()}"
                    ):
                        st.markdown(
                            f"<span style='color: {schwere_farbe}; "
                            f"font-weight: bold;'>Schwere: {schwere.upper()}"
                            f"</span> &nbsp;|&nbsp; "
                            f"<span style='color: {haeufigkeit_farbe};'>"
                            f"Haeufigkeit: {haeufigkeit}</span>",
                            unsafe_allow_html=True,
                        )
                        st.write(sw.get("problem", ""))
            else:
                st.info("Keine Schwachstellen bekannt.")

        st.divider()

        # === Checkliste ===
        checkliste = ergebnis.get("checkliste", [])
        if checkliste:
            st.subheader("Besichtigungs-Checkliste")
            for i, punkt in enumerate(checkliste, 1):
                st.checkbox(punkt, key=f"check_{i}", value=False)

        # === Quellen ===
        quellen = ergebnis.get("quellen", [])
        if quellen:
            st.divider()
            st.subheader("Quellen")
            tags_html = " ".join(
                f"<span style='"
                f"background-color: #f0f0f0; "
                f"padding: 4px 10px; "
                f"border-radius: 12px; "
                f"font-size: 0.85em; "
                f"margin-right: 6px; "
                f"color: #555;"
                f"'>{q['source'].upper()} ({q['doc_type']})</span>"
                for q in quellen
            )
            st.markdown(tags_html, unsafe_allow_html=True)

# === Footer ===
st.divider()
st.caption("DSGVO-konform – Alle Daten werden lokal verarbeitet. Kein Cloud-Service, keine externen APIs.")
