# === Dockerfile fuer den EV Gebrauchtwagen-Berater ===
#
# Was ist ein Dockerfile?
# -----------------------
# Ein Rezept das beschreibt, wie ein Container-Image gebaut wird.
# Ein Image ist wie eine "Momentaufnahme" eines kompletten Systems:
# Betriebssystem + Python + alle Abhaengigkeiten + unser Code.
#
# Warum Container?
# - "Works on my machine" Problem geloest: Laeuft ueberall gleich
# - Isoliert: Veraendert nichts am Host-System
# - Reproduzierbar: Jeder Build ergibt das gleiche Image
#
# Gebaut wird mit: docker compose build
# Gestartet wird mit: docker compose up

# === Basis-Image ===
# python:3.11-slim ist ein minimales Linux mit Python vorinstalliert.
# "slim" bedeutet: Nur das Noetigste, kein gcc, kein git, etc.
# Das macht das Image kleiner (ca. 150 MB statt 1 GB).
FROM python:3.11-slim

# === Arbeitsverzeichnis im Container ===
# Alle folgenden Befehle werden relativ zu /app ausgefuehrt.
# Wird automatisch erstellt falls es nicht existiert.
WORKDIR /app

# === Abhaengigkeiten installieren ===
# TRICK: Wir kopieren ZUERST nur die requirements.txt und installieren.
# Warum? Docker cached jeden Schritt (Layer). Wenn sich nur der Code
# aendert aber nicht die requirements.txt, muss Docker die Pakete
# nicht nochmal installieren. Das spart beim Rebuild viel Zeit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === App-Code kopieren ===
# Jetzt erst den restlichen Code. Aenderungen am Code invalidieren
# nur diesen Layer, nicht den pip install Layer darüber.
COPY app/ app/
COPY frontend/ frontend/

# === Daten kopieren ===
# Die Wissensbasis (JSON, TXT Dateien) wird ins Image gebacken.
# Alternativ koennten wir sie als Volume mounten (flexibler).
COPY data/ data/

# === Port freigeben ===
# EXPOSE dokumentiert welchen Port der Container nutzt.
# Es oeffnet den Port NICHT automatisch – das macht docker-compose
# mit der "ports" Konfiguration.
EXPOSE 8000

# === Startbefehl ===
# CMD definiert was beim Container-Start ausgefuehrt wird.
# uvicorn startet unsere FastAPI-App.
# --host 0.0.0.0 bedeutet: Auf ALLEN Netzwerk-Interfaces lauschen.
# Ohne das waere die App nur innerhalb des Containers erreichbar,
# nicht von aussen (vom Host oder anderen Containern).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
