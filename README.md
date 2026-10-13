# Track Lab

Detecta BPM y tono (Camelot), sugiere estilo y renombra tus tracks.

## Usar con ventana
    pip install -r requirements.txt
    python track_lab_gui.py

## Usar por consola
    python track_lab.py "carpeta" [--apply] [--write-tags] [--pattern 0|1|2] [--recursive] [--notags]

## Generar los ejecutables (Windows y Mac)
1. Creá un repositorio en GitHub y subí todo el contenido de esta carpeta (incluida `.github`).
2. En GitHub: pestaña **Actions** → **Build Track Lab** → **Run workflow**.
3. Cuando termine (10-20 min), descargá `TrackLab-Windows` y `TrackLab-macOS` desde **Artifacts**.

Windows: descomprimí y abrí `TrackLab/TrackLab.exe` (si sale SmartScreen: Más información → Ejecutar de todos modos).
Mac: descomprimí, click derecho sobre `TrackLab.app` → Abrir (la primera vez). Build para Apple Silicon (M1+).
