#!/usr/bin/env python3
"""Track Lab: detecta BPM y tono, y renombra tus tracks en lote.

Instalar:  pip install librosa numpy mutagen
Probar:    python track_lab.py "C:/Musica/Nuevos"            (solo muestra, no toca nada)
Aplicar:   python track_lab.py "C:/Musica/Nuevos" --apply
Deshacer:  python track_lab.py --undo track_lab_log.csv
Opciones:  --pattern 0|1|2   0: 8A - 124 - Titulo | 1: Titulo [8A 124] | 2: 124 - 8A - Titulo
           --recursive       incluye subcarpetas
           --write-tags      con --apply, escribe BPM y tono (Camelot) dentro del archivo
           --notags          ignora los tags ID3 y usa solo el nombre del archivo
"""
import argparse, csv, re, sys
from pathlib import Path
import numpy as np
import librosa

try:
    import mutagen
except ImportError:
    mutagen = None

EXT = {".mp3", ".wav", ".flac", ".m4a", ".aif", ".aiff", ".ogg"}
NOTES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJ = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]  # numero Camelot por tonica mayor
KM = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KN = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_bpm(y, sr):
    """BPM con resolucion fina: autocorrelacion + ajuste por tren de pulsos (0.1 BPM)."""
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env - env.mean()
    fps = sr / hop
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 70)
    ac = librosa.autocorrelate(env, max_size=2 * hi + 2)
    score = ac[lo:hi + 1] + 0.5 * ac[2 * lo:2 * hi + 1:2]
    b0 = 60 * fps / (lo + int(np.argmax(score)))
    x = np.arange(len(env))
    best, bpm = -1e18, b0
    for b in np.arange(b0 - 3, b0 + 3, 0.1):
        p = 60 * fps / b
        k = np.arange(int(len(env) / p) - 1)
        pos = np.arange(0, p, 1.0)[:, None] + k[None, :] * p
        sc = np.interp(pos, x, env).sum(axis=1).max()
        if sc > best:
            best, bpm = sc, b
    while bpm < 80: bpm *= 2
    while bpm > 175: bpm /= 2
    return round(float(bpm), 1)


def analyze(path):
    dur = librosa.get_duration(path=str(path))
    y, sr = librosa.load(str(path), sr=22050, mono=True,
                         offset=max(0, (dur - 70) / 2), duration=70)
    # BPM
    bpm = detect_bpm(y, sr)
    # Tono
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr).mean(axis=1)
    best, res = -9, None
    for t in range(12):
        c = np.roll(chroma, -t)
        for minor, prof in ((0, KM), (1, KN)):
            r = np.corrcoef(c, prof)[0, 1]
            if r > best:
                best, res = r, (t, minor)
    t, minor = res
    num = MAJ[(t + 3) % 12] if minor else MAJ[t]
    return dict(bpm=round(bpm, 1), num=num, letter="A" if minor else "B",
                cam=f"{num}{'A' if minor else 'B'}", note=NOTES[t] + ("m" if minor else ""))


def clean(stem):
    s = stem.replace("_", " ")
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\((official|video|audio|lyrics|hq|hd|free)[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"www\.\S+", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,2}[AB]\s*[-–]\s*\d{2,3}(\.\d)?\s*[-–]\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,3}\s*[-.]\s+(?=\D)", "", s)
    s = re.sub(r"[\\/:*?\"<>|]", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def title_from(path, use_tags=True):
    """Artista - Titulo desde los tags; si faltan, desde el nombre del archivo."""
    if use_tags and mutagen:
        try:
            t = mutagen.File(str(path), easy=True)
            artist = (t.get("artist") or [""])[0].strip()
            title = (t.get("title") or [""])[0].strip()
            if title:
                return clean(f"{artist} - {title}" if artist else title)
        except Exception:
            pass
    return clean(path.stem)


def write_tags(path, info):
    """Escribe BPM y tono (Camelot) en MP3, FLAC y M4A. Devuelve True si pudo."""
    bpm, key, ext = str(round(info["bpm"])), info["cam"], path.suffix.lower()
    if ext == ".mp3":
        from mutagen.id3 import ID3, ID3NoHeaderError, TBPM, TKEY
        try:
            t = ID3(str(path))
        except ID3NoHeaderError:
            t = ID3()
        t.setall("TBPM", [TBPM(encoding=3, text=bpm)])
        t.setall("TKEY", [TKEY(encoding=3, text=key)])
        t.save(str(path))
    elif ext == ".flac":
        from mutagen.flac import FLAC
        t = FLAC(str(path))
        t["BPM"], t["INITIALKEY"] = bpm, key
        t.save()
    elif ext == ".m4a":
        from mutagen.mp4 import MP4, MP4FreeForm
        t = MP4(str(path))
        t["tmpo"] = [int(bpm)]
        t["----:com.apple.iTunes:initialkey"] = [MP4FreeForm(key.encode())]
        t.save()
    else:
        return False
    return True


def new_name(title, a, ext, pattern):
    b = round(a["bpm"])
    k = a["cam"]
    base = [f"{k} - {b} - {title}", f"{title} [{k} {b}]", f"{b} - {k} - {title}"][pattern]
    return base + ext


def compat(a, b):
    d = (b["num"] - a["num"]) % 12
    d = d - 12 if d > 6 else d
    same = a["letter"] == b["letter"]
    if same and d == 0: s = 100
    elif d == 0: s = 92
    elif same and abs(d) == 1: s = 90
    elif same and abs(d) == 2: s = 70
    elif abs(d) == 1: s = 55
    else: s = 20
    pc = abs(b["bpm"] - a["bpm"]) / a["bpm"] * 100
    return round(s * (1 if pc <= 3 else .85 if pc <= 6 else .6))


def style(b):
    for lim, n in ((100, "Reggaeton / Cumbia / Hip-Hop"), (115, "Funk / Nu-Disco / Pop"),
                   (122, "Deep House / Disco"), (128, "House"), (135, "Tech House"),
                   (145, "Techno"), (160, "Hard Techno / Trance")):
        if b < lim: return n
    return "Drum & Bass"


def undo(log):
    with open(log, newline="", encoding="utf-8") as f:
        for old, new in csv.reader(f):
            if Path(new).exists() and not Path(old).exists():
                Path(new).rename(old)
                print(f"↩  {Path(new).name} -> {Path(old).name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--pattern", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--write-tags", action="store_true")
    ap.add_argument("--notags", action="store_true")
    ap.add_argument("--undo")
    a = ap.parse_args()
    if a.undo:
        return undo(a.undo)
    if not a.folder:
        ap.error("falta la carpeta")
    root = Path(a.folder)
    files = sorted(p for p in (root.rglob("*") if a.recursive else root.glob("*"))
                   if p.suffix.lower() in EXT)
    if not files:
        sys.exit("No encontré audios en esa carpeta.")
    rows = []
    for i, p in enumerate(files, 1):
        try:
            info = analyze(p)
        except Exception as e:
            print(f"[{i}/{len(files)}] ✗ {p.name} ({e})")
            continue
        target = p.with_name(new_name(title_from(p, not a.notags), info, p.suffix, a.pattern))
        rows.append((p, target, info))
        print(f"[{i}/{len(files)}] {info['cam']:>3} {info['note']:<4} {info['bpm']:>6} BPM  "
              f"{style(info['bpm'])}\n        {p.name}\n     -> {target.name}")

    # orden de mezcla sugerido (greedy por compatibilidad)
    if len(rows) > 1:
        left, order = rows[1:], [rows[0]]
        while left:
            nxt = max(left, key=lambda r: compat(order[-1][2], r[2]))
            left.remove(nxt)
            order.append(nxt)
        print("\nOrden de mezcla sugerido:")
        for n, (_, t, inf) in enumerate(order, 1):
            print(f"  {n:>2}. {inf['cam']:>3} {inf['bpm']:>6}  {t.name}")

    if not a.apply:
        print("\nVista previa. Agregá --apply para renombrar de verdad.")
        return
    with open("track_lab_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p, t, info in rows:
            if a.write_tags and mutagen:
                try:
                    if not write_tags(p, info):
                        print(f"(sin tags para {p.suffix}: {p.name})")
                except Exception as e:
                    print(f"✗ tags en {p.name}: {e}")
            if t != p and not t.exists():
                p.rename(t)
                w.writerow([str(p), str(t)])
    print("\nListo. Para deshacer: python track_lab.py --undo track_lab_log.csv")


if __name__ == "__main__":
    main()
