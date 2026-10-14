#!/usr/bin/env python3
"""Track Lab: detecta BPM y tono, y renombra tus tracks en lote (liviano y en paralelo).

Instalar:  pip install numpy soundfile mutagen
Probar:    python track_lab.py "C:/Musica/Nuevos"            (solo muestra, no toca nada)
Aplicar:   python track_lab.py "C:/Musica/Nuevos" --apply
Deshacer:  python track_lab.py --undo track_lab_log.csv
Opciones:  --pattern 0|1|2   0: 8A - 124 - Titulo | 1: Titulo [8A 124] | 2: 124 - 8A - Titulo
           --recursive       incluye subcarpetas
           --jobs N          tracks analizados en paralelo (por defecto: nucleos - 1)
           --write-tags      con --apply, escribe BPM y tono (Camelot) dentro del archivo
           --notags          ignora los tags ID3 y usa solo el nombre del archivo
           --artist-first    Artista - Tono - BPM - Titulo (el artista va adelante)
           --file-order X    title-artist (por defecto) o artist-title: como se llaman tus archivos sin tags
MP3, WAV, FLAC, OGG y AIFF se leen directo. M4A/AAC necesitan ffmpeg instalado.
"""
import argparse, csv, multiprocessing, os, re, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import mutagen
except ImportError:
    mutagen = None

EXT = {".mp3", ".wav", ".flac", ".m4a", ".aif", ".aiff", ".ogg"}
SEG = 75  # segundos analizados (zona central del track)
NOTES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJ = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]  # numero Camelot por tonica mayor
# Perfiles de tonalidad: Krumhansl-Kessler y Temperley (se promedian)
PROFILES = [
    (np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
     np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])),
    (np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
     np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])),
]


# ---------- lectura de audio ----------
def _ffmpeg(path, seg):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(f"No se pudo leer {Path(path).suffix}: instalá ffmpeg o convertí a MP3/WAV/FLAC")
    for ss in (30, 0):
        out = subprocess.run([exe, "-v", "error", "-ss", str(ss), "-t", str(seg), "-i", path,
                              "-f", "f32le", "-ac", "1", "-ar", "22050", "-"],
                             capture_output=True).stdout
        if len(out) > 4 * 22050 * 10:
            return np.frombuffer(out, np.float32).copy(), 22050
    raise RuntimeError("Audio muy corto o ilegible")


def load_segment(path):
    path = str(path)
    try:
        info = sf.info(path)
        sr, n = info.samplerate, min(info.frames, int(SEG * info.samplerate))
        y, _ = sf.read(path, start=max(0, (info.frames - n) // 2), frames=n, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        y, sr = _ffmpeg(path, SEG)
    if len(y) < sr * 10:
        raise RuntimeError("Audio muy corto")
    d = max(1, int(sr // 22050))
    if d > 1:
        y = y[: len(y) // d * d].reshape(-1, d).mean(axis=1)
        sr = sr / d
    return y / (np.abs(y).max() + 1e-9), sr


# ---------- analisis ----------
def stft_mag(x, n_fft, hop):
    win = np.hanning(n_fft).astype(np.float32)
    nf = 1 + (len(x) - n_fft) // hop
    out = np.empty((nf, n_fft // 2 + 1), np.float32)
    for s in range(0, nf, 512):
        e = min(nf, s + 512)
        idx = hop * np.arange(s, e)[:, None] + np.arange(n_fft)[None, :]
        out[s:e] = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    return out


def detect_bpm(x, sr):
    """Flujo espectral -> autocorrelacion -> ajuste fino por tren de pulsos (pasos de 0.05 BPM)."""
    n_fft, hop = 1024, max(1, int(round(sr / 172)))
    fps = sr / hop
    S = stft_mag(x, n_fft, hop)
    k0, k1 = int(30 * n_fft / sr), int(8000 * n_fft / sr)
    L = np.log1p(1000 * S[:, k0:k1] / (S.max() + 1e-9))
    flux = np.concatenate([[0], np.maximum(0, np.diff(L, axis=0)).sum(axis=1)])
    w = max(3, int(fps))
    env = flux - np.convolve(flux, np.ones(w) / w, mode="same")
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 70)
    n = len(env)
    f = np.fft.rfft(env, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: 4 * hi + 8]
    lags = np.arange(lo, hi + 1)
    # un tempo real repite en 1, 2, 3 y 4 pulsos; un falso (ej. 3/2) no -> se suman los multiplos
    sc = sum(wm * ac[m * lags] for m, wm in ((1, 1.0), (2, 0.8), (3, 0.6), (4, 0.5)))
    # leve preferencia por tempos de pista (centro 122 BPM, ancho 0.6 octavas)
    sc = np.maximum(sc, 0) * np.exp(-0.5 * (np.log2(60 * fps / lags / 122) / 0.6) ** 2)
    b0 = 60 * fps / lags[int(np.argmax(sc))]
    xs = np.arange(n)
    best, bpm = -1e18, b0
    for b in np.arange(b0 - 3, b0 + 3, 0.05):
        p = 60 * fps / b
        k = np.arange(int(n / p) - 1)
        pos = np.arange(0, p, 1.0)[:, None] + k[None, :] * p
        sc = np.interp(pos, xs, env).sum(axis=1).max()
        if sc > best:
            best, bpm = sc, b
    while bpm < 80: bpm *= 2
    while bpm > 175: bpm /= 2
    return round(float(bpm), 1)


def detect_key(x, sr):
    """Picos espectrales con frecuencia exacta + correccion de afinacion -> chroma -> perfiles."""
    n_fft, hop = 8192, 4096
    S = stft_mag(x, n_fft, hop)
    # filtro armonico: la mediana en el tiempo conserva notas sostenidas y borra bombos/hats (percusion)
    Sp = np.pad(S, ((4, 4), (0, 0)), mode="edge")
    A = np.median(np.lib.stride_tricks.sliding_window_view(Sp, 9, axis=0), axis=-1).mean(axis=0)
    fr = np.arange(len(A)) * sr / n_fft
    k = np.arange(1, len(A) - 1)
    pk = k[(A[k] > A[k - 1]) & (A[k] >= A[k + 1]) & (fr[k] >= 65) & (fr[k] <= 5000)]
    pk = pk[A[pk] > 0.03 * A[pk].max()]
    a, b, c = A[pk - 1], A[pk], A[pk + 1]
    freq = (pk + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * sr / n_fft
    m = 12 * np.log2(freq / 440) + 69
    w = np.sqrt(b)
    frac = m - np.round(m)
    tune = np.angle((w * np.exp(2j * np.pi * frac)).sum()) / (2 * np.pi)
    chroma = np.bincount(np.round(m - tune).astype(int) % 12, weights=w, minlength=12)
    res = []
    for t in range(12):
        cr = np.roll(chroma, -t)
        for minor in (0, 1):
            res.append((float(np.mean([np.corrcoef(cr, P[minor])[0, 1] for P in PROFILES])), t, minor))
    res.sort(reverse=True)
    (s1, t, minor), s2 = res[0], res[1][0]
    num = MAJ[(t + 3) % 12] if minor else MAJ[t]
    return dict(num=num, letter="A" if minor else "B", cam=f"{num}{'A' if minor else 'B'}",
                note=NOTES[t] + ("m" if minor else ""), kconf=round(s1 - s2, 3))


def analyze(path):
    x, sr = load_segment(path)
    return dict(bpm=detect_bpm(x, sr), **detect_key(x, sr))


def analyze_safe(path):
    """Version para procesos paralelos: devuelve (info, error)."""
    try:
        return analyze(path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def default_jobs():
    return max(1, (os.cpu_count() or 2) - 1)


# ---------- nombres, tags, mezcla ----------
def clean(stem):
    s = stem.replace("_", " ")
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\((official|video|audio|lyrics|hq|hd|free)[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"www\.\S+", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,2}[AB]\s*[-–]\s*\d{2,3}(\.\d)?\s*[-–]\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,3}\s*[-.]\s+(?=\D)", "", s)
    s = re.sub(r"[\\/:*?\"<>|]", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def read_tags(path):
    """(artista, titulo) de los tags ID3/Vorbis/MP4, o ('', '') si no hay."""
    if mutagen:
        try:
            t = mutagen.File(str(path), easy=True)
            return (t.get("artist") or [""])[0].strip(), (t.get("title") or [""])[0].strip()
        except Exception:
            pass
    return "", ""


def split_parts(path, tags, use_tags, order, bpm):
    """Devuelve (artista, titulo). Usa los tags; si faltan, parte el nombre del archivo.
    order 0: el archivo es 'Titulo - Artista (Mix)'   order 1: 'Artista - Titulo'."""
    artist, title = tags if use_tags else ("", "")
    if title:
        return clean(artist), clean(title)
    stem = clean(path.stem)
    m = re.search(r"\s+(\d{2,3})(?:\.\d)?$", stem)  # saca el BPM viejo del final si coincide
    if m and abs(int(m.group(1)) - bpm) <= 2:
        stem = stem[:m.start()].strip()
    if " - " not in stem:
        return "", stem
    p1, p2 = stem.split(" - ", 1)
    if order == 1:
        return p1, p2
    mix = re.search(r"\s*(\([^)]*\))\s*$", p2)  # el "(Original Mix)" vuelve al titulo
    if mix:
        p2, p1 = p2[:mix.start()].strip(), f"{p1} {mix.group(1)}"
    return p2, p1


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


def new_name(parts, a, ext, pattern, artist_first=False):
    artist, title = parts
    b, k = round(a["bpm"]), a["cam"]
    if artist_first:  # Artista - Tono - BPM - Titulo
        core = [f"{k} - {b} - {title}", f"{title} [{k} {b}]", f"{b} - {k} - {title}"][pattern]
        return (f"{artist} - {core}" if artist else core) + ext
    at = f"{artist} - {title}" if artist else title
    return [f"{k} - {b} - {at}", f"{at} [{k} {b}]", f"{b} - {k} - {at}"][pattern] + ext


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
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--write-tags", action="store_true")
    ap.add_argument("--notags", action="store_true")
    ap.add_argument("--artist-first", action="store_true", help="Artista - Tono - BPM - Titulo")
    ap.add_argument("--file-order", choices=["title-artist", "artist-title"], default="title-artist",
                    help="como se llaman tus archivos sin tags (por defecto: Titulo - Artista)")
    ap.add_argument("--undo")
    a = ap.parse_args()
    if a.undo:
        return undo(a.undo)
    if not a.folder:
        ap.error("falta la carpeta")
    root = Path(a.folder)
    files = sorted(p for p in (root.rglob("*") if a.recursive else root.glob("*"))
                   if p.is_file() and p.suffix.lower() in EXT)
    if not files:
        sys.exit("No encontré audios en esa carpeta.")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for i, (p, (info, err)) in enumerate(zip(files, ex.map(analyze_safe, map(str, files))), 1):
            if err:
                print(f"[{i}/{len(files)}] ✗ {p.name} ({err})")
                continue
            parts = split_parts(p, read_tags(p), not a.notags, int(a.file_order == "artist-title"), info["bpm"])
            target = p.with_name(new_name(parts, info, p.suffix, a.pattern, a.artist_first))
            rows.append((p, target, info))
            print(f"[{i}/{len(files)}] {info['cam']:>3} {info['note']:<4} {info['bpm']:>6} BPM  "
                  f"{style(info['bpm'])}\n        {p.name}\n     -> {target.name}")

    if len(rows) > 1:  # orden de mezcla sugerido (greedy por compatibilidad)
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
    multiprocessing.freeze_support()
    main()
