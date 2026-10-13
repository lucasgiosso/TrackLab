#!/usr/bin/env python3
"""Track Lab con ventana: elegí una carpeta, revisá la vista previa y renombrá."""
import csv, queue, threading, tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import track_lab as core

PATTERNS = ["Tono - BPM - Título", "Título [Tono BPM]", "BPM - Tono - Título"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Track Lab")
        self.geometry("1020x580")
        self.rows, self.folder, self.q = [], None, queue.Queue()

        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        self.b_pick = ttk.Button(top, text="Elegir carpeta…", command=self.pick); self.b_pick.pack(side="left")
        self.lbl = ttk.Label(top, text="Ninguna carpeta elegida"); self.lbl.pack(side="left", padx=10)
        ttk.Button(top, text="Deshacer…", command=self.undo).pack(side="right")

        opt = ttk.Frame(self, padding=(10, 0)); opt.pack(fill="x")
        self.pat = ttk.Combobox(opt, state="readonly", width=24, values=PATTERNS)
        self.pat.current(0); self.pat.pack(side="left")
        self.pat.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.rec, self.tags, self.wtags = tk.BooleanVar(), tk.BooleanVar(value=True), tk.BooleanVar()
        ttk.Checkbutton(opt, text="Subcarpetas", variable=self.rec).pack(side="left", padx=8)
        ttk.Checkbutton(opt, text="Usar tags ID3", variable=self.tags).pack(side="left", padx=8)
        ttk.Checkbutton(opt, text="Escribir BPM y tono en el archivo", variable=self.wtags).pack(side="left", padx=8)

        cols = (("orig", "Original", 280), ("tono", "Tono", 70), ("bpm", "BPM", 60),
                ("estilo", "Estilo probable", 170), ("nuevo", "Nombre nuevo", 380))
        self.tree = ttk.Treeview(self, columns=[c[0] for c in cols], show="headings")
        for k, t, w in cols:
            self.tree.heading(k, text=t); self.tree.column(k, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        bot = ttk.Frame(self, padding=(10, 0, 10, 10)); bot.pack(fill="x")
        self.prog = ttk.Progressbar(bot, length=300); self.prog.pack(side="left")
        self.status = ttk.Label(bot, text=""); self.status.pack(side="left", padx=10)
        self.b_apply = ttk.Button(bot, text="Renombrar", command=self.apply, state="disabled")
        self.b_apply.pack(side="right")
        self.after(100, self.poll)

    def pick(self):
        d = filedialog.askdirectory()
        if not d: return
        self.folder = Path(d); self.lbl.config(text=str(self.folder))
        files = sorted(p for p in (self.folder.rglob("*") if self.rec.get() else self.folder.glob("*"))
                       if p.suffix.lower() in core.EXT)
        if not files:
            messagebox.showinfo("Track Lab", "No encontré audios en esa carpeta."); return
        self.rows.clear(); self.tree.delete(*self.tree.get_children())
        self.prog.config(maximum=len(files), value=0)
        self.b_pick.state(["disabled"]); self.b_apply.state(["disabled"])
        threading.Thread(target=self.work, args=(files,), daemon=True).start()

    def work(self, files):
        use_tags = self.tags.get()
        for p in files:
            try:
                self.q.put(("row", p, core.title_from(p, use_tags), core.analyze(p)))
            except Exception as e:
                self.q.put(("err", p, str(e)))
        self.q.put(("done",))

    def poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "row":
                    self.rows.append(m[1:]); self.refresh(); self.prog.step(1)
                    self.status.config(text=f"Analizados: {len(self.rows)}")
                elif m[0] == "err":
                    self.prog.step(1)
                else:
                    self.b_pick.state(["!disabled"])
                    self.b_apply.state(["!disabled" if self.rows else "disabled"])
                    self.status.config(text=f"Listo: {len(self.rows)} tracks")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def name_for(self, r):
        p, title, info = r
        return core.new_name(title, info, p.suffix, self.pat.current())

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.rows:
            p, _, info = r
            self.tree.insert("", "end", values=(p.name, f"{info['cam']} {info['note']}", round(info["bpm"]),
                                                core.style(info["bpm"]), self.name_for(r)))

    def apply(self):
        if not self.rows: return
        if not messagebox.askyesno("Renombrar", f"Se van a renombrar {len(self.rows)} archivos. ¿Seguir?"):
            return
        ok = skipped = 0
        with open(self.folder / "track_lab_log.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in self.rows:
                p, _, info = r
                t = p.with_name(self.name_for(r))
                try:
                    if self.wtags.get() and core.mutagen: core.write_tags(p, info)
                    if t != p and not t.exists():
                        p.rename(t); w.writerow([str(p), str(t)]); ok += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
        self.rows.clear(); self.tree.delete(*self.tree.get_children()); self.b_apply.state(["disabled"])
        messagebox.showinfo("Track Lab", f"Renombrados: {ok}\nOmitidos: {skipped}\n\nLog para deshacer: track_lab_log.csv")

    def undo(self):
        log = filedialog.askopenfilename(title="Elegí track_lab_log.csv", filetypes=[("CSV", "*.csv")])
        if not log: return
        n = 0
        with open(log, newline="", encoding="utf-8") as f:
            for old, new in csv.reader(f):
                if Path(new).exists() and not Path(old).exists():
                    Path(new).rename(old); n += 1
        messagebox.showinfo("Track Lab", f"Restaurados: {n}")


if __name__ == "__main__":
    App().mainloop()
