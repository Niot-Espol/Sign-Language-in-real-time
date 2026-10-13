"""Panel sencillo para que el equipo inicie capturas sin memorizar comandos."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from src.data_collection.catalogo import cargar_catalogo


class PanelCaptura(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Captura del dataset de señas")
        self.geometry("720x650")
        self.catalogo = cargar_catalogo()

        ttk.Label(self, text="Integrante").pack(anchor="w", padx=20, pady=(20, 4))
        self.integrante = ttk.Entry(self)
        self.integrante.pack(fill="x", padx=20)

        ttk.Label(self, text="Selecciona las palabras asignadas").pack(
            anchor="w", padx=20, pady=(16, 4)
        )
        self.lista = tk.Listbox(self, selectmode="extended", height=20, exportselection=False)
        for sena in self.catalogo:
            referencia = "✓" if (Path("docs/referencias") / f"{sena.slug}.jpg").exists() else "○"
            config = f"{sena.tipo}, manos={sena.manos_requeridas or '?'}"
            self.lista.insert("end", f"{referencia} {sena.etiqueta}  [{config}]")
        self.lista.pack(fill="both", expand=True, padx=20)

        opciones = ttk.Frame(self)
        opciones.pack(fill="x", padx=20, pady=12)
        ttk.Label(opciones, text="Muestras por palabra (80–100):").pack(side="left")
        self.objetivo = tk.IntVar(value=100)
        ttk.Spinbox(opciones, from_=80, to=100, textvariable=self.objetivo, width=6).pack(
            side="left", padx=8
        )

        botones = ttk.Frame(self)
        botones.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Button(botones, text="Capturar frames", command=self._frames).pack(
            side="left", expand=True, fill="x", padx=(0, 6)
        )
        ttk.Button(botones, text="Capturar secuencia dinámica", command=self._secuencia).pack(
            side="left", expand=True, fill="x", padx=(6, 0)
        )

        ttk.Label(
            self,
            text="✓ referencia disponible  ·  ○ falta agregar JPG en docs/referencias",
        ).pack(pady=(0, 12))

    def _seleccion(self) -> tuple[str, list[str]] | None:
        integrante = self.integrante.get().strip()
        indices = self.lista.curselection()
        if not integrante or not indices:
            messagebox.showwarning("Faltan datos", "Escribe el integrante y selecciona al menos una palabra.")
            return None
        return integrante, [self.catalogo[i].etiqueta for i in indices]

    def _frames(self) -> None:
        seleccion = self._seleccion()
        if seleccion is None:
            return
        integrante, palabras = seleccion
        comando = [
            sys.executable, "-m", "src.data_collection.recolector_csv", integrante,
            "--objetivo", str(self.objetivo.get()), "--palabras", *palabras,
        ]
        subprocess.Popen(comando, cwd=Path.cwd())

    def _secuencia(self) -> None:
        seleccion = self._seleccion()
        if seleccion is None:
            return
        integrante, palabras = seleccion
        if len(palabras) != 1:
            messagebox.showwarning("Selección", "Para secuencias selecciona exactamente una palabra.")
            return
        subprocess.Popen(
            [sys.executable, "-m", "src.data_collection.recolector_secuencias", integrante, palabras[0]],
            cwd=Path.cwd(),
        )


if __name__ == "__main__":
    PanelCaptura().mainloop()

