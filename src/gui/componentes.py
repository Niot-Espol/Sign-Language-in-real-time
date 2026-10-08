"""Componentes reutilizables de la interfaz."""

import tkinter as tk


class PanelResultado(tk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg="#172033", padx=16, pady=12)
        self._texto = tk.StringVar(value="Esperando una mano...")
        tk.Label(
            self,
            textvariable=self._texto,
            bg="#172033",
            fg="white",
            font=("Segoe UI", 20, "bold"),
        ).pack()

    def actualizar(self, etiqueta: str, confianza: float | None = None) -> None:
        texto = etiqueta
        if confianza is not None:
            texto += f" ({confianza:.0%})"
        self._texto.set(texto)

